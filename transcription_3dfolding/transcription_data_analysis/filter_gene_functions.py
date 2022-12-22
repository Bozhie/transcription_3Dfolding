import pandas as pd
import numpy as np
import bbi
import bioframe as bf
from gtfparse import read_gtf

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from functools import partial
import multiprocessing as mp
import warnings


####################
#
# Project File locations
#
####################
default_mm10_gtf = ( '/project/fudenber_735/collaborations/karissa_2022/old/RNAseq/' +
                    'STAR_Gencode_alignment/tss_annotions_gencode.vM23.primary_assembly.gtf'
                   )

default_project = ('/project/fudenber_735/collaborations/' +
                   'karissa_2022/20220812_EA18-1_RNAseq-Analysis_forGeoff/'
                  )

day1_sig_results = (default_project +
                    'EA18.1_ESC_1d-depletion_DESeq2/' + 
                    '20220817_EA18-1_resSig_ESC_1d-depletion.csv'
                   )
raw_counts = default_project+'20220816_featureCounts.csv'

vst_norm_counts = (default_project +
                   'EA18.1_ESC_1d-depletion_DESeq2/' +
                   '20220817_EA18-1_ESC-1d_sf-normalized_vst-transformed.csv'
                  )


def load_tss_df(gtf=default_mm10_gtf,
                rna_tsv=day1_sig_results,
                chrom_keep='autosomes',
                counts_tables={'raw_counts_name' : raw_counts,
                                'vst_counts' : vst_norm_counts,
                              },
                counts_usage={'raw_counts_name' : 'append_name',
                              'vst_counts' : 'wt_avg'
                             },
                cutoff=6,
                cutoff_col='avg_vst_counts',
                wt_samples=['KHRNA1', 'KHRNA7', 'KHRNA13', 'KHRNA22', 'KHRNA23', 'KHRNA50']
               ):
    """
    Processes rna_tsv and returns a dataframe. Performs data manipulations 
    to annotate gene names and join additional features from experiment files.
    
    gtf: string for filename of .gtf file for this genome.
    rna_tsv: string filename of RNA seq results
    chrom_keep: ['autosomes', 'chromosomes'] defines which genes to keep based
        on chromosome mapping.
    counts_tables: {string name : string for filename} of additional rna-seq counts data tables
    counts_usage: {string name : string usage_setting} describing what data manipulations to apply
                   'append_name' gene_id from raw counts to re-index rna_tsv with full dataset
                   'wt_avg' takes average of wt samples and appends to rna_tsv
    cutoff: int cutoff value for filtering rows that fall below threshold
    cutoff_col: column in rna_tsv to compare to cutoff
    wt_samples: [list of sample names] that designate columns in counts_tables 
                only relevant if using 'wt_avg'
    ------
    Returns

    rna_tsv
    """
    
    rna_df = pd.read_csv(rna_tsv)
    
    for name, file in counts_tables.items():

        if counts_usage[name] == 'append_name':
            # add feature counts information to label genes not in the significant results table
            feat_df = pd.read_csv(file)
            rna_df = rna_df.merge(feat_df['Geneid'], how='outer')
            
        elif counts_usage[name] == 'wt_avg':
            counts_df = pd.read_csv(file).rename(
                                            columns={'Unnamed: 0' : 'Geneid'}
                                            ).astype({'Geneid': 'object'})
            
            counts_df['avg_'+name] = counts_df[wt_samples].mean(axis='columns')
            rna_df = rna_df.merge(counts_df[['Geneid', 'avg_'+name]], on='Geneid', how='outer')
            rna_df['avg_'+name].fillna(0, inplace=True)
    
    gtf_df = read_gtf(gtf)
    tss_intervals = get_tss_gene_intervals(gtf_df)
    tss_intervals['tss'] = tss_intervals['start'].copy()

    tss_df = tss_intervals.merge(rna_df.copy(),  how='left',
                left_on='gene_id', right_on='Geneid')
    
    if cutoff is not None:
        cut = (tss_df[cutoff_col] > cutoff)
        tss_df = tss_df[cut]
    
    return tss_df.reset_index(drop=True)


def get_tss_gene_intervals(
    tss_df, 
    return_cols=["gene_id", "chrom", "start", "end", "strand"],
    chrom_keep='autosomes',
    ):
    """
    Input: a .gtf file containing the chr, start, end
    corresponding to the TSS for the transcripts ready from a
    genomic .gtf format annotation file.
    Output: a dataframe in bioframe format with a single TSS
    per gene, with non-autosomes removed.
    """
    
    # rename column to chrom to match bedframe/bioframe format
    tss_df = tss_df.rename(columns={"seqname": "chrom"})

    # Removing pseudo chromosomes
    if chrom_keep == 'autosomes':
        tss_df = bioframe_clean_autosomes(tss_df)
    elif chrom_keep == 'chromosomes':
        tss_df = bioframe_clean_chromosomes(tss_df)

    # drop duplicate TSSes
    return tss_df[return_cols].drop_duplicates(["gene_id"])



def label_DE_status(
    df,
    significance_col="padj",
    significance_threshold=0.05,
    fold_change_col="log2FoldChange",
    fold_change_threshold=0,
    DE_status_column="DE_status",
):
    """
    Appends a column with differential expression status to a DataFrame with
    padj and log2FoldChange columns

    Returns
    --------
    df_out : pd.DataFrame
        DataFrame with DE_status_column

    """
    df_out = df.copy()
    df_out[DE_status_column] = "nonsig"
    sig_inds = df[significance_col] < significance_threshold
    down_sig_inds = (df[fold_change_col] < fold_change_threshold) * sig_inds
    up_sig_inds = (df[fold_change_col] > fold_change_threshold) * sig_inds
    df_out.loc[up_sig_inds, DE_status_column] = "up"
    df_out.loc[down_sig_inds, DE_status_column] = "down"

    return df_out


def label_quantiles(
    df,
    quantile_label_col="DE_status",
    quantile_value_col="avg_vst_counts",
    quantile_array=[0.0, 0.5, 0.75, 0.95, 1.0],
    label_subset="nonsig",
):
    """
    Appends a quantile label to values in a column matching "label_subset".
    Quantiles are derived from the full distribution of values in quantile_value_col
    Quantile_array contains the borders of values for binning.
    Returns
    --------
    df_out : pd.DataFrame
        DataFrame where quantile labels have been appended to strings in quantile_label_col.
    Notes
    ------
    Quantile label col must be str or object.
    """
    if quantile_array[1] < 0.3:
        warnings.warn(
            (
                "The lowest quantile cut-off might be too low, "
                "leading to non-unique bin edges if there are "
                "excessive zeros (or lowest value) within bottom range."
            )
        )

    df_out = df.copy()
    q_strs = [str(num) for num in quantile_array]
    q_strs = [s + "-" + e for s, e in zip(q_strs[:-1], q_strs[1:])]
    quantile_labels = pd.qcut(
        df[quantile_value_col], quantile_array, labels=q_strs
    ).astype(str)
    if label_subset:
        inds = df[quantile_label_col] == label_subset
        df_out.loc[inds, quantile_label_col] = (
            df_out.loc[inds, quantile_label_col] + "_" + quantile_labels.loc[inds]
        )
    else:
        df_out.loc[quantile_label_col] = (
            df_out.loc[quantile_label_col] + "_" + quantile_labels
        )
    return df_out
 
def bioframe_clean_chromosomes(frame):
    """
    Takes a dataframe or bioframe and returns
    a sanitized bioframe with only the intervals on autosomes.
    """
    
    frame = frame[frame.chrom.str.match('^chr[\dXY]+$')]
    frame = bf.sanitize_bedframe(frame)
        
    return frame

def bioframe_clean_autosomes(frame):
    """
    Takes a dataframe or bioframe and returns
    a sanitized bioframe with only the intervals on autosomes.
    """
    
    frame = frame[frame.chrom.str.match('^chr\d+$')]
    frame = bf.sanitize_bedframe(frame)
        
    return frame

def get_enhancer_bioframe(enhancer_file):
    """
    Reads enhancer .bed file with no header and returns
    a clean bioferame, with only enhancers one numerical chromosomes.
    """
    enhancers = get_peak_bioframe(enhancer_file, schema='bed3')
    
    return enhancers

def get_peak_bioframe(peak_file,
                        schema='bed',
                        column_names=None):
    """
    Reads a .bed file containing called peaks, 
    with no header but columns: chrom, start, end, peak_name, score
    into a bioframe
    """
    
    peaks = bf.read_table(peak_file, schema=schema)
    if column_names is not None:
        peaks.rename(column_names)
    
    return bioframe_clean_autosomes(peaks)

def label_closest_peak(df, feature_intervals, feature_name):
    """ Appends a column with the distance from each gene 
    in df to the closest feature in feature_intervals.
    
    Returns
    --------
    df_out : pd.DataFrame
        DataFrame with [feature_name]_distance column
        
    """
    
    df_out = df.copy()
    df_out[feature_name+'_distance'] = bf.closest(
        df, feature_intervals, suffixes=('','_'+feature_name)
    )['distance']
    
    return df_out


def windows_from_boundaries(boundaries, chromsizes, take_midpoint=False):
    """
    Using a set of boundaries, builds the set of genomic interval 'windows'
        that represent the regions separated by these boundaries. 
    
    Parameters:
    -----------
    boundaries: df containing boundaries of some feature, defined by genomic
        intervals (chrom, start, end).
    chromsizes: a dictionary with the chromosome sizes for the final window boundaries.
    take_midpoint: when False, the windows are the regions outside the boundaries. 
        When true, the windows start and end at the midpoint, inside each boundary.
    
    Returns:
    --------
    window_df: A bioframe containing genomic intervals that represent the
        regions between these boundaries.
    """

    window_df = pd.DataFrame(columns=['chrom', 'start', 'end'])

    for chrom in boundaries.chrom.unique():

        chrom_boundaries = boundaries[boundaries['chrom'] == chrom]
        if take_midpoint:
            mid = (chrom_boundaries['end'] + chrom_boundaries['start'])/2
        size = len(chrom_boundaries)
        
        window_starts = np.empty([size+1])
        window_starts[0] = 0
        if take_midpoint:
            window_starts[1:] = mid + 1
        else:
            window_starts[1:] = chrom_boundaries['end']
        
        window_ends = np.empty([size+1])
        window_ends[-1] = chromsizes[chrom]
        if take_midpoint:
            window_ends[0:-1] = mid
        else:
            window_ends[0:-1] = chrom_boundaries['start']

        tmp = pd.DataFrame({'chrom' : chrom,
                            'start' : window_starts,
                            'end' : window_ends}
                          )
        
        window_df = window_df.append(tmp, ignore_index=True)
    
    return bf.sanitize_bedframe(window_df)


def extract_chrom_sizes_from_insulation(insulation_table):
    """ Returns a dictionary of the chromosome sizes from an insulation table. """
    
    chrom_sizes = {}
    for chrom in insulation_table['chrom'].unique():
        size = insulation_table[insulation_table['chrom'] == chrom][-1:]['end']
        chrom_sizes[chrom] = size
    return chrom_sizes

def tad_windows_from_boundaries(insulation_table, take_midpoint=True):
    """
    Using a set of insulation boundaries, builds a set of genomic intervals 
        that represent the TADs, or regions between these boundaries. 
        
        Each chromosome has the following set of TAD intervals:
            - first TAD interval begins at start of chromosome and ends at 
                the beginning of the next boundary.
            - middle TAD intervals start at end of previous insulation boundary,
                and end at beginning of the next one.
            - last TAD interval starts at end of last insulation boundary
                and ends at the end of the chromosome.
    
    Parameters:
    -----------
    boundaries: df containing boundaries of some feature, with 'start' and 'ends'.
    chromsizes: a dictionary with the chromosome sizes for the final window boundaries.
    take_midpoint: when False, the windows are the regions outside the 
    
    Returns:
    --------
    tad_df: A bioframe containing genomic intervals that represent the TADs,
        or regions between these boundaries. Starts from 
    
    """
    
    # Get chrom_sizes
    chrom_sizes = extract_chrom_sizes_from_insulation(insulation_table)
    # Take only boundaries called strong from this table
    insulation_boundaries = insulation_table.query('is_boundary_200000 == True')
    # Dropping any coordinates with chrX
    insulation_boundaries = insulation_boundaries[~insulation_boundaries.chrom.isin(['chrX'])]
    
    tad_df = windows_from_boundaries(insulation_boundaries, chrom_sizes, take_midpoint=take_midpoint)
    return tad_df

# todo: move to helper scripts. 

def extract_features_around_TSS(tss_df, feature_df, window_size=int(1e6)):
    """
    Generate a dataframe of distances to all features within a window around a TSS.
    used as input for histogram of element density within a window.
    
    Parameters:
    -----------
    tss_df: pandas dataframe
        Dataframe of TSS positions
    feature_df : pd.DataFrame
        DataFrame of features in bed format
    window_size : int
        maximum distance considered
        
    Returns
    --------
    tss_feature_df : pd.DataFrame
        Dataframe distances between each TSS and features within the window
        
    """
    
    tss_feature_df = bf.overlap(
        bf.expand(tss_df, pad= window_size),
        feature_df,
        how = 'left'
    )
    mids = .5*(tss_feature_df['start_']+tss_feature_df['end_']).values 
    dists = np.abs( (mids- tss_feature_df['tss'].values))
    tss_feature_df['dist'] = dists
    
    return tss_feature_df

def mask_tss_proximal_features( feature_df, tss_df, window_size=int(5e3)) :
    """
    Drop features within window_size of a tss.
      
    Parameters:
    -----------

    feature_df : pd.DataFrame
        DataFrame of features in bed format
    tss_df: pandas dataframe
        Dataframe of TSS positions        
    window_size : int
        masking distance around each TSS

    Returns
    --------
    feature_df_pruned : pd.DataFrame
        filtered dataframe
        
    """
    feature_df_pruned = bf.setdiff(feature_df, 
        bf.expand(tss_df, pad= window_size))
    return feature_df_pruned


def mask_gene_body_features( feature_df, tss_df, extend_gene_bp=int(1e3)):
    """
    Drop features within gene bodies
    
    Parameters:
    -----------

    feature_df : pd.DataFrame
        DataFrame of features in bed format
    tss_df: pandas dataframe
        Dataframe of TSS positions, also has keys start_gene and stop_gene 
        used for dropping gene-body overlaps.
        
    Returns
    --------
    feature_df_pruned : pd.DataFrame
        filtered dataframe
        
    """
    feature_df_pruned = bf.setdiff(
                            feature_df,
                            bf.expand(tss_df, pad= extend_gene_bp, 
                                cols=['chrom','start_gene','end_gene']), 
                            cols2=['chrom','start_gene','end_gene']
                        )
    return feature_df_pruned

def split_by_proximal_feature(tss_df, feature_df, window_size=int(5e3), contains_feature=True) :
    """
    Filter tss_df for genes containing a feature within 
    window_size of a tss.
      
    Parameters:
    -----------

    tss_df: pandas dataframe
        Dataframe of TSS positions        
    feature_df : pd.DataFrame
        DataFrame of features in bed format
    window_size : int
        masking distance around each TSS
    contains_feature : True/False
        indicates whether the desired df should include
        the feature, when true, or exclude it when false.

    Returns
    --------
    feature_df_pruned : pd.DataFrame
        filtered dataframe
        
    """
    
    tss_df_masked = mask_tss_proximal_features(tss_df, 
                                              feature_df,
                                              window_size=window_size)
    if contains_feature:
        filtered_df = bf.setdiff(tss_df, 
                            bf.expand(tss_df_masked, pad=1))
    else:
        filtered_df = tss_df_masked
        
    return filtered_df