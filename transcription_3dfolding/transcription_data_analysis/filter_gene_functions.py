import pandas as pd
import numpy as np
import bbi
import bioframe as bf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from functools import partial
import multiprocessing as mp
import warnings


def get_tss_gene_intervals(tss_df):
    """
    Input: a .gtf file containing the chr, start, end
    corresponding to the TSS for the transcripts ready from a
    genomic .gtf format annotation file.
    Output: a dataframe in bioframe format with a single TSS
    per gene, with non-autosomes removed.
    """

    # cleaning out less-well defined chromosome numbers
    tss_df = tss_df.loc[False == (tss_df["seqname"].str.contains("NT_"))]
    tss_df = tss_df.loc[False == (tss_df["seqname"].str.contains("MT"))]

    # paste 'chr' to all chromosome names
    tss_df["seqname"] = tss_df["seqname"]

    # rename column to chrom to match bedframe/bioframe format
    tss_df = tss_df.rename(columns={"seqname": "chrom"})

    # Removing pseudo chromosomes
    tss_df = tss_df.loc[False == (tss_df["chrom"].str.contains("chrGL"))]
    tss_df = tss_df.loc[False == (tss_df["chrom"].str.contains("chrJH"))]
    tss_df = tss_df.loc[False == (tss_df["chrom"].str.contains("chrY"))]
    tss_df = tss_df.loc[False == (tss_df["chrom"].str.contains("chrM"))]
    tss_df = tss_df.loc[True == tss_df["chrom"].str.contains("chr")]

    # drop duplicate TSSes
    return tss_df[["gene_id", "chrom", "start", "end"]].drop_duplicates(["gene_id"])


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

def get_enhancer_bioframe(enhancer_file):
    """
    Reads enhancer .bed file with no header and returns
    a clean bioferame, with only enhancers one numerical chromosomes.
    """
    
    enhancers = bf.read_table(enhancer_file).rename(
        columns={0: 'chrom',  1: 'start', 2: 'end'}
    )
    
    # filter by only numerical enhancers
    enhancers = enhancers[enhancers.chrom.str.match('^chr\d+$')]
    enhancers = bf.sanitize_bedframe(enhancers)
    
    return enhancers

def label_closest_enhancer(df, enhancer_file, enhancer_set):
    """ Appends a column with the distance from each gene 
    in df to the closest enhancer.
    
    Returns
    --------
    df_out : pd.DataFrame
        DataFrame with [enhancer_set]_distance column
        
    """
    
    df_out = df.copy()
    enhancers = get_enhancer_bioframe(enhancer_file)
    df_out[enhancer_set+'_distance'] = bf.closest(
        df, enhancers, suffixes=('','_enh')
    )['distance']
    
    return df_out


def tad_windows_from_boundaries(insulation_boundaries, cooler):
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
    insulation_boudnaries: df containing insulation boundaries and original cooler file.
    cooler: 
    
    Returns:
    --------
    tad_df: A bioframe containing genomic intervals that represent the TADs,
        or regions between these boundaries. Starts from 
    
    """

    tad_df = pd.DataFrame(columns=['chrom', 'start', 'end'])

    for chrom in insulation_boundaries.chrom.unique():

        chrom_boundaries = insulation_boundaries[insulation_boundaries['chrom'] == chrom]
        size = len(chrom_boundaries)
        TAD_starts = np.empty([size+1])
        TAD_starts[0] = 0
        TAD_starts[1:] = chrom_boundaries['end']

        TAD_ends = np.empty([size+1])
        TAD_ends[-1] = cooler.chromsizes[chrom]
        TAD_ends[0:-1] = chrom_boundaries['start']

        tmp = pd.DataFrame({'chrom' : chrom,
                            'start' : TAD_starts,
                            'end' : TAD_ends}
                          )
        
        tad_df = tad_df.append(tmp, ignore_index=True)
    
    return bf.sanitize_bedframe(tad_df)