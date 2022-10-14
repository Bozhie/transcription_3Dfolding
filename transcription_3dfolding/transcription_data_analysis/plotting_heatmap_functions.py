import pandas as pd
import numpy as np
import bbi
import bioframe as bf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from functools import partial
import multiprocessing as mp

def generate_signal_matrix(interval_df,
                           chip_seq_file, 
                           columns=['chrom', 'start', 'end'],
                           window_size=1000, 
                           window_type='extend', 
                           nbins=40):
    """
    Uses pybbi to measure signal over a set of input intervals. 
    Returns a matrix [n_intervals x nbins] of the average of ChIP signal over 
    each bin in the matrix.
    
    Parameters:
    -----------
    interval_df: pandas dataframe that has the list of intervals (usually, 
                set of genes)
    chip_seq_file: filepath to ChIP-seq file of type bigWig or bigBed
    columns: the columns in interval_df that define the interval
    window_size: the distance of the 
    window_type: 'extend': window_size defines padding length added to ends
                 'centered': window_size extends out from each side of middle
    nbins: number of bins the window is divided into.
    
    Returns:
    --------
    matrix: np.array of shape [n_intervals x nbins]
            rows correspond to intervals in interval_df and ChIP signal measured
            across column bins.
    
    """
    
    intervals = bf.from_any(interval_df[columns])
    intervals = intervals.rename(columns = {columns[0]: 'chrom',
                                                       columns[1]: 'start',
                                                       columns[2]: 'end'})
    intervals = bf.sanitize_bedframe(intervals)
    
    if window_type == 'extend':
        shapes = pd.Series(intervals['start'] - intervals['end']).nunique()
        msg = """The size of intervals should be equal to perform stackup.
                 Try window_type: 'centered'"""
        assert shapes == 1, msg
        
        intervals = bf.expand(intervals, pad=window_size)
        
    if window_type == 'centered':
        intervals = bf.expand(bf.expand(expanded, scale=0), pad = 1000)
    
    with bbi.open(chip_seq_file) as f:
        matrix = f.stackup(intervals['chrom'],
                           intervals['start'],
                           intervals['end'], bins=nbins)
        
    return matrix


def matgen(df, columns, window_size, window_type, nbins, cond, chip_file):
        """
        Helper function for parallelizing stackup generation over bigWigs.
        Feeds the kargs into generate_signal_matrix.
        """
        return (cond, generate_signal_matrix(df,  chip_file, columns, window_size, window_type, nbins))

def generate_signal_matrix_parallel(df, chip_seq_condition_dict, max_procs=8, 
                                    columns=['chrom', 'start', 'end'],
                                    window_size=1000, 
                                    window_type='extend', 
                                    nbins=40
                                   ):
    """
    Parallelizes stackup generation over bigWigs.
    
    Returns
    -------
    chip_seq_condition_matrix_dict : dict of np.arrays
    
    """
    
    chip_seq_condition_matrix_dict ={}
    with mp.Pool(np.min([len(chip_seq_condition_dict), max_procs])) as pool:
        chip_seq_condition_matrix_dict = dict(pool.starmap(
            partial(matgen, df, columns, window_size, window_type, nbins), list(chip_seq_condition_dict.items())))
        
    return chip_seq_condition_matrix_dict
    
def plot_avg_signal(DE_results_df, 
                    stackup_matrix, 
                    plot_title, 
                    ax=None, 
                    DE_value_col='log2Fold_Change',
                    agg_key='DE_status', 
                    agg_categories=['up', 'down', 'nonsig'], 
                    color_categories=['r', 'b', 'k'], 
                    window_size=1000, 
                    nbins=40):
    """
    Plots average signal of each category defined in DE_results_df['agg_key'], 
    sorted by differential expression
    
    Parameters:
    -----------
    DE_results_df: pandas dataframe that has the list of intervals (usually, 
                   set of genes) and the categories labeled in 'agg_key'
    plot_title: title or this plot
    ax: the axis for plotting this heatmap.
    DE_value_col: column in DE_results_df that has measure of differential 
                  expression, for sorting in descending order
    sort_by_DE: True/False sort the intervals by their change in expression
    agg_key: column in DE_results_df containing category labels
    agg_categories: the category label values in DE_results_df[agg_key]
    window_size: the size of the interval windows (end - start)
    nbins: the number of bins the interval window is broken up into.
    
    Returns:
    --------
    
    """
    
    if ax == None:
        ax = plt.subplot()
    
    for category, color in zip(agg_categories, color_categories):
        
        cat_ix = np.where(DE_results_df[agg_key] == category)
        cat_matrix = stackup_matrix[cat_ix]
        
        ax.plot(np.nanmean(cat_matrix, axis=0), color = color, label=category)
        
    ticks = np.arange(0, (window_size*2)+1, (window_size/2))-(window_size*2)//2
    ax.set(xticks=np.arange(0, nbins+1, 10),
    xticklabels=ticks,
    xlabel='Distance from boundary (bp)',
    ylabel='ChIP-Seq mean fold change over input')
    plt.legend()
    ax.set_title(plot_title)
    
    
def plot_binned_signal_heatmap(DE_results_df, 
                               stackup_matrix, 
                               plot_title, 
                               ax=None,
                               cax=None,
                               DE_value_col='log2FoldChange', 
                               sort_by_DE=True, 
                               agg_key='DE_status', 
                               agg_categories=['up', 'down'],
                               window_size=1000, 
                               nbins=40):
    """
    Plot heatmap of ChIP signal in the matrix window for each interval 
    (e.g. DE gene), grouped by category.
    
    Parameters:
    -----------
    DE_results_df: pandas dataframe that has the list of intervals (usually, 
                   set of genes) and the categories labeled in 'agg_key'
    plot_title: title or this plot
    ax: the axis for plotting this heatmap.
    cax: the axis for plotting the colorbar.
    DE_value_col: column in DE_results_df with measure of differential 
                  expression, for sorting.
    sort_by_DE: True/False sort the intervals by their change in expression in
                descending order.
    agg_key: column in DE_results_df containing category labels
    agg_categories: the category label values in DE_results_df[agg_key]
    window_size: the size of the interval windows (end - start)
    nbins: the number of bins the interval window is broken up into.
    
    Returns:
    --------
    Greyscale heatmap visualization of ChIP signal in stackup_matrix, 
    grouped and sorted.
    
    """
    
    # if include_category_map will need two sub-plots
    if ax == None:
        ax = plt.subplot()
        
        
    # build the heatmap matrices 
    ordered_heatmap = []
    ordered_values = []
        
    # rearrange the matrix for plotting everything together
    for cat in agg_categories:
        
        # get first category
        cat_ix = np.where(DE_results_df[agg_key] == cat)
        sub_matrix = stackup_matrix[cat_ix]
        sub_results = DE_results_df.iloc[cat_ix]
        
         # sort according to DE_value in descending order
        if sort_by_DE:
            ordering = (-sub_results[DE_value_col]).argsort()
            ordered_heatmap.append(sub_matrix[ordering])
            ordered_values.append(sub_results.iloc[ordering][DE_value_col])
        else:
            ordered_heatmap.append(sub_matrix)
            ordered_values.append(sub_results[DE_val_col])
    
    # Normalize the greyscale centered around the 95th percentile of ChIP signal
    bottom = 0.
    center = np.percentile(stackup_matrix, 95)
    top = np.percentile(stackup_matrix, 99)
    norm = colors.TwoSlopeNorm(vmin=bottom, vcenter=center, vmax=top)
    
    hm = ax.imshow(np.vstack(ordered_heatmap), cmap='gray_r', norm=norm, aspect='auto')
    if cax == None:
        cb = plt.colorbar(hm, ax=ax, extend='max', orientation='vertical')
    else:
        cb = plt.colorbar(hm, cax=cax, extend='max', orientation='vertical', pad=5.0)
    cb.set_ticks(np.percentile(stackup_matrix, [0, 95, 98, 99, 100]))
    cb.set_label('ChIP signal', rotation=270)

    ax.set(xticks=np.arange(0, nbins+1, 10),
    xticklabels=(np.arange(0, (window_size*2)+1, (window_size/2))-(window_size*2)//2),
    xlabel='Distance from boundary (bp)')
    ax.set_title(plot_title)
        
        
def plot_category_heatmap(DE_results_df, 
                          plot_title, 
                          ax=None,
                          cax=None,
                          DE_value_col='log2FoldChange', 
                          sort_by_DE=True, 
                          agg_key='DE_status', 
                          agg_categories=['up', 'down'],
                          color_categories=['r', 'b']
                         ):
    
    """
    Generate a heatmap to label categories of intervals (usually, DE genes).
    Parameters:
    -----------
    DE_results_df: pandas dataframe that has the list of intervals (usually, 
                   set of genes) and the levels of expression or categories 
                   labeled in a column.
    plot_title: title or this plot
    ax: the axis for plotting this heatmap.
    cax: the axis for plotting the colorbar.
    DE_value_col: column in DE_results_df with measure of differential 
                  expression, for sorting.
    sort_by_DE: True/False sort the intervals by their change in expression in
                descending order.
    agg_key: column in DE_results_df defining category labels
    agg_categories: the category label values in DE_results_df[agg_key]
    color_categories: the colors assigned to each category for the heatmap
    
    Returns:
    --------
    Heatmap plot with colored categories.
    """
    
    if ax == None:
        ax = plt.subplot()
        
        
    # build the heatmap matrices 
    ordered_values = []
        
    # rearrange the matrix for plotting everything together
    for cat in agg_categories:
        
        # get first category
        cat_ix = np.where(DE_results_df[agg_key] == cat)
        sub_results = DE_results_df.iloc[cat_ix]
        
         # sort according to DE_value in descending order
        if sort_by_DE:
            ordering = (-sub_results[DE_value_col]).argsort()
            ordered_values.append(sub_results.iloc[ordering][DE_value_col])
        else:
            ordered_values.append(sub_results[DE_val_col])
    
    # collecting the set of DE values and normalizing for plotting
    change_vals = np.transpose(np.expand_dims(np.concatenate(ordered_values), axis=0))
    minval = np.percentile(change_vals, 5)
    maxval = np.percentile(change_vals, 95)
    
    # plotting heatmap
    divnorm = colors.TwoSlopeNorm(vmin=minval, vcenter=0., vmax=maxval)
    hotcoldmap = plt.cm.get_cmap('RdBu').reversed()
    occ = ax.imshow(change_vals, cmap=hotcoldmap, norm=divnorm, aspect='auto')
    ax.xaxis.set_ticklabels([])
    ax.set_title(plot_title)
    
    # adding colorbar
    if cax == None:
        cbar = plt.colorbar(occ, cax=ax, extend='max', location='left', ticklocation='left')
    else:
        cbar = plt.colorbar(occ, cax=cax, extend='max', ticklocation='left')
    cbar.set_ticks(np.percentile(change_vals, [0, 5, 25, 50, 95, 100]))
    cbar.set_label(DE_value_col, rotation=90)
    
    
    ####################################
    
    # HELPER FUNCTIONS
    
    ####################################
    
def get_tss_gene_intervals(tss_df):
    '''
    Input: a .gtf file containing the chr, start, end
    corresponding to the TSS for the transcripts ready from a 
    genomic .gtf format annotation file.
    Output: a dataframe in bioframe format with a single TSS 
    per gene, with non-autosomes removed.
    '''
    
    # cleaning out less-well defined chromosome numbers
    tss_df = tss_df.loc[False==( tss_df['seqname'].str.contains('NT_'))]
    tss_df = tss_df.loc[False==( tss_df['seqname'].str.contains('MT'))]

    # paste 'chr' to all chromosome names
    tss_df["seqname"] = tss_df["seqname"]

    # rename column to chrom to match bedframe/bioframe format
    tss_df = tss_df.rename(columns= {"seqname" : "chrom"})

    # Removing pseudo chromosomes
    tss_df = tss_df.loc[False==( tss_df['chrom'].str.contains('chrGL'))]
    tss_df = tss_df.loc[False==( tss_df['chrom'].str.contains('chrJH'))]
    tss_df = tss_df.loc[False==( tss_df['chrom'].str.contains('chrY'))]
    tss_df = tss_df.loc[False==( tss_df['chrom'].str.contains('chrM'))]
    tss_df =tss_df.loc[True==tss_df['chrom'].str.contains('chr')]
    
    # drop duplicate TSSes
    return tss_df[['gene_id','chrom', 'start', 'end']].drop_duplicates(['gene_id'])


def label_DE_status(df, 
                    significance_col = 'padj', 
                    significance_threshold=0.05, 
                    fold_change_col='log2FoldChange', 
                    fold_change_threshold = 0,
                    DE_status_column = 'DE_status',
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
    df_out[DE_status_column] = 'nonsig'
    sig_inds = (df[significance_col] < significance_threshold)
    down_sig_inds = (df[fold_change_col] < fold_change_threshold) * sig_inds
    up_sig_inds = (df[fold_change_col] > fold_change_threshold) * sig_inds
    df_out.loc[up_sig_inds, DE_status_column] = 'up'
    df_out.loc[down_sig_inds, DE_status_column] = 'down'
    
    return df_out


def label_quantiles(df,
                    quantile_label_col='DE_status',
                    quantile_value_col= 'avg_vst_counts',
                    num_quantiles =4,
                    label_subset='nonsig'
                    ):
    """
    Appends a quantile label to values in a column matching "label_subset".
    Quantiles are derived from the full distribution of values in quantile_value_col
    Returns
    --------
    df_out : pd.DataFrame
        DataFrame where quantile labels have been appended to strings in quantile_label_col.
    Notes
    ------
    Quantile label col must be str or object.
    """
    df_out = df.copy()
    q_strs = pd.Series(np.linspace(0,100, num_quantiles).astype(int).astype(str))
    q_strs = q_strs[:-1].values+'-'+q_strs[1:].values
    quantile_labels = pd.qcut(df[quantile_value_col],
                              num_quantiles,
                              duplicates='drop',
                              labels=q_strs).astype(str)
    if label_subset:
        inds =df[quantile_label_col]==label_subset
        df_out.loc[inds, quantile_label_col] = df_out.loc[inds, quantile_label_col]+ "_" + quantile_labels.loc[inds]
    else:
        df_out.loc[quantile_label_col] =  df_out.loc[quantile_label_col] +'_' + quantile_labels
    return df_out
