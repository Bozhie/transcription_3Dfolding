import pandas as pd
import numpy as np
import bbi
import bioframe as bf
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
    ax.set_title(plot_title)
    
    
def plot_binned_signal_heatmap(DE_results_df, 
                               stackup_matrix, 
                               plot_title, 
                               ax=None,
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
    cb = plt.colorbar(hm, ax=ax, extend='max', orientation='vertical')
    cb.set_ticks(np.percentile(stackup_matrix, [0, 95, 98, 99, 100]))
    cb.set_label('ChIP signal', rotation=270)

    ax.set(xticks=np.arange(0, nbins+1, 10),
    xticklabels=(np.arange(0, (window_size*2)+1, (window_size/2))-(window_size*2)//2),
    xlabel='Distance from boundary (bp)')
    ax.set_title(plot_title)
        
        
def plot_category_heatmap(DE_results_df, 
                           title, 
                           ax=None,
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
    title: title or this plot
    ax: the axis for plotting this heatmap.
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
    
    # adding colorbar
    cbar = plt.colorbar(occ, ax=ax, extend='max', location='left')
    cbar.set_ticks(np.percentile(change_vals, [0, 5, 25, 50, 95, 100]))
    cbar.set_label(DE_value_col, rotation=90)