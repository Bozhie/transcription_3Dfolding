import pandas as pd
import numpy as np
import bbi
import bioframe as bf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings


def plot_categorized_histogram(
    df, 
    column,
    agg_key='DE_status',
    category_colors={"up": 'tab:red', 
                     "down": 'tab:blue', 
                     "nonsig": 'tab:gray'},
    bins=50,
    val_range=(2,8),
    cumulative=False,
    density=False,
    plot_title=None,
    ax=None,
    add_legend=True
):
    """
    Plots distribution of values in column across different bins in 
    each category defined in df['agg_key'].
    
    Parameters:
    -----------
    df: pandas dataframe that has 'column' and the categories labeled in 
        'agg_key'.
    distance_col: the column in df with the distances for plotting
    agg_key: column in df containing category labels
    category_colors: the category label values in df[agg_key] mapped
        to the colors for plotting.
    bins: number of bins for histogram.
    val_range: range of log10 values in 'distance_col' for setting max and
        min.
    cumulative: True/False for whether the plot should build up cumulative
        distribution on x-axis.
    plot_title: title or this plot
    ax: the axis for plotting.

    Returns:
    --------
    plot
    """
    
    if ax == None:
        ax = plt.subplot()
    
    for cat, col in category_colors.items():
    
        cat_ix = np.where(df[agg_key] == cat)

        values = df.iloc[cat_ix][column].values
        ax.hist(values,
                bins=bins,
                range=val_range,
                density=density,
                cumulative=cumulative,
                histtype='step',
                lw=1.5,
                label=cat, 
                color=col)

    if add_legend:
        ax.legend(loc='upper right')
    if density:
        ylab = 'frequency'
    else:
        ylab = 'counts'
    ax.set(
        xlabel= column,
        ylabel= ylab
    )
    if plot_title != None:
        ax.set_title(plot_title)
        
def plot_distance_histogram(
    df, 
    distance_col,
    pseudocount=1,
    agg_key='DE_status',
    category_colors={"up": 'tab:red', 
                     "down": 'tab:blue', 
                     "nonsig": 'tab:gray'},
    bins=50,
    val_range=(2,8),
    cumulative=False,
    plot_title=None,
    ax=None,
    add_legend=True
):
    """
    Plots distribution of log10 distances across different bins in 
    each category defined in df['agg_key'].
    
    Parameters:
    -----------
    df: pandas dataframe that has distances in 'distance_col'
        and the categories labeled in 'agg_key'.
    distance_col: the column in df with the distances for plotting
    agg_key: column in df containing category labels
    category_colors: the category label values in df[agg_key] mapped
        to the colors for plotting.
    bins: number of bins for histogram.
    val_range: range of log10 values in 'distance_col' for setting max and
        min.
    cumulative: True/False for whether the plot should build up cumulative
        distribution on x-axis.
    plot_title: title or this plot
    ax: the axis for plotting.

    Returns:
    --------
    plot
    """
    
    df = df.copy().dropna(subset=[distance_col])
    
    if ax == None:
        ax = plt.subplot()
    
    for cat, col in category_colors.items():
    
        cat_ix = np.where(df[agg_key] == cat)

        dist = df.iloc[cat_ix][distance_col] + pseudocount
        ax.hist(np.log10(dist),
                bins=bins,
                range=val_range,
                density=True,
                cumulative=cumulative,
                histtype='step',
                lw=1.5,
                label=cat, 
                color=col)

    if add_legend:
        ax.legend(loc='upper left')
    if cumulative:
        ylab = 'cumulative count'
    else:
        ylab = 'frequency'
    ax.set(
        xlabel = 'log10 distance',
        ylabel = ylab
    )
    if plot_title != None:
        ax.set_title(plot_title)
        
        
def plot_count_histogram(
    df, 
    count_col,
    agg_key='DE_status',
    category_colors={"up": 'tab:red', 
                     "down": 'tab:blue', 
                     "nonsig": 'tab:gray'},
    bins=50,
    cumulative=False,
    plot_title=None,
    ax=None,
    add_legend=True
):
    """
    Plots distribution of counts across different bins in 
    each category defined in df['agg_key'].
    
    Parameters:
    -----------
    df: pandas dataframe that has counts in 'count_col'
        and the categories labeled in 'agg_key'.
    count_col: the column in df with the counts for plotting
    agg_key: column in df containing category labels
    category_colors: the category label values in df[agg_key] mapped
        to the colors for plotting.
    bins: number of bins for histogram.
    cumulative: True/False for whether the plot should build up cumulative
        distribution on x-axis.
    plot_title: title or this plot
    ax: the axis for plotting.

    Returns:
    --------
    plot
    """
    
    df = df.copy().dropna(subset=[count_col])
    
    if ax == None:
        ax = plt.subplot()
        
    val_range = (df[count_col].min(), df[count_col].max())
    
    for cat, col in category_colors.items():
    
        cat_ix = np.where(df[agg_key] == cat)

        dist = df.iloc[cat_ix][count_col]
        ax.hist(dist,
                bins=bins,
                density=True,
                range=val_range,
                cumulative=cumulative,
                histtype='step',
                lw=1.5,
                label=cat, 
                color=col)

    if add_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    if cumulative:
        ylab = 'cumulative count'
    else:
        ylab = 'frequency'
    ax.set(
        xlabel = ' '.join(count_col.split('_')),
        ylabel = ylab
    )
    if plot_title != None:
        ax.set_title(plot_title)
    
    
def distribution_features_by_region(
    region_df,
    feature_df,
    region_group_col='num_enhancers',
    feature_agg_key='DE_status',
    feature_name='genes',
    feature_category_colors={"up": 'tab:red', 
                             "down": 'tab:blue',
                             "nonsig": 'tab:grey'},
    bin_size=1,
    percentage=True,
    ax=None,
    plot_title=None
):
    """
    Plots bar graph of the percentage features that overlap with the intervals
    in region_df. The x axis is categories of regions defined by region_df[region_group_col]
    and y axis is number of features, grouped by category labels in feature_df[feature_agg_key].
    
    Parameters:
    -----------
    region_df: bioframe df with one interval (chr, start, end) for regions.
    feature_df: bioframe df with one interval (chr, start, end) for features (e.g. genes)
    region_group_col: column in region_df with categories of the regions.
    feature_agg_key: column in feature_df that has category labels for aggregation.
    feature_category_colors: categories in feature_df[feature_agg_key] and colors for plotting.
    
    Returns:
    --------
    bar plot
    """
    
    if ax == None:
        ax = plt.subplot()
    
    
    groups = pd.DataFrame()
    bar_size = 1/(len(feature_category_colors) + 1)
    x_pos = 0
    max_count_val = max(region_df[region_group_col])

    # collect the number of overlapping features per region for each category
    for cat, _ in feature_category_colors.items():

        cat_ix = np.where(feature_df[feature_agg_key] == cat)
        if feature_df.iloc[cat_ix].shape[0] < 1:
            warnings.warn(
                (
                "category {} is empty, skipped in plotting".
                    format(cat)
                )
            )
            continue

        region_df[cat+'_counts'] = bf.count_overlaps(region_df, feature_df.iloc[cat_ix])['count']

    # create x-axis values, where each bin is [a, b)
    bins = np.arange(0, max_count_val+1, bin_size)

    # aggregate across bins to generate bar plot
    for cat, color in feature_category_colors.items():

        counts = []
        for i in bins:
            counts.append(region_df.groupby(region_group_col).sum([cat+'_counts'])[cat+'_counts']
                          [i:i+bin_size].sum())

        groups[cat] = counts
        if percentage:
            vals = groups[cat]/groups[cat].sum()
            lab = 'Percentage of {} by {}'.format(feature_name,
                                                ' '.join(feature_agg_key.split('_')))
        else:
            vals = counts
            lab = 'Number of {}by {}'.format(feature_name,
                                              ' '.join(feature_agg_key.split('_')))
        ax.bar(groups.index + x_pos + .75*bar_size, vals, width=bar_size, color=color, label=cat)
        x_pos += bar_size

        
    ax.set(
        xticks=groups.index,
        xticklabels=bins,
        xlabel=' '.join(region_group_col.split('_')),
        ylabel=lab
    )
    ax.legend()
    
    if plot_title != None:
        ax.set_title(plot_title)

        
def distribution_regions_by_features(
    region_df,
    feature_df,
    region_group_col='num_enhancers',
    feature_agg_key='DE_status',
    feature_name='TADs',
    feature_category_colors={"up": 'tab:red', 
                             "down": 'tab:blue',
                             "nonsig": 'tab:grey'},
    bin_size=1,
    percentage=True,
    ax=None,
    num_genes_cutoff=1,
    plot_title=None
):
    """
    Plots bar graph of regions containing one or more of some
    aggregated category. e.g. TADs with number of features on 
    x-axis, count of TADs split by DE_status on y-axis
    
    Parameters:
    -----------
    region_df: bioframe df with one interval (chr, start, end) for regions.
    feature_df: bioframe df with one interval (chr, start, end) for features (e.g. genes)
    region_group_col: column in region_df with categories of the regions.
    feature_agg_key: column in feature_df that has category labels for aggregation.
    feature_category_colors: categories in feature_df[feature_agg_key] and colors for plotting.
    bin_size: for grouping region_group_col into bins
    percentage: True/False whether to take percentage of total TADs
    ax: axis for plotting
    num_genes_cutoff: selecting a cutoff value for counting in feature_agg_key
    plot_title: title
    
    Returns:
    --------
    bar plot
    """
    
    if ax == None:
        ax = plt.subplot()

    groups = pd.DataFrame()
    bar_size = 1/(len(feature_category_colors) + 1)
    x_pos = 0
    max_count_val = max(region_df[region_group_col])


    # collect the number of overlapping features per region for each category
    for cat, _ in feature_category_colors.items():

        cat_ix = np.where(feature_df[feature_agg_key] == cat)

        if feature_df.iloc[cat_ix].shape[0] < 1:
            warnings.warn(
                (
                "category {} is empty, skipped in plotting".
                    format(cat)
                )
            )
            continue

        region_df[cat+'_counts'] = bf.count_overlaps(region_df, feature_df.iloc[cat_ix])['count']

    # create x-axis values, where each bin is [a, b)
    bins = np.arange(0, max_count_val+1, bin_size)

    # aggregate across bins to generate bar plot
    for cat, color in feature_category_colors.items():

        counts = []
        for i in bins:

            counts.append(region_df.iloc[
                            np.where(region_df[cat+'_counts'] >= num_genes_cutoff)
                            ].groupby(region_group_col).count()[cat+'_counts'][i:i+bin_size].sum())

        groups[cat] = counts
        
        if percentage:
            vals = groups[cat]/groups[cat].sum()
            lab = 'Percentage of {} containing \n >= {} gene with {}'.format(feature_name,
                                                                   num_genes_cutoff,
                                            ' '.join(feature_agg_key.split('_')))
        else:
            vals = counts
            lab = 'Number of {} containing \n >= {} gene with {}'.format(feature_name,
                                                                   num_genes_cutoff,
                                            ' '.join(feature_agg_key.split('_')))
            
        ax.bar(groups.index + x_pos + .75*bar_size, vals, width=bar_size, color=color, label=cat)
        x_pos += bar_size

    ax.set(
    xticks=groups.index,
    xticklabels=bins,
    xlabel=' '.join(region_group_col.split('_')),
    ylabel='Percentage of {} containing \n >= {} gene with {}'.format(feature_name,
                                                                   num_genes_cutoff,
                                            ' '.join(feature_agg_key.split('_')))
    )
    ax.legend()

    if plot_title != None:
        ax.set_title(plot_title)
        
def assign_bin(df, value_col, bin_edges, bin_col_name):
    """
    Takes a dataframe and appends a column of bins to
    categorize the values in df['value_col'].
    
    Parameters:
    -----------
    df: dataframe
    value_col: values (numerical) to be sorted into bins
    bin_edges: an array of the [min, max) for sorting every bin
    bin_col_name: new column in df with bin
    
    Returns:
    --------
    df with df[bin_col_name] populated with the value of
        the bottom edge of the bin (e.g. if value_col=3, 
        bin_edges=[0, 5, 10], bin = 0)
    """
    
    df_out = df.copy()
    
    prev = None
    
    for i in bin_edges:
        if prev == None:
            prev = i
            continue

        bin_ix = (df['count_ctcf_peaks'] >= prev) & (df['count_ctcf_peaks'] < i)
        df_out.loc[bin_ix, 'count_ctcf_bin'] = prev

        prev = i
    
    return df_out