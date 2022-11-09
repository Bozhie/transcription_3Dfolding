import pandas as pd
import numpy as np
import bbi
import bioframe as bf
import matplotlib.pyplot as plt
import matplotlib.colors as colors


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
    ax.set(
        xlabel='log10 distance',
        ylabel='frequency'
    )
    if plot_title != None:
        ax.set_title(plot_title)

        
def group_features_by_region(
    region_df,
    feature_df,
    region_group_col='num_enhancers',
    feature_agg_key='DE_status',
    feature_name='genes',
    feature_category_colors={"up": 'tab:red', 
                             "down": 'tab:blue',
                             "nonsig": 'tab:grey'},
    ax=None,
    plot_title=None
):
    """
    Plots bar graph of the sum of features that overlap with the intervals
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
    for cat, col in feature_category_colors.items():

        cat_ix = np.where(feature_df[feature_agg_key] == cat)
        
        if feature_df.iloc[cat_ix].shape[0] < 1:
            print('category {} is empty, skipped'.format(cat))
            continue
            
        region_df[cat+'_counts'] = bf.count_overlaps(region_df, feature_df.iloc[cat_ix])['count']
        groups[cat] = region_df.groupby(region_group_col).sum([cat+'_counts'])[cat+'_counts']

        ax.bar(groups.index + x_pos, groups[cat], width=bar_size, color=col, label=cat)
        x_pos += bar_size
        
    ax.set(
        xlabel=' '.join(region_group_col.split('_')),
        ylabel='Number of {}by {}'.format(feature_name,
                                              ' '.join(feature_agg_key.split('_')))
    )
    ax.legend()
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
    for cat, col in feature_category_colors.items():

        cat_ix = np.where(feature_df[feature_agg_key] == cat)
        
        if feature_df.iloc[cat_ix].shape[0] < 1:
            print('category {} is empty, skipped'.format(cat))
            continue

        region_df[cat+'_counts'] = bf.count_overlaps(region_df, feature_df.iloc[cat_ix])['count']
        groups[cat] = region_df.groupby(region_group_col).sum([cat+'_counts'])[cat+'_counts']
        perc = groups[cat]/groups[cat].sum()

        ax.bar(groups.index + x_pos, perc, width=bar_size, color=col, label=cat)
        x_pos += bar_size
        
    ax.set(
        xlabel=' '.join(region_group_col.split('_')),
        ylabel='Percentage of {} by {}'.format(feature_name,
                                              ' '.join(feature_agg_key.split('_')))
    )
    ax.legend()
    
    if plot_title != None:
        ax.set_title(plot_title)
