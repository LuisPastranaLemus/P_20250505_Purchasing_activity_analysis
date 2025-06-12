# Exploratory Data Analysis for Visualizations and summary statistics

from IPython.display import display, HTML
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Function to detect outlier boundaries with optional clamping of lower bound to zero
def outlier_limit_bounds(df, column, bound='both', clamp_zero=False):
    """
    Detects outlier thresholds based on the IQR method and returns rows beyond those limits.

    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): The name of the numerical column to analyze.
    bound (str): One of 'both', 'lower', or 'upper' to indicate which bounds to evaluate.
    clamp_zero (bool): If True, clamps the lower bound to zero (useful for non-negative metrics).

    Returns:
    DataFrame(s): Rows identified as outliers, depending on the bound selected.
    """

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = max(q1 - 1.5 * iqr, 0) if clamp_zero else q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    if bound == 'both':
        df_outliers_lb = df[df[column] < lower_bound]
        df_outliers_ub = df[df[column] > upper_bound]
        display(HTML(f"> Lower outlier threshold for column <i>'{column}'</i>: <b>{lower_bound}</b>"))
        display(HTML(f"> Upper outlier threshold for column <i>'{column}'</i>: <b>{upper_bound}</b>"))

        if df_outliers_lb.empty and df_outliers_ub.empty:
            display(HTML(f"> No outliers found in column <i>'{column}'</i>."))

        return df_outliers_lb, df_outliers_ub

    elif bound == 'upper':
        df_outliers_ub = df[df[column] > upper_bound]
        display(HTML(f"> Upper outlier threshold for column <i>'{column}'</i>: <b>{upper_bound}</b>"))

        if df_outliers_ub.empty:
            display(HTML(f"> No upper outliers found in column <i>'{column}'</i>."))

        return df_outliers_ub

    elif bound == 'lower':
        df_outliers_lb = df[df[column] < lower_bound]
        display(HTML(f"> Lower outlier threshold for column <i>'{column}'</i>: <b>{lower_bound}</b>"))

        if df_outliers_lb.empty:
            display(HTML(f"> No lower outliers found in column <i>'{column}'</i>."))

        return df_outliers_lb

    else:
        display(HTML(f"> Invalid 'bound' parameter. Use <b>'both'</b>, <b>'upper'</b>, or <b>'lower'</b>."))
        return None


# Function to evaluate the central tendency of a numerical feature
def evaluate_central_trend(df, column):
    """
    Evaluates the central tendency of a given column using the coefficient of variation (CV).
    
    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): Name of the numerical column to evaluate.
    
    Output:
    Displays the coefficient of variation and recommends the most reliable measure of central tendency
    based on the level of variability.
    """
    
    cv = (df[column].std() / df[column].mean()) * 100
    display(HTML(f"> Coefficient of variation for column <i>'{column}'</i>: <b>{cv:.2f} %</b>"))

    if 0 <= cv <= 10:
        display(HTML("> Very low variability: <i>highly reliable mean</i>. Recommended central measure: <b>mean</b>."))
    elif 10 < cv <= 20:
        display(HTML("> Moderate variability: <i>reasonably reliable mean</i>. Recommended central measure: <b>mean</b>."))
    elif 20 < cv <= 30:
        display(HTML("> Considerable variability: <i>potentially biased mean</i>. Recommended central measure: <b>mean</b> with caution."))
    else:
        display(HTML("> High variability: <i>mean may be misleading</i>. Recommended central measure: <b>median</b>."))
    
    print()

# Function to evaluate pairwise correlations among numerical columns
def evaluate_correlation(df):
    """
    Evaluates pairwise Pearson correlations between numerical columns in a DataFrame.
    
    Parameters:
    df (DataFrame): The input DataFrame with at least two numerical columns.
    
    Output:
    Displays correlation coefficients with interpretation levels:
    - Strong correlation (>|0.7|)
    - Moderate correlation (0.3 < |r| ≤ 0.7)
    - No linear relationship (r = 0)
    - Negative correlations (inverted relationship)
    """
    
    for column_x in df.columns:
        if df[column_x].dtype != 'object':
            for column_y in df.columns:
                if df[column_y].dtype != 'object' and column_x != column_y:
                    corr_value = df[column_x].corr(df[column_y])
                    
                    if 0.7 < corr_value <= 1.0:
                        display(HTML(f"> Correlation (<i>{column_x}</i>, <i>{column_y}</i>): <b>{corr_value:.2f}</b><br><b>Strong positive correlation</b>"))
                    elif 0.3 < corr_value <= 0.7:
                        display(HTML(f"> Correlation (<i>{column_x}</i>, <i>{column_y}</i>): <b>{corr_value:.2f}</b><br><b>Moderate positive correlation</b>"))
                    elif corr_value == 0:
                        display(HTML(f"> Correlation (<i>{column_x}</i>, <i>{column_y}</i>): <b>{corr_value:.2f}</b><br><b>No linear relationship</b>"))
                    elif -0.7 < corr_value <= -0.3:
                        display(HTML(f"> Correlation (<i>{column_x}</i>, <i>{column_y}</i>): <b>{corr_value:.2f}</b><br><b>Moderate negative correlation</b>"))
                    elif -1.0 <= corr_value <= -0.7:
                        display(HTML(f"> Correlation (<i>{column_x}</i>, <i>{column_y}</i>): <b>{corr_value:.2f}</b><br><b>Strong negative correlation</b>"))

# Function to visualize missing values within a DataFrame using a heatmap
def missing_values_heatmap(df):
    """
    Displays a heatmap of missing (NaN) values in the given DataFrame.
    
    Parameters:
    df (DataFrame): The input DataFrame to analyze.
    
    Output:
    A heatmap visualization showing the presence of missing values per column and row.
    """
    plt.figure(figsize=(15, 7))
    sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Heatmap of Missing Values')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.tight_layout()
    plt.show()

# Function to plot multiple boxplots side by side for comparison
# plot_boxplots(ds_list=[serie1, serie2, serie3], xlabels=['Group A', 'Group B', 'Group C'], ylabel='Values', 
#               title='Comparison of Value Distributions Across Groups', yticks_range=(0, 40, 5), rotation=45,
#               color=['skyblue', 'lightgreen', 'salmon']
def plot_boxplots(ds_list, xlabels, ylabel, title, yticks_range=None, rotation=0, color='grey'):
    """
    Plots multiple boxplots side by side, allowing for visual comparison across groups.

    Parameters:
    ds_list (list of Series): List of numerical pandas Series to plot.
    xlabels (list of str): Corresponding labels for each dataset.
    ylabel (str): Label for the y-axis.
    title (str): Title of the plot.
    yticks_range (tuple, optional): Range for y-axis ticks, e.g., (min, max, step).
    rotation (int, optional): Rotation angle for x and y tick labels.
    color (str or list, optional): Either a single color or a list of colors matching the groups.

    Raises:
    ValueError: If the number of datasets and labels do not match.

    Output:
    Displays a customized boxplot figure for group-wise value comparison.
    """

    if len(ds_list) != len(xlabels):
        raise ValueError("*** Error *** > The data list and labels must be the same length.")
    
    df = pd.DataFrame({
        'value': pd.concat(ds_list, ignore_index=True),
        'group': sum([[label] * len(s) for label, s in zip(xlabels, ds_list)], [])
    })

    plt.figure(figsize=(15, 7))

    # If color is a list, assign a custom palette; if string, use a solid color
    if isinstance(color, (list, tuple)) and len(color) == len(xlabels):
        palette = dict(zip(xlabels, color))
        sns.boxplot(x='group', y='value', hue='group', data=df, palette=palette)
    else:
        sns.boxplot(x='group', y='value', data=df, color=color)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rotation)

    if yticks_range is not None:
        plt.ylim(yticks_range[0], yticks_range[1])
        plt.yticks(np.arange(*yticks_range), rotation=rotation)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a histogram with mean and median reference lines
# plot_histogram(ds=series1, bins=np.arange(100, 170, 5), color='skyblue', title='Distribution of Durations', xlabel='Duration (minutes)',
#                ylabel='Frequency', xticks_range=(100, 170, 10), yticks_range=(0, 5, 1), rotation=45)
def plot_histogram(ds, bins=10, color='grey', title='', xlabel='', ylabel='Frequency',
                   xticks_range=None, yticks_range=None, rotation=0):
    """
    Plots a histogram for a given numerical Series with optional customization.

    Parameters:
    ds (Series): The numerical data to plot.
    bins (int or array-like): Number or range of histogram bins.
    color (str): Fill color for the bars.
    title (str): Plot title.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    xticks_range (tuple, optional): Range and step for x-ticks (min, max, step).
    yticks_range (tuple, optional): Range and step for y-ticks (min, max, step).
    rotation (int): Angle of tick label rotation.

    Output:
    Displays a histogram with vertical lines for mean and median.
    """

    ds = ds.dropna()
    
    mean_val = ds.mean()
    median_val = ds.median()

    plt.figure(figsize=(15, 7))
    sns.histplot(ds, bins=bins, edgecolor='black', color=color, kde=False)

    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='blue', linestyle='dashdot', linewidth=1.5, label=f'Median: {median_val:.2f}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xticks_range is not None:
        plt.xlim(xticks_range[0], xticks_range[1])
        plt.xticks(np.arange(*xticks_range), rotation=rotation)
    if yticks_range is not None:
        plt.ylim(yticks_range[0], yticks_range[1])
        plt.yticks(np.arange(*yticks_range), rotation=rotation)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a stacked histogram by group (hue)
# plot_hue_histogram(df=my_df, x_col='duration', hue_col='subscription_type', bins=20, title='Distribution of Durations by Subscription Type',
#                    xlabel='Duration (minutes)', ylabel='Frequency', legend_title='Subscription', legend_labels=['Free', 'Premium'])


def plot_hue_histogram(df, x_col='', hue_col='', bins=30, title='', xlabel='', ylabel='',
                       legend_title='', legend_labels=[]):
    """
    Plots a stacked histogram with grouping by a categorical variable (hue).

    Parameters:
    df (DataFrame): Input dataset.
    x_col (str): Numerical column to plot on the x-axis.
    hue_col (str): Categorical column used to group data.
    bins (int): Number of histogram bins.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    legend_title (str): Title for the legend.
    legend_labels (list, optional): Custom labels for legend categories.

    Output:
    Displays a stacked histogram with hue-based grouping.
    """
    
    plt.figure(figsize=(15, 7))
    sns.histplot(data=df, x=x_col, hue=hue_col, multiple='stack', bins=bins)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if legend_labels:
        plt.legend(title=legend_title, labels=legend_labels)
    else:
        plt.legend(title=legend_title)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to compare two distributions using overlapping histograms
# plot_dual_histogram(ds1=ages_no_show, ds2=ages_showed_up, bins=18, color1='tomato', color2='mediumseagreen', 
#                     title='Age Distribution: No-Show vs Show', xlabel='Age', ylabel='Number of Patients', label1='No Show',
#                     label2='Showed Up', xticks_range=(0, 100, 10), yticks_range=(0, 80, 10), rotation=45)
def plot_dual_histogram(ds1, ds2, bins=10, color1='black', color2='grey',
                        title='Histogram Comparison', xlabel='', ylabel='',
                        label1='', label2='', xticks_range=None, yticks_range=None, rotation=0):
    """
    Plots two overlapping histograms to visually compare distributions.

    Parameters:
    ds1 (Series): First numerical dataset.
    ds2 (Series): Second numerical dataset.
    bins (int): Number of bins for the histogram.
    color1 (str): Color for the first dataset.
    color2 (str): Color for the second dataset.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    label1 (str): Legend label for the first dataset.
    label2 (str): Legend label for the second dataset.
    xticks_range (tuple, optional): Range and step for x-ticks (min, max, step).
    yticks_range (tuple, optional): Range and step for y-ticks (min, max, step).
    rotation (int): Tick label rotation angle.

    Output:
    Displays overlapping histograms with mean and median lines for both datasets.
    """

    # Clean missing values
    ds1 = ds1.dropna()
    ds2 = ds2.dropna()

    # Compute statistics
    mean1_val = ds1.mean()
    median1_val = ds1.median()
    mean2_val = ds2.mean()
    median2_val = ds2.median()

    plt.figure(figsize=(15, 7))

    sns.histplot(ds1, bins=bins, edgecolor='black', kde=False, color=color1, label=label1, alpha=0.6)
    sns.histplot(ds2, bins=bins, edgecolor='black', kde=False, color=color2, label=label2, alpha=0.4)

    plt.axvline(mean1_val, color='red', linestyle='dashed', linewidth=1.5, label=f'{label1} Mean: {mean1_val:.2f}')
    plt.axvline(mean2_val, color='darkred', linestyle='dashed', linewidth=1.5, label=f'{label2} Mean: {mean2_val:.2f}')
    plt.axvline(median1_val, color='blue', linestyle='dashdot', linewidth=1.5, label=f'{label1} Median: {median1_val:.2f}')
    plt.axvline(median2_val, color='darkblue', linestyle='dashdot', linewidth=1.5, label=f'{label2} Median: {median2_val:.2f}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xticks_range is not None:
        plt.xlim(xticks_range[0], xticks_range[1])
        plt.xticks(np.arange(*xticks_range), rotation=rotation)
    if yticks_range is not None:
        plt.ylim(yticks_range[0], yticks_range[1])
        plt.yticks(np.arange(*yticks_range), rotation=rotation)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a frequency density histogram with optional KDE overlay
# plot_frequency_density(ds=series1, bins=np.arange(0, 1200, 50), color='grey', title='Frequency Density of Duration', xlabel='Duration (minutes)',
#                        ylabel='Density', xticks_range=(0, 1200, 100), show_kde=True, rotation=45)
def plot_frequency_density(ds, bins=10, color='grey', title='', xlabel='', ylabel='Density',
                           xticks_range=None, rotation=0, show_kde=True):
    """
    Plots a frequency density histogram with optional KDE curve.

    Parameters:
    ds (Series): Numerical data to plot.
    bins (int or array-like): Number or range of bins for the histogram.
    color (str): Histogram bar color.
    title (str): Plot title.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis (default: 'Density').
    xticks_range (tuple, optional): Tuple (min, max, step) for x-tick configuration.
    rotation (int, optional): Angle for tick label rotation.
    show_kde (bool, optional): Whether to overlay a KDE curve.

    Output:
    Displays a histogram normalized to show frequency density, with mean/median lines and optional KDE.
    """

    ds = ds.dropna()
    mean_val = ds.mean()
    median_val = ds.median()

    plt.figure(figsize=(15, 7))
    sns.histplot(ds, bins=bins, stat='density', edgecolor='black', color=color, alpha=0.7)

    if show_kde:
        sns.kdeplot(ds, color='darkblue', linewidth=2, label='KDE')

    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='blue', linestyle='dashdot', linewidth=1.5, label=f'Median: {median_val:.2f}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xticks_range:
        plt.xlim(xticks_range[0], xticks_range[1])
        plt.xticks(np.arange(*xticks_range), rotation=rotation)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a grouped barplot (categorical x-axis, grouped by hue)
# plot_grouped_barplot(ds=dataframe, x_col='month', y_col='median_duration', hue_col='plan', palette=['black', 'grey'],
#                      title='Average Call Duration by Plan and Month', xlabel='Month', ylabel='Average Call Duration (min)',
#                      xticks_range=range(0, 13, 1), yticks_range=range(0, 500, 50), rotation=65)
def plot_grouped_barplot(ds, x_col, y_col, hue_col=None, palette=['black', 'grey'],
                         title='', xlabel='', ylabel='', xticks_range=None,
                         yticks_range=None, rotation=0):
    """
    Plots a grouped bar chart with categorical grouping (hue).

    Parameters:
    ds (DataFrame): The dataset to use for plotting.
    x_col (str): The column to use for the x-axis (categorical).
    y_col (str): The column to plot as the bar height (numerical).
    hue_col (str, optional): The column to group by within each x-category.
    palette (list, optional): List of colors for each hue category.
    title (str): Plot title.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    xticks_range (range, optional): Tick range and step for the x-axis.
    yticks_range (range, optional): Tick range and step for the y-axis.
    rotation (int): Rotation angle for tick labels.

    Output:
    Displays a grouped bar plot with optional axis customization and legend.
    """

    plt.figure(figsize=(15, 7))
    sns.barplot(data=ds, x=x_col, y=y_col, hue=hue_col, palette=palette)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xticks_range is not None:
        plt.xticks(ticks=xticks_range, rotation=rotation)
    if yticks_range is not None:
        plt.yticks(ticks=yticks_range, rotation=rotation)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a horizontal bar chart from categorical data
# plot_horizontal_bar(ds=series_categorica, colors=['skyblue', 'salmon', 'lightgreen'], xlabel='Count', ylabel='Categories',
#                     title='Distribution of Categorical Values', xticks_range=(0, 100, 10), rotation=0)
def plot_horizontal_bar(ds, colors=['black', 'grey'], xlabel='', ylabel='', title='',
                        xticks_range=None, rotation=0):
    """
    Plots a horizontal bar chart for a categorical pandas Series.

    Parameters:
    ds (Series): Categorical data to summarize and visualize.
    colors (list): Color palette for each category.
    xlabel (str): Label for the x-axis (typically counts).
    ylabel (str): Label for the y-axis (categories).
    title (str): Title of the plot.
    xticks_range (tuple, optional): Tuple (min, max, step) for x-axis ticks.
    rotation (int): Rotation angle for x-axis tick labels.

    Output:
    Displays a horizontal bar chart with optional hue differentiation.
    """

    categories = ds.value_counts().index
    values = ds.value_counts().values

    plt.figure(figsize=(15, 7))
    sns.barplot(y=categories, x=values, hue=categories, dodge=False, palette=colors)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if xticks_range is not None:
        plt.xticks(np.arange(*xticks_range), rotation=rotation)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot grouped bar charts from a DataFrame with multiple columns
# plot_grouped_bars(df=music_activity_df.set_index('city'), title='Music Activity per City', xlabel='City', ylabel='Activity Count',
#                   x_rotation=45, y_rotation=0, grid_axis='y')
def plot_grouped_bars(df, title='', xlabel='', ylabel='', x_rotation=0, y_rotation=0, grid_axis='y'):
    """
    Plots grouped (clustered) bar charts for comparing multiple categories across an index.

    Parameters:
    df (DataFrame): A DataFrame where the index defines groups (e.g., cities) and columns are subcategories.
    title (str): Title of the chart.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    x_rotation (int): Rotation angle for x-axis tick labels.
    y_rotation (int): Rotation angle for y-axis tick labels.
    grid_axis (str): Axis along which to display grid lines ('x', 'y', or 'both').

    Output:
    Displays a grouped bar chart comparing values across index categories and columns.
    """

    df.plot(kind='bar', figsize=(15, 7))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=x_rotation)
    plt.yticks(rotation=y_rotation)
    plt.grid(axis=grid_axis)
    plt.tight_layout()
    plt.show()


# Function to plot a grouped bar chart with a specified index column
# plot_grouped_bars_indx(df=music_df, index_name='city', title='Music Activity per City', xlabel='City', ylabel='Activity Count',
#                        rotation=45, grid_axis='y')
def plot_grouped_bars_indx(df, index_name='', title='', xlabel='', ylabel='', rotation=0, grid_axis='y'):
    """
    Plots a grouped bar chart where rows are grouped by a specified index column
    (e.g., city) and bars represent multiple numeric columns.

    Parameters:
    df (DataFrame): Input DataFrame with categorical and numeric columns.
    index_name (str): Name of the column to use as the index (grouping variable).
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    rotation (int): Rotation angle for x-axis tick labels.
    grid_axis (str): Axis to show grid lines on ('x', 'y', or 'both').

    Returns:
    None: Displays the bar chart.
    """

    df_plot = df.set_index(index_name)
    df_plot.plot(kind='bar', figsize=(15, 7))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.grid(axis=grid_axis)
    plt.tight_layout()
    plt.show()

# Function to generate a customizable seaborn pairplot for exploratory correlation analysis
# plot_pairplot(df_music_activity_city_cov)
def plot_pairplot(df, height=3, aspect=2.5):
    """
    Plots a Seaborn pairplot for all numeric columns in a DataFrame.

    Parameters:
    df (DataFrame): The dataset to plot.
    height (float): Height (in inches) of each facet (subplot).
    aspect (float): Aspect ratio of each facet (width = height × aspect).

    Returns:
    None: Displays the pairplot.
    """
    sns.pairplot(df, height=height, aspect=aspect)
    plt.tight_layout()
    plt.show()

# Function to plot a scatter matrix for exploring pairwise relationships
def plot_scatter_matrix(df, figsize=(15, 7), diagonal='hist'):
    """
    Plots a scatter matrix for all numeric columns in a DataFrame using pandas' plotting tools.

    Parameters:
    df (DataFrame): The dataset to visualize.
    figsize (tuple): Size of the overall figure.
    diagonal (str): Type of plot on the diagonal ('hist' or 'kde').

    Returns:
    None: Displays the scatter matrix.
    """
    pd.plotting.scatter_matrix(df, figsize=figsize, diagonal=diagonal)
    plt.tight_layout()
    plt.show()
