# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import pandas_profiling as pp  # Ensure this library is installed
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib import transforms
import warnings

def drop_rep_columns(df):
    """
    Drops columns from the DataFrame where the column name contains 'REP' (case insensitive).

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with 'REP' columns removed.
    """
    # Identify columns containing 'REP' (case insensitive)
    columns_to_drop = [col for col in df.columns if 'REP' in col.upper()]

    # Drop those columns
    df = df.drop(columns=columns_to_drop, inplace=True)

    return df


# Generate a discrete palette from the colormap

warnings.filterwarnings("ignore")


TRn =  df["TR"].nunique()

cmap = cm.get_cmap("rocket", TRn)  # Sample 10 colors from the colormap
palette = [cmap(i) for i in range(cmap.N)]

# Number of subplots
num_charts = len([col for col in df.columns if col != "TR"])  # Exclude 'TR'
num_cols = 2  # Number of columns in the grid
num_rows = (num_charts + num_cols - 1) // num_cols  # Calculate rows dynamically

# Create the grid of subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()  # Flatten the axes array for easy indexing

# Plot each chart
plot_idx = 0
for i in df.columns:
    if i == "TR":  # Skip 'TR' and 'FI' columns
        continue

    ax = axes[plot_idx]  # Access the correct subplot

    # Point plot with updated parameters
    sns.pointplot(
        x="TR", y=i, data=df, linestyles='',
        markers='x', hue="TR", palette=palette, err_kws={'linewidth': 1.5}, capsize=0.2, ax=ax,
        legend=False  # Disable legend here
    )

    # Apply transform if collections exist
    if ax.collections:
        offset = transforms.ScaledTranslation(5 / 72., 0, ax.figure.dpi_scale_trans)
        trans = ax.collections[0].get_transform()
        ax.collections[0].set_transform(trans + offset)

    # Swarm plot for individual data points
    sns.swarmplot(
        x="TR", y=i, data=df, edgecolor="black", hue='TR', linewidth=0.9, ax=ax, palette=palette,
        legend=False  # Disable legend here
    )

    # Box plot for data distribution
    sns.boxplot(
        x="TR", y=i, data=df, saturation=1, ax=ax, hue='TR', palette=palette,
        legend=False  # Disable legend here
    )

    # Secondary point plot for additional information
    sns.pointplot(
        x="TR", y=i, data=df, hue='TR', linestyles='--', markers='o', color='k', err_kws={'linewidth': 0},
        capsize=0, ax=ax, legend=False  # Disable legend here
    )

    # Customize subplot appearance
    ax.set_ylabel(i)
    ax.set_title(f"{i} vs TR")
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    plot_idx += 1  # Move to the next subplot

# Remove any unused subplots
for idx in range(plot_idx, len(axes)):
    fig.delaxes(axes[idx])  # Remove extra axes

# Adjust layout
fig.tight_layout()
plt.show()



df.describe().round(3)