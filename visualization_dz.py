import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm, transforms
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
    return df.drop(columns=columns_to_drop, inplace=False)  # Set inplace to False to return a new DataFrame


def plot_visualization(df):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import cm, transforms
    import warnings
    """
    Creates a grid of subplots visualizing the relationship of all columns (except 'TR')
    with respect to the 'TR' column using point plots, swarm plots, and box plots.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    """
    warnings.filterwarnings("ignore")  # Ignore warnings

    # Get the number of unique values in the 'TR' column
    TRn = df["TR"].nunique()

    # Generate a discrete palette from the colormap
    cmap = cm.get_cmap("rocket", TRn)
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
        if i == "TR":  # Skip 'TR' column
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

    # Print the summary statistics
    display(df.describe().round(3))




def plot_distributions(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import cm
    import warnings
    """
    Plots distribution plots (histograms with KDE) for each column in the DataFrame except the 'TR' column.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    """
    warnings.filterwarnings("ignore")  # Ignore warnings

    # Number of unique values in "TR"
    TRn = df["TR"].nunique()

    # Generate a discrete color palette
    cmap = cm.get_cmap("rocket", TRn)  # Sample colors from colormap
    palette = [cmap(i) for i in range(cmap.N)]

    # Number of subplots
    num_charts = len([col for col in df.columns if col != "TR"])  # Exclude 'TR'
    num_cols = 2  # Number of columns in the grid
    num_rows = (num_charts + num_cols - 1) // num_cols  # Calculate rows dynamically

def summary_statistics(df):
    """
    Prints a summary of descriptive statistics for the given DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    """
    print("Visão geral dos dados")
    print(df.describe().round(3))
    print("\nMédias por tratamento (TR)")
    print(df.groupby('TR').mean().round(3))
    print("\nDesvio padrão por tratamento (TR)")
    print(df.groupby('TR').std().round(3))
    print("\nCoeficiente de variação por tratamento (TR)")
    coef_var = (df.groupby('TR').std() / df.groupby('TR').mean()).round(3)
    print(coef_var)

