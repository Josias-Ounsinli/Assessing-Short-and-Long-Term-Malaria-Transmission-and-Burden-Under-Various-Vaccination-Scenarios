""" Utils plotting functions """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDAPlots:
    """A class for EDA plotting functions"""

    def __init__(self) -> None:
        """Init function"""

    def heatmap_correlation_matrix(
        self, data: pd.DataFrame, figtitle: str, figname: str, country: str
    ):
        """Plotting Heatmap Correlation Matrix given a dataset"""
        # Create the correlation matrix
        corr = data.corr()

        # Generate a mask for the upper triangle; True = do NOT show
        mask = np.zeros_like(corr, dtype=np.bool_)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        figure, axis = plt.subplots(figsize=(30, 28))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            annot=True,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.5},
            ax=axis,
        )

        # Give title
        plt.title(f"{country}: {figtitle}", fontdict={'fontsize': 14})

        # Save fig
        figure.savefig(f"../plots/{country}: {figname}.png")

    def plot_time_series(self):
        """Plotting a time series"""
