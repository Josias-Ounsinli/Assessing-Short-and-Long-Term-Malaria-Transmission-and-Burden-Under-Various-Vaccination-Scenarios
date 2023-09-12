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
        self,
        data: pd.DataFrame,
        figtitle: str,
        figname: str,
        country: str,
        corr_data=False,
        not_symmetric=False,
    ):
        """Plotting Heatmap Correlation Matrix given a dataset"""
        # Create the correlation matrix
        if corr_data:
            corr = data
        else:
            corr = data.corr()

        if not_symmetric:
            mask = None
        else:
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
        plt.title(f"{country}: {figtitle}", fontdict={"fontsize": 14})

        # Save fig
        figure.savefig(f"../plots/{country}: {figname}.png")

    def subset_correlation_matrix(
        self,
        data,
        get_features=False,
        thresholdup=None,
        get_target_feat=False,
        thresholdlow=None,
    ):
        """Subset a correlation and plotting heatmap"""

        corr = data.corr().unstack()

        corr = corr.reset_index().rename(columns={0: "correlation"})

        corr = pd.pivot_table(
            corr, values="correlation", index="level_0", columns="level_1"
        )
        corr = corr.loc[list(data.columns), list(data.columns)]

        features = [
            "Feature_1",
            "Feature_2",
            "Feature_3",
            "Feature_4",
            "Feature_5",
            "Feature_6",
            "Feature_7",
            "Feature_8",
            "Feature_9",
            "Feature_10",
            "Feature_11",
            "Feature_12",
            "Feature_13",
            "Feature_14",
            "Feature_15",
            "Feature_16",
            "Feature_17",
            "Feature_18",
            "Feature_19",
            "Feature_20",
            "Feature_21",
            "Feature_22",
            "Feature_23",
            "Feature_24",
            "Feature_25",
            "Feature_26",
            "Feature_27",
            "Feature_28",
            "Feature_29",
            "Feature_30",
            "Feature_31",
            "Feature_32",
            "Feature_33",
            "Feature_34",
            "Feature_35",
        ]

        targets = [
            "Target_1",
            "Target_2",
            "Target_3",
            "Target_4",
            "Target_5",
            "Target_6",
            "Target_7",
            "Target_8",
            "Target_9",
        ]

        if get_features:
            corr = corr.loc[features, features]

            if thresholdup:
                mask_greater = np.abs(corr.values) > thresholdup
                corr[mask_greater] = None

        if get_target_feat:
            corr = corr.loc[targets, features]

            if thresholdlow:
                mask_less = np.abs(corr.values) < thresholdlow
                corr[mask_less] = None

        return corr

    def plot_time_series(self):
        """Plotting a time series"""
