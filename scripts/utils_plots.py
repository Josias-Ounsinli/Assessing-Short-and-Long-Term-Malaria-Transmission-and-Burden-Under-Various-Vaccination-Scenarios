""" Utils plotting functions """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import gridspec


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
        data: pd.DataFrame,
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
        ]

        targets = [
            "Target_1",
            "Target_2",
            "Target_3",
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

    def _subplots_centered(self, nrows: int, ncols: int, figsize: tuple, nfigs: int):
        """
        Modification of matplotlib plt.subplots(),
        useful when some subplots are empty.

        It returns a grid where the plots
        in the **last** row are centered.
        """
        # Check empty
        assert nfigs < nrows * ncols

        fig = plt.figure(figsize=figsize)
        axis = []

        modulo = nfigs % ncols
        modulo = range(1, ncols + 1)[-modulo]  # subdivision of columns
        gridspecobject = gridspec.GridSpec(nrows, modulo * ncols)

        for i in range(0, nfigs):
            row = i // ncols
            col = i % ncols

            if row == nrows - 1:  # center only last row
                off = int(modulo * (ncols - nfigs % ncols) / 2)
            else:
                off = 0

            axis_element = plt.subplot(
                gridspecobject[row, modulo * col + off : modulo * (col + 1) + off]
            )
            axis.append(axis_element)

        return fig, axis

    def describe_per_country(
        self, data: pd.DataFrame, country_column: str, variables: list, **kwargs
    ):
        """Describe a list of variable per country using pointplot"""

        if kwargs["centered"]:
            fig, axis = self._subplots_centered(
                nrows=kwargs["nrows"],
                ncols=kwargs["ncols"],
                figsize=kwargs["figsize"],
                nfigs=kwargs["nfigs"],
            )

            for column in variables:
                sns.pointplot(
                    data=data,
                    y=column,
                    x=country_column,
                    estimator="min",
                    color="green",
                    linestyles="--",
                    ci=None,
                    label="min",
                    ax=axis[variables.index(column)],
                )
                sns.pointplot(
                    data=data,
                    y=column,
                    x=country_column,
                    estimator="mean",
                    ci="sd",
                    label="mean",
                    ax=axis[variables.index(column)],
                )
                sns.pointplot(
                    data=data,
                    y=column,
                    x=country_column,
                    estimator="max",
                    color="red",
                    linestyles="--",
                    ci=None,
                    label="max",
                    ax=axis[variables.index(column)],
                )
                axis[variables.index(column)].legend()
                axis[variables.index(column)].set_title(
                    kwargs["subtitles"][variables.index(column)]
                )

        else:
            fig, axis = plt.subplots(
                nrows=kwargs["nrows"],
                ncols=kwargs["ncols"],
                figsize=kwargs["figsize"],
                sharex=True,
                constrained_layout=True,
            )

            for i in range(kwargs["nrows"]):
                for j in range(kwargs["ncols"]):
                    sns.pointplot(
                        data=data,
                        y=variables[i * kwargs["ncols"] + j],
                        x=country_column,
                        estimator="min",
                        color="green",
                        linestyles="--",
                        ci=None,
                        label="min",
                        ax=axis[i, j],
                    )
                    sns.pointplot(
                        data=data,
                        y=variables[i * kwargs["ncols"] + j],
                        x=country_column,
                        estimator="mean",
                        ci="sd",
                        label="mean",
                        ax=axis[i, j],
                    )
                    sns.pointplot(
                        data=data,
                        y=variables[i * kwargs["ncols"] + j],
                        x=country_column,
                        estimator="max",
                        color="red",
                        linestyles="--",
                        ci=None,
                        label="max",
                        ax=axis[i, j],
                    )
                    axis[i, j].legend()
                    axis[i, j].set_title(kwargs["subtitles"][i * kwargs["ncols"] + j])
                    axis[0, 1].set_xlabel("")

        fig.suptitle(
            "\nSummary of the target variables per country over the study period (2000-2022)\n",
            y=0.98,
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig("../plots/description_targets_per_country.png")

    def plot_serie_per_country(
        self, data: pd.DataFrame, country_column: str, targets_list: list, **kwargs
    ):
        """Describe a list of variable per country using pointplot"""

        if kwargs["centered"]:
            fig, axis = self._subplots_centered(
                nrows=kwargs["nrows"],
                ncols=kwargs["ncols"],
                figsize=kwargs["figsize"],
                nfigs=kwargs["nfigs"],
            )

            if kwargs["vaccination_focus"]:
                # for country in countries_iso3:
                #     (
                #         data.loc[country, :]
                #         .groupby(kwargs["vaccination_variable"])[variable]
                #         .plot(
                #             ax=axis[countries_iso3.index(country)],
                #             ylabel=variable,
                #             style=".-",
                #             title=f"{kwargs['countries_names'][countries_iso3.index(country)]}",
                #         )
                #     )

                focus_on_rts = data[data["Feature_27"] == 1]
                focus_on_r21 = data[data["Feature_28"] == 1]

                for target in targets_list:
                    sns.lineplot(
                        data=data,
                        x=kwargs["x_columns"],
                        y=target,
                        style=country_column,
                        markers=True,
                        dashes=False,
                        markersize=10,
                        ax=axis[targets_list.index(target)],
                    )
                    sns.lineplot(
                        data=focus_on_rts,
                        x=kwargs["x_columns"],
                        y=target,
                        style=country_column,
                        markers=True,
                        dashes=False,
                        markersize=10,
                        linestyle="",
                        legend=None,
                        ax=axis[targets_list.index(target)],
                    )
                    sns.lineplot(
                        data=focus_on_r21,
                        x=kwargs["x_columns"],
                        y=target,
                        style=country_column,
                        markers=True,
                        dashes=False,
                        markersize=10,
                        linestyle="",
                        legend=None,
                        alpha=0.5,
                        ax=axis[targets_list.index(target)],
                    )

                    axis[targets_list.index(target)].set_title(
                        kwargs["subtitles"][targets_list.index(target)]
                    )

            else:
                # for country in countries_iso3:
                #     (
                #         data.loc[country, variable].plot(
                #             ax=axis[countries_iso3.index(country)],
                #             ylabel=variable,
                #             style=".-",
                #             title=f"{kwargs['countries_names'][countries_iso3.index(country)]}",
                #         )
                #     )

                for target in targets_list:
                    sns.lineplot(
                        data=data,
                        x=kwargs["x_columns"],
                        y=target,
                        style=country_column,
                        markers=True,
                        dashes=False,
                        markersize=10,
                        ax=axis[targets_list.index(target)],
                    )

                    axis[targets_list.index(target)].set_title(
                        kwargs["subtitles"][targets_list.index(target)]
                    )

        else:
            fig, axis = plt.subplots(
                nrows=kwargs["nrows"],
                ncols=kwargs["ncols"],
                figsize=kwargs["figsize"],
                sharex=True,
                constrained_layout=True,
            )

            if kwargs["vaccination_focus"]:
                # for i in range(kwargs["nrows"]):
                #     for j in range(kwargs["ncols"]):
                #         (
                #             data.loc[countries_iso3[i * kwargs["ncols"] + j], :]
                #             .groupby(kwargs["vaccination_variable"])[variable]
                #             .plot(
                #                 ax=axis[i, j],
                #                 ylabel=variable,
                #                 style=".-",
                #                 title=f"{kwargs['countries_names'][i * kwargs['ncols'] + j]}",
                #             )
                #         )

                focus_on_rts = data[data["Feature_27"] == 1]
                focus_on_r21 = data[data["Feature_28"] == 1]

                for i in range(kwargs["nrows"]):
                    for j in range(kwargs["ncols"]):
                        sns.lineplot(
                            data=data,
                            x=kwargs["x_columns"],
                            y=targets_list[i * kwargs["ncols"] + j],
                            style=country_column,
                            markers=True,
                            dashes=False,
                            markersize=10,
                            ax=axis[i, j],
                        )
                        sns.lineplot(
                            data=focus_on_rts,
                            x=kwargs["x_columns"],
                            y=targets_list[i * kwargs["ncols"] + j],
                            style=country_column,
                            markers=True,
                            dashes=False,
                            markersize=10,
                            linestyle="",
                            legend=None,
                            ax=axis[i, j],
                        )
                        sns.lineplot(
                            data=focus_on_r21,
                            x=kwargs["x_columns"],
                            y=targets_list[i * kwargs["ncols"] + j],
                            style=country_column,
                            markers=True,
                            dashes=False,
                            markersize=10,
                            linestyle="",
                            legend=None,
                            alpha=0.5,
                            ax=axis[i, j],
                        )

                        axis[i, j].set_title(
                            kwargs["subtitles"][i * kwargs["ncols"] + j]
                        )
                        axis[0, 1].set_xlabel("")

            else:
                # for i in range(kwargs["nrows"]):
                #     for j in range(kwargs["ncols"]):
                #         (
                #             data.loc[
                #                 countries_iso3[i * kwargs["ncols"] + j], variable
                #             ].plot(
                #                 ax=axis[i, j],
                #                 ylabel=variable,
                #                 style=".-",
                #                 title=f"{kwargs['countries_names'][i * kwargs['ncols'] + j]}",
                #             )
                #         )

                for i in range(kwargs["nrows"]):
                    for j in range(kwargs["ncols"]):
                        sns.lineplot(
                            data=data,
                            x=kwargs["x_columns"],
                            y=targets_list[i * kwargs["ncols"] + j],
                            style=country_column,
                            markers=True,
                            dashes=False,
                            markersize=10,
                            ax=axis[i, j],
                        )

                        axis[i, j].set_title(
                            kwargs["subtitles"][i * kwargs["ncols"] + j]
                        )
                        axis[0, 1].set_xlabel("")

        if kwargs["vaccination_focus"]:
            fig.suptitle(
                "\nTarget variables series with focus on RTS trials (orange color) and R21 trial (transparent green color) per country over the study period (2000-2022)\n",
                y=0.98,
                fontsize=14,
            )
            fig.tight_layout()
            fig.savefig("../plots/Target_series_per_country_focus_on_vaccination.png")

        else:
            fig.suptitle(
                "\nTarget variables series per country over the study period (2000-2022)\n",
                y=0.98,
                fontsize=14,
            )
            fig.tight_layout()
            fig.savefig(f"../plots/Target_series_per_country.png")

    def serve_country_predictions(self, country, data, targets_list, **kwargs):
        """Serve country predictions"""

        if kwargs["centered"]:
            fig, axis = self._subplots_centered(
                nrows=kwargs["nrows"],
                ncols=kwargs["ncols"],
                figsize=kwargs["figsize"],
                nfigs=kwargs["nfigs"],
            )

            for variable in targets_list:
                (
                    data.set_index("Date")
                    .groupby("Predicted")[variable]
                    .plot(
                        ax=axis[targets_list.index(variable)],
                        ylabel=variable,
                        style=".-",
                        title=f"{kwargs['target_names'][targets_list.index(variable)]}",
                    )
                )

        else:
            fig, axis = plt.subplots(
                nrows=kwargs["nrows"],
                ncols=kwargs["ncols"],
                figsize=kwargs["figsize"],
                sharex=True,
                constrained_layout=True,
            )

            for i in range(kwargs["nrows"]):
                for j in range(kwargs["ncols"]):
                    (
                        data.set_index("Date")
                        .groupby("Predicted")[targets_list[i * kwargs["ncols"] + j]]
                        .plot(
                            ax=axis[i, j],
                            ylabel=targets_list[i * kwargs["ncols"] + j],
                            style=".-",
                            title=f"{kwargs['target_names'][i * kwargs['ncols'] + j]}",
                        )
                    )

        fig.suptitle(
            f"\nMalaria Trends in {country} from 2000 to 2070\n",
            y=0.98,
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(f"../plots/{country}_normal_predictions.png")

    # def sirvd_dynamics(self, t_pred, compartments: dict, for_scenario):
    #     """Plot model compartments dynamics

    #     S, I, R, V and D over time
    #     """

    #     plt.figure(figsize=(20, 10))
    #     for c in compartments.keys():
    #         plt.plot(t_pred, compartments[c], label=f"{c}")

    #     plt.xlabel("Time")
    #     plt.ylabel("Population")
    #     plt.title(
    #         f"Scenario {for_scenario}: Simulated evolution of compartments size over time"
    #     )
    #     plt.legend()
    #     plt.savefig(f"../plots/Scenario_{for_scenario}_compartment_dynamics")
    #     plt.show()

    def evolution_plot_per_country(
        self, data: pd.DataFrame, country_column: str, variables: list, **kwargs
    ):
        """Describe a list of variable per country using pointplot"""

        if kwargs["centered"]:
            fig, axis = self._subplots_centered(
                nrows=kwargs["nrows"],
                ncols=kwargs["ncols"],
                figsize=kwargs["figsize"],
                nfigs=kwargs["nfigs"],
            )

            for column in variables:
                sns.lineplot(
                    data=data,
                    y=column,
                    x=kwargs["x_column"],
                    hue=country_column,
                    style=country_column,
                    markers=True,
                    dashes=False,
                    linewidth=1.5,
                    markersize=8,
                    ax=axis[variables.index(column)],
                )

                axis[variables.index(column)].set_title(
                    kwargs["subtitles"][variables.index(column)]
                )

        else:
            fig, axis = plt.subplots(
                nrows=kwargs["nrows"],
                ncols=kwargs["ncols"],
                figsize=kwargs["figsize"],
                sharex=True,
                constrained_layout=True,
            )

            for i in range(kwargs["nrows"]):
                for j in range(kwargs["ncols"]):
                    sns.lineplot(
                        data=data,
                        y=variables[i * kwargs["ncols"] + j],
                        x=kwargs["x_column"],
                        hue=country_column,
                        style=country_column,
                        markers=True,
                        dashes=False,
                        linewidth=2,
                        markersize=10,
                        ax=axis[i, j],
                    )

                    axis[i, j].set_title(kwargs["subtitles"][i * kwargs["ncols"] + j])
                    axis[0, 1].set_xlabel("")

        fig.suptitle(
            "\nEvolution of the target variables per study country over the study period (2000-2022)\n",
            y=0.98,
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig("../plots/evolution_targets_per_study_country.png")
