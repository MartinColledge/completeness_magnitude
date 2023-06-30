#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:53:16 2023

Completeness magnitude estimations with different techniques:
    1. Maximum curvature method
    2. Goodness of fit

Reference for techniques:

http://dx.doi.org/10.5078/corssa-00180805

R code taken from the above reference and converted to Python,
except for the maximum curvature method.

@author: Martin Colledge
"""

# Import usual modules

import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from manage_paths import save_fig_custom

# Import custom style for figures
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use(
    "/home/martin/.config/matplotlib/mpl_configdir/stylelib/"
    "BoldAndBeautiful.mplstyle"
)


class Catalog:
    """Investigate catalog instance's completeness magnitude."""

    def __init__(self, path_to_figures, catalog, delta_magnitude):
        """
        Initialize catalog instance.

        Parameters
        ----------
        path_to_figures : string
            Path to figures.
        catalog : pandas DataFrame
            Seimicity catalog.
        delta_magnitude : float
            Difference of magnitude.

        Returns
        -------
        None.

        """
        self.path_to_figures = path_to_figures
        self.catalog = catalog
        self.delta_magnitude = delta_magnitude

        self.catalog["Magnitude"] = (
            np.round(self.catalog["Magnitude"] / delta_magnitude)
            * delta_magnitude
        )
        self.magnitudes_in_catalog = np.arange(
            self.catalog.groupby("Magnitude").count().index[0],
            self.catalog.groupby("Magnitude").count().index[-1]
            + delta_magnitude,
            delta_magnitude,
        )
        self.cumulative_magnitude_frequency = []
        for magnitude in self.magnitudes_in_catalog:
            self.cumulative_magnitude_frequency.append(
                len(
                    self.catalog[
                        self.catalog["Magnitude"]
                        > magnitude - self.delta_magnitude / 2
                    ]
                )
            )

    def plot_magnitude_distribution(self):
        """
        Plot the magnitude distribution.

        Plot the cumulative and non cumulative magnitude frequency
        distribution.

        Returns
        -------
        None.

        """
        sns.histplot(
            data=self.catalog, x="Magnitude", fill=False, binwidth=0.1
        )
        plt.yscale("log")
        save_fig_custom(
            self.path_to_figures,
            plt.gcf(),
            figname="MagnitudeDistribution",
        )
        plt.close()

        sns.ecdfplot(
            data=self.catalog, x="Magnitude", complementary=True, stat="count"
        )
        plt.yscale("log")
        save_fig_custom(
            self.path_to_figures,
            plt.gcf(),
            figname="CumulativeMagnitudeDistribution",
        )
        plt.close()

    def maximum_curvature(self, plot=True, verbose=True):
        """
        Determine maximum curvature of magnitude frequency distribution.

        PROS :
        - Non-parametric
        - Straightforward
        - Statistically robust
        CONS :
        - Underestimates Mc in bulk data

        Parameters
        ----------
        plot : bool, optional
            Whether to plot the magnitude distribution derivative with maximum.
            Default is True.
        verbose : bool, optional
            Whether to print details.
            Default is True.

        Returns
        -------
        inflection_magnitude : float
            Completeness magnitude estimated.

        """
        # Retrieve the magnitude that is most represented in the catalog
        most_common_magnitude = (
            self.catalog.groupby("Magnitude").count()["DateTime"].idxmax()
        )

        # Calculate the first derivative of the frequency-magnitude cumulative
        # curve
        distribution_derivative = (
            self.catalog.groupby("Magnitude")
            .count()["DateTime"]
            .cumsum()
            .diff()
        )
        # The completeness magnitude is taken as the maximum of the derivative
        inflection_magnitude = distribution_derivative.idxmax()
        # Plot the magnitude distribution derivative with the maximum
        if plot:
            sns.lineplot(distribution_derivative)
            plt.axvline(
                inflection_magnitude,
            )
            plt.ylabel("Magnitude distribution derivative")
            save_fig_custom(
                self.path_to_figures,
                plt.gcf(),
                figname="MaximumCuvatureInflectionPoint",
            )
            plt.close()

        # If the most common magnitude and the inflection magnitude disagree,
        # warn the user. If they agree, return one.
        if verbose:
            print("Maximum Curvature completeness magnitude estimation.")
            if most_common_magnitude != inflection_magnitude:
                print(f"{inflection_magnitude = }, {most_common_magnitude = }")
                print("Returning inflexion magnitude.")
            else:
                print(f"Completeness magnitude = {inflection_magnitude}")
        return inflection_magnitude

    def goodness_of_fit_test(self, plot=True):
        """
        Determine goodness of fit of magnitude frequency distribution.

        PROS :
        - G-R deviation definition 90% conf. not always reached
        CONS :
        - May underestimate Mc

        Parameters
        ----------
        plot : bool, optional
            Whether to plot the magnitude distribution derivative with maximum.
            Default is True.

        Returns
        -------
        completeness_magnitude : float
            Completeness magnitude of the catalog as estimated.

        """
        # Get the maximum curvature estimate as a starting point
        max_curvature_estimation = self.maximum_curvature(
            plot=False, verbose=False
        )
        # Define the magnitudes to be tested as around the maximum curvature
        # magnitude
        tested_magnitudes = (
            max_curvature_estimation - 0.4 + (np.arange(15) - 1) / 10
        )
        # Initiialize the residuals vector
        residuals = np.zeros(15)

        # For every magnitude to test, do a goodness of fit evaluation
        for i in range(15):
            # Select the catalog above the tested magnitude
            catalog_above_tested_magnitude = self.catalog[
                self.catalog["Magnitude"]
                > tested_magnitudes[i] - self.delta_magnitude / 2
            ]
            # Determien the b value with the Aki-Utsu method mle method.
            b_value_gr = np.log10(np.exp(1)) / (
                np.mean(catalog_above_tested_magnitude["Magnitude"])
                - (tested_magnitudes[i] - self.delta_magnitude / 2)
            )
            # Determine the a value.
            a_value_gr = (
                np.log10(len(catalog_above_tested_magnitude))
                + b_value_gr * tested_magnitudes[i]
            )
            # Get the modeled distribution with these GR parameters
            frequency_magnitude_cumulative_model = 10 ** (
                a_value_gr - b_value_gr * self.magnitudes_in_catalog
            )
            # Get the index of the magnitudes above the cutoff
            index_of_magnitude_above_tested = np.where(
                self.magnitudes_in_catalog >= tested_magnitudes[i]
            )[0]

            # Retrieve the magnitude distribution above the cutoff
            cumulative_frequency_above_magnitude = [
                self.cumulative_magnitude_frequency[mag]
                for mag in index_of_magnitude_above_tested
            ]
            cumulative_frequency_above_magnitude_model = [
                frequency_magnitude_cumulative_model[mag]
                for mag in index_of_magnitude_above_tested
            ]

            # For every magnitude above the tested cutoff magnitude, determine
            # the fit of the model.
            residuals[i] = (
                np.sum(
                    np.abs(
                        [
                            obs - mod
                            for obs, mod in zip(
                                cumulative_frequency_above_magnitude,
                                cumulative_frequency_above_magnitude_model,
                            )
                        ]
                    )
                )
                / np.sum(cumulative_frequency_above_magnitude)
                * 100
            )
        # Identify the first occuerence of a low residual.
        index_of_first_low_residual = np.where(residuals <= 5)[
            0
        ]  # 95% confidence
        print("Goodness of fit completeness magnitude estimation.")
        # Assign the completeness as the magnitude the gives the first low residual.
        if len(index_of_first_low_residual) != 0:
            completeness_magnitude = tested_magnitudes[
                index_of_first_low_residual[0]
            ]
            print(
                f"Completness magnitude = {completeness_magnitude} "
                f"with 95% confidence"
            )
        # Loosen the definition of "low residual" if 95% criterion not reached.
        else:
            index_of_first_low_residual = np.where(residuals <= 10)[
                0
            ]  # 90% confidence
            if len(index_of_first_low_residual) != 0:
                completeness_magnitude = tested_magnitudes[
                    index_of_first_low_residual[0]
                ]
                print(
                    f"Completness magnitude = {completeness_magnitude} "
                    f"with 90% confidence"
                )
            # Fall back to the maximum curvature if not good enough.
            else:
                completeness_magnitude = max_curvature_estimation
                print("Fell back to maximum curvature")
                print(f"Completness magnitude = {completeness_magnitude}")
        
        # Plot the residuals distribution.
        if plot:
            sns.scatterplot(x=tested_magnitudes, y=residuals)
            plt.ylabel("Residual in %")
            plt.xlabel("Cutoff Magnitude")
            plt.ylim(bottom=0)
            plt.axhline(5, ls=":", c="k")
            plt.axvline(completeness_magnitude, ls="--")
            save_fig_custom(
                self.path_to_figures, plt.gcf(), "GoodnessOfFitResidual"
            )
        return completeness_magnitude
