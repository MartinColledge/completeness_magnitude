#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:53:16 2023

@author: Martin Colledge
"""

# Import usual modules
import os
import sys
import datetime
import importlib

import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import completeness_magnitude
from manage_paths import manage_paths, get_file_names

# Import custom style for figures
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use(
    "/home/martin/.config/matplotlib/mpl_configdir/stylelib/"
    "BoldAndBeautiful.mplstyle"
)

# Get Script Name
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# Get path of data input
PATHS_TO_DATA = ["../0_data"]

# Extension type of data file
FILE_EXTENSIONS = [".parquet"]

# Manage Paths
path_to_output, path_to_figures = manage_paths(
    SCRIPT_NAME, output=True, figures=True
)

# Get the file names
path_to_data, file_extension, file_names = get_file_names(
    PATHS_TO_DATA, FILE_EXTENSIONS, path_number=0
)

# Get the file
for file_name in sorted(file_names):
    catalog = pd.read_parquet(os.path.join(path_to_data, file_name))
    short_name = file_name[: -len(file_extension)]

catalog = catalog[catalog["Magnitude"] > -2]

catalog_instance = completeness_magnitude.Catalog(
    path_to_figures, catalog, delta_magnitude=0.2
)

catalog_instance.plot_magnitude_distribution()

Mc = catalog_instance.maximum_curvature()
Mc = catalog_instance.goodness_of_fit_test()
