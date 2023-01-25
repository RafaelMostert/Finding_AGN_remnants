__author__ = "rafaël mostert"
__credits__ = ["rafaël mostert", "kai polsterer"]
__email__ = "mostert@strw.leidenuniv.nl"

from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting
from astropy.table import Table
from astropy.nddata.utils import PartialOverlapError, NoOverlapError
from astropy.io.votable import parse_single_table
from astropy.cosmology import Planck15 as cosmo
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate
from scipy.spatial import cKDTree
from scipy.interpolate import interp2d

# from astropy.convolution import Gaussian2DKernel
# from astropy.convolution import convolve
from io import BytesIO
import astropy.visualization as vis
import collections
import heapq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Rectangle, Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import os
import re
import pandas as pd
import scipy
import seaborn as sns
import struct
import subprocess
import sys
import requests
from urllib.request import urlopen
import warnings
import time
import pickle
import signal
import ftplib
import copy

"""
Notes on terminology.
    BMU: Best Matching Unit
    SOM: Self-Organizing Map
    unit, node, prototype: A neuron of a SOM
"""


class SOM(object):
    """Class to represent a (trained) SOM object."""

    def __init__(
        self,
        data_som_,
        number_of_channels_,
        som_width_,
        som_height_,
        som_depth_,
        layout_,
        output_directory_,
        trained_subdirectory_,
        som_label_,
        rotated_size_,
        run_id_,
        version=1,
    ):
        self.run_id = run_id_
        self.version = version
        self.data_som = data_som_
        self.number_of_channels = number_of_channels_
        self.som_width = som_width_
        self.som_height = som_height_
        self.som_depth = som_depth_
        self.layout = layout_
        self.output_directory = output_directory_
        self.trained_subdirectory = trained_subdirectory_
        self.som_label = som_label_
        self.rotated_size = int(round(rotated_size_))
        self.fullsize = int(np.ceil(self.rotated_size * np.sqrt(2)))
        # Training parameters
        if self.som_width != None or self.som_height != None:
            self.gauss_start = max(self.som_width, self.som_height) / 2
        if self.layout == "quadratic":
            self.layout_v2 = 0
        else:
            self.layout_v2 = 1
        self.learning_constraint = 0.05
        self.epochs_per_epoch = 1
        self.gauss_decrease = 0.95
        self.gauss_end = 0.3
        self.pbc = False
        self.learning_constraint_decrease = 0.95
        self.random_seed = 42
        self.init = "zero"
        self.pix_angular_res = 1.5
        self.rotated_size_in_arcsec = self.rotated_size * self.pix_angular_res
        self.fullsize_in_arcsec = self.fullsize * self.pix_angular_res
        self.training_dataset_name = ""
        self.flip_axis0 = False
        self.flip_axis1 = False
        self.rot90 = False
        self.save()

    def print(self):
        print(f"\nSOM ID{self.run_id} info")
        print(
            f"Input data dimensions: ({self.number_of_channels}x{self.fullsize}x{self.fullsize})"
        )
        print(
            f"Neuron/prototype dimensions: ({self.number_of_channels}x{self.rotated_size}x{self.rotated_size})"
        )
        print(
            f"SOM dimensions: ({self.som_width}x{self.som_height}x{self.som_depth}), layout {self.layout}"
        )
        print(f"Train parameters:")
        print(f"Periodic boundary conditions: {self.pbc}")
        print(
            f"learn. constr. {self.learning_constraint}, decrease {self.learning_constraint_decrease}"
        )
        print(
            f"Neighbourhood function: start {self.gauss_start}, decrease {self.gauss_decrease},"
            f" stop {self.gauss_end}\n"
        )
        print(f"Output dir: {self.output_directory}")
        print(f"Trained subdir: {self.trained_subdirectory}")

    def save(self):
        """save class pkl"""
        save_path = os.path.join(
            self.output_directory, f"SOM_object_id{self.run_id}.pkl"
        )
        # print(f"saving at {save_path}")
        with open(save_path, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


class CutoutSettings:
    def __init__(
        self,
        experiment=None,
        run_id=None,
        store_filename=None,
        fullsize=None,
        data_dir=None,
        arcsec_per_pixel=None,
        overwrite=True,
        apply_clipping=True,
        lower_sigma_limit=None,
        upper_sigma_limit=None,
        rescale=True,
        normalize=False,
        zoom_in=True,
        variable_size=False,
        map_to_run_id=None,
        map_to_binpath=None,
        flip_axis1=False,
        flip_axis2=False,
        rot90=False,
    ):
        self.experiment = experiment
        self.run_id = run_id
        self.store_filename = store_filename
        self.fullsize = fullsize
        self.data_dir = data_dir
        self.arcsec_per_pixel = arcsec_per_pixel
        self.overwrite = overwrite
        self.apply_clipping = apply_clipping
        self.lower_sigma_limit = lower_sigma_limit
        self.upper_sigma_limit = upper_sigma_limit
        self.variable_size = variable_size
        self.map_to_run_id = map_to_run_id
        self.map_to_binpath = map_to_binpath
        self.map_path = None
        self.normalize = normalize
        self.zoom_in = zoom_in
        self.som = None
        self.flip_axis1 = flip_axis1
        self.flip_axis2 = flip_axis2
        self.rot90 = rot90
        self.rescale = rescale
        self.highlight_neurons = [[], [], [], [(8, 8)]]
        self.highlight_colors = [[], [], [], ["red"]]
        self.legendlist = [[], [], [], ["possible AGN remnants"]]

    def save(self, save_dir):
        save_path = os.path.join(save_dir, f"SOM_settings_object_id{self.run_id}.pkl")
        with open(save_path, "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def load_SOM(output_directory, run_id):
    """Load SOM object"""
    som_path = os.path.join(output_directory, f"SOM_object_id{run_id}.pkl")
    assert os.path.getsize(som_path) > 0, "SOM object is empty"

    with open(som_path, "rb") as input:
        return pickle.load(input)


def load_pickle(path):
    """Load pickle object"""
    with open(path, "rb") as input:
        return pickle.load(input)


def identity(x):
    return x


def write_about_trained_som(
    som,
    data_map,
    website_path,
    run_id,
    bin_filename,
    catalogue_download_link,
    pickle_link,
    pandas_catalogue,
    pandas_version,
    full_dataset=False,
    bin_filename2=None,
    pandas_catalogue2=None,
    catalogue_download_link2=None,
    pickle_link2=None,
):
    """Writes info about the SOM and its training set to a text file.
    Can be used by the PINK-LOFAR Visualization web tool."""

    som_names = [
        "Number of channels",
        "SOM dimensions",
        "Layout",
        "Node/prototype size",
        "Periodic boundary conditions",
        "Neighbourhood radius start | decrease | end",
        "Learning constraint start | decrease",
        "Initialization",
    ]
    som_values = [
        som.number_of_channels,
        str(som.som_width)
        + "&#215;"
        + str(som.som_height)
        + "&#215;"
        + str(som.som_depth),
        som.layout,
        str(som.rotated_size)
        + "&#215;"
        + str(som.rotated_size)
        + " pixels or "
        + str(som.rotated_size_in_arcsec)
        + "&#215;"
        + str(som.rotated_size_in_arcsec)
        + " arcsec",
        som.pbc,
        str(som.gauss_start)
        + " | "
        + str(som.gauss_decrease)
        + " | "
        + str(som.gauss_end),
        str(som.learning_constraint) + " | " + str(som.learning_constraint_decrease),
        som.init,
    ]
    dataset_names = [
        "Data set name",
        "Number of cut-outs",
        "Size of cut-outs",
        "Download catalogue + mapping",
    ]
    dataset_values = [
        bin_filename,
        len(pandas_catalogue),
        str(som.fullsize)
        + "&#215;"
        + str(som.fullsize)
        + " pixels or "
        + str(som.fullsize_in_arcsec)
        + "&#215;"
        + str(som.fullsize_in_arcsec)
        + " arcsec",
        '<a href="{}">CSV</a> or <a href="{}">pickle-format</a> (pandas version: {})'.format(
            catalogue_download_link, pickle_link, pandas_version
        ),
    ]
    if full_dataset:
        full_dataset_names = [
            "Data set name",
            "Number of cut-outs",
            "Size of cut-outs",
            "Download catalogue + mapping",
        ]
        full_dataset_values = [
            bin_filename2,
            len(pandas_catalogue2),
            str(som.fullsize)
            + "&#215;"
            + str(som.fullsize)
            + " pixels or "
            + str(som.fullsize_in_arcsec)
            + "&#215;"
            + str(som.fullsize_in_arcsec)
            + " arcsec",
            '<a href="{}">CSV</a> or <a href="{}">pickle-format</a> (pandas version: {})'.format(
                catalogue_download_link2, pickle_link2, pandas_version
            ),
        ]
    performance_names = ["Average Quantization Error", "Topological Error"]
    performance_values = [
        str(round((calculate_AQEs(som.rotated_size, [data_map])[0][0]), 1))
        + "&#177;"
        + str(round(calculate_AQEs(som.rotated_size, [data_map])[1][0], 1)),
        round(
            calculate_TEs(
                [data_map], som.som_width, som.som_height, som.pbc, som.layout
            )[0],
            1,
        ),
    ]

    # Open and write the file
    about_path = os.path.join(website_path, "about_som.html")
    with open(about_path, "w") as f:
        f.write(
            """<table class="table table-striped table-sm">
        <thead>
            <tr>
            <th colspan="2">SOM properties</td>
            </tr>
          </thead>
          <tbody>"""
        )
        for n, t in zip(som_names, som_values):
            f.write("<tr> <td>{}</td><td>{}</td></tr>".format(n, t))
        f.write(
            """<thead> <tr> <th colspan="2">Training data set properties</td>
            </tr>
          </thead>"""
        )
        for n, t in zip(dataset_names, dataset_values):
            f.write("<tr> <td>{}</td><td>{}</td></tr>".format(n, t))
        f.write(
            """<thead> <tr> <th colspan="2">Full data set properties</td>
            </tr>
          </thead>"""
        )
        for n, t in zip(full_dataset_names, full_dataset_values):
            f.write("<tr> <td>{}</td><td>{}</td></tr>".format(n, t))
        f.write(
            """<thead> <tr> <th colspan="2">Performance evaluation</td>
            </tr>
          </thead>"""
        )
        for n, t in zip(performance_names, performance_values):
            f.write("<tr> <td>{}</td><td>{}</td></tr>".format(n, t))
        f.write("</tbody></table>")


def write_bash_script_to_run_pink(
    som,
    run_id,
    bin_filename,
    gpu_id,
    data_dir,
    output_dir,
    bash_path,
    verbose=True,
    pink_command="Pink",
):
    """Write bash script to run pink, automatically map the training set to
    the SOM after every epoch. After every epoch, new values for neighborhood radius
    and learning constraint can be set."""

    if som.pbc:
        pbc = 'PBC="--pbc" '
    else:
        pbc = 'PBC=" " '
    learning_constraint = som.learning_constraint * som.gauss_start * np.sqrt(2 * np.pi)
    initial_lines = """
#!/bin/bash
# Script to decrease width of gaussian

# Hyper-parameters, tune these
RUN_ID="_ID{}"
W={} # Width of the SOM
H={} # Height of the SOM
GAUSS={} # Initial gaussian width
SIGMA={}
GPU_ID={}
EPOCHS_PER_EPOCH={}
ALPHA={} # Should be between 0 and 1
FullSize={}
RotatedSize={}
LAYOUT={}
{} # Periodic boundary conditions
BIN={}
DATA_DIR={}
OUTPUT_DIR={}
LEARNING_CONSTRAINT_START={}
LEARNING_CONSTRAINT_DECREASE={}
echo "Starting Pink script"
echo "Using binary file: $BIN.bin"
""".format(
        run_id,
        som.som_width,
        som.som_height,
        som.gauss_start,
        learning_constraint,
        gpu_id,
        som.epochs_per_epoch,
        som.gauss_decrease,
        som.fullsize,
        som.rotated_size,
        som.layout,
        pbc,
        bin_filename,
        data_dir,
        output_dir,
        som.learning_constraint,
        som.learning_constraint_decrease,
    )

    first_loop = f"""
clear
o="_"
x="x"
echo "Starting Pink script"
echo "Using binary file: $BIN.bin"
#echo "Output directory: $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/"
# Create output directory if it doesn't exist yet
if [ ! -d "$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/" ]; then
    mkdir $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID;
else
    echo "Output folder already exists, proceeding to training.";
fi
# Create runlogs directory if it doesn't exist yet
if [ ! -d "$OUTPUT_DIR/run_logs/" ]; then
    mkdir $OUTPUT_DIR/run_logs;
else
    echo "Run_logs folder already exists, proceeding to training.";
fi

# Copy this script to the output directory, to archive the used parameters
cp ${{0}} $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/${{0##*/}}

# First cycle with full size neurons and no initialization
if [ ! -f $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin ]; then
    echo "Epoch 1/$END: Gaussian = $GAUSS"
    echo "CUDA_VISIBLE_DEVICES=$GPU_ID {pink_command} --seed {som.random_seed} --dist-func gaussian $GAUSS $SIGMA --inter-store overwrite \
--neuron-dimension $RotatedSize --numrot 360 --num-iter $EPOCHS_PER_EPOCH --progress 0.1 --som-width $W \
--som-height $H --layout $LAYOUT  $PBC --init \
{som.init} --train $DATA_DIR/$BIN.bin \
$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > $OUTPUT_DIR/run_logs/out_run$RUN_ID.txt"
    CUDA_VISIBLE_DEVICES=$GPU_ID {pink_command} --seed {som.random_seed} --dist-func gaussian $GAUSS $SIGMA --inter-store overwrite \
--neuron-dimension $RotatedSize --numrot 360 --num-iter $EPOCHS_PER_EPOCH --progress 0.1 --som-width $W \
--som-height $H --layout $LAYOUT  $PBC --init \
{som.init} --train $DATA_DIR/$BIN.bin \
$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > $OUTPUT_DIR/run_logs/out_run$RUN_ID.txt
else
    echo "Previously trained SOM found, skipping epoch 1.";
fi

# Perform first mapping on data
if [ ! -f "$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin" ];
then
    echo "Start train mapping first file";
    # Start mapping binary to last result
    CUDA_VISIBLE_DEVICES=$GPU_ID {pink_command} --inter-store overwrite \
	--neuron-dimension $RotatedSize --numrot 360 --progress 0.1 \
	--som-width $W --som-height $H --layout $LAYOUT --map $DATA_DIR/$BIN.bin \
	$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin \
	$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > $OUTPUT_DIR/run_logs/out_map$i$RUN_ID.txt;
else
    echo "Previous mapping result found, skipping first mapping.";
fi"""

    with open(bash_path, "w") as f:
        f.write(initial_lines)
        f.write(first_loop)
        gauss = som.gauss_start * som.gauss_decrease
        if verbose:
            print(
                "start, and end gauss, and first newgauss:",
                som.gauss_start,
                som.gauss_end,
                gauss,
            )
        learning_constraint_decrease = som.learning_constraint_decrease

        # Check number of runs needed to get to gauss_end
        i = 1
        end = 1
        g = som.gauss_start * som.gauss_decrease
        while g > som.gauss_end:
            g *= som.gauss_decrease
            end += 1

        # Get list of declining learning constraints
        learning_constraints = declining_learning_constraint(
            som.learning_constraint, som.learning_constraint_decrease, end - 1
        )

        # Loop until gauss declined so that gauss_end is met
        while gauss > som.gauss_end:
            learning_constraint = (
                learning_constraints[i - 1] * gauss * np.sqrt(2 * np.pi)
            )
            i += 1
            if verbose:
                print(
                    "newgauss:",
                    round(gauss, 2),
                    "new learning constraint:",
                    learning_constraint,
                    "start lc:",
                    som.learning_constraint,
                    "lc decrease:",
                    som.learning_constraint_decrease,
                    "Nt:",
                    learning_constraint / (gauss * np.sqrt(2 * np.pi)),
                )
            latter_loop = f"""
            i={i}
            END={end}
            # Train phase
                old_GAUSS=$GAUSS
                GAUSS={gauss} # Decrease gaussian width
                old_SIGMA=$SIGMA
                SIGMA={learning_constraint} # Decrease

                if [ ! -f $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin ]; then
                    echo "Epoch $i/$END: Gaussian = $GAUSS";

                    CUDA_VISIBLE_DEVICES=$GPU_ID {pink_command} --seed {som.random_seed} \
                    --dist-func gaussian $GAUSS {learning_constraint} --inter-store overwrite \
            --neuron-dimension $RotatedSize --numrot 360 --num-iter $EPOCHS_PER_EPOCH --progress 0.1 --som-width $W \
            --som-height $H --layout $LAYOUT $PBC --init \
            $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$old_GAUSS$o$old_SIGMA$RUN_ID.bin  --train $DATA_DIR/$BIN.bin \
            $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > $OUTPUT_DIR/run_logs/out_run$i$RUN_ID.txt;
                else
                    echo "Previously trained SOM found, skipping epoch $i.";
                fi

            # Mapping phase
                if [ ! -f $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin ]; then
                    echo "Start train mapping $i/$END:";
                    # Start mapping binary to last result
                    CUDA_VISIBLE_DEVICES=$GPU_ID {pink_command} --inter-store overwrite \
                        --neuron-dimension $RotatedSize --numrot 360 --progress 0.1 \
                        --som-width $W --som-height $H --layout $LAYOUT --map $DATA_DIR/$BIN.bin \
                        $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin \
                        $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > $OUTPUT_DIR/run_logs/out_map$i$RUN_ID.txt;
                else
                    echo "Previous mapping result found, skipping final mapping.";
                fi
            """
            f.write(latter_loop)
            gauss *= som.gauss_decrease

        final_string = f"""
        echo "Last SOM written to file: result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin";
        echo "Last mapping written to file as well.";
        echo "Done.";

        echo "CUDA_VISIBLE_DEVICES=$GPU_ID {pink_command} --seed {som.random_seed} --dist-func gaussian $GAUSS $SIGMA --inter-store overwrite \
        --neuron-dimension $RotatedSize --numrot 360 --num-iter $EPOCHS_PER_EPOCH --progress 0.1 --som-width $W \
        --som-height $H --layout $LAYOUT $PBC --init \
         {som.init} --train $DATA_DIR/$BIN.bin \
        $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin";

        """
        f.write(final_string)


def write_bash_script_to_run_pink_v2(
    som,
    run_id,
    bin_filename,
    gpu_id,
    data_dir,
    output_dir,
    bash_path,
    verbose=True,
    small_som_size=False,
):
    """Write bash script to run pink, automatically map the training set to
    the SOM after every epoch. After every epoch, new values for neighborhood radius
    and learning constraint can be set."""

    total_string = ""

    if som.layout == "quadratic":
        layout = "cartesian"
    else:
        layout = som.layout

    pbc = ""
    if som.pbc:
        pbc = "--pbc"
        raise NotImplementedError("PBC is not yet implemented in PINK v2.")
    if small_som_size:
        neuron_size = som.rotated_size
        print("neuron size:", neuron_size)
    else:
        neuron_size = int(np.ceil(som.rotated_size * np.sqrt(2)))

    learning_constraint = som.learning_constraint * som.gauss_start * np.sqrt(2 * np.pi)
    initial_lines = """
#!/bin/bash
# Script to decrease width of gaussian

# Hyper-parameters, tune these
RUN_ID="_ID{}"
W={} # Width of the SOM
H={} # Height of the SOM
D={} # Depth of the SOM
GAUSS={} # Initial gaussian width
SIGMA={}
GPU_ID={}
EPOCHS_PER_EPOCH={}
ALPHA={} # Should be between 0 and 1
FullSize={}
RotatedSize={}
LAYOUT={}
#PBC={} # Periodic boundary conditions
BIN={}
DATA_DIR={}
OUTPUT_DIR={}
LEARNING_CONSTRAINT_START={}
LEARNING_CONSTRAINT_DECREASE={}
echo "Starting Pink script"
echo "Using binary file: $BIN.bin"
""".format(
        run_id,
        som.som_width,
        som.som_height,
        som.som_depth,
        som.gauss_start,
        learning_constraint,
        gpu_id,
        som.epochs_per_epoch,
        som.gauss_decrease,
        som.fullsize,
        neuron_size,
        layout,
        pbc,
        bin_filename,
        data_dir,
        output_dir,
        som.learning_constraint,
        som.learning_constraint_decrease,
    )

    first_loop = f"""
clear
o="_"
x="x"
echo "Starting Pink script"
echo "Using binary file: $BIN.bin"
#echo "Output directory: $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/"
# Create output directory if it doesn't exist yet
if [ ! -d "$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/" ]; then
    mkdir $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID;
else
    echo "Output folder already exists, proceeding to training.";
fi
# Create runlogs directory if it doesn't exist yet
if [ ! -d "$OUTPUT_DIR/run_logs/" ]; then
    mkdir $OUTPUT_DIR/run_logs;
else
    echo "Run_logs folder already exists, proceeding to training.";
fi

# Copy this script to the output directory, to archive the used parameters
cp ${{0}} $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/${{0##*/}}

# First cycle with full size neurons and no initialization
if [ ! -f $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin ]; then
    echo "Epoch 1/$END: Gaussian = $GAUSS"
    echo "CUDA_VISIBLE_DEVICES=$GPU_ID Pink --euclidean-distance-type float \
    --euclidean-distance-dimension {som.rotated_size} --seed {som.random_seed} --dist-func gaussian $GAUSS $SIGMA \
--neuron-dimension $RotatedSize --numrot 360 --num-iter $EPOCHS_PER_EPOCH --progress 10 --som-width $W \
--som-height $H --som-depth $D --layout $LAYOUT  --init {som.init} \
 --train $DATA_DIR/$BIN.bin \
$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > $OUTPUT_DIR/run_logs/out_run$RUN_ID.txt";
    CUDA_VISIBLE_DEVICES=$GPU_ID Pink --euclidean-distance-type float --euclidean-distance-dimension {som.rotated_size} --seed {som.random_seed} --dist-func gaussian $GAUSS $SIGMA \
--neuron-dimension $RotatedSize --numrot 360 --num-iter $EPOCHS_PER_EPOCH --progress 10 --som-width $W \
--som-height $H --som-depth $D --layout $LAYOUT  --init {som.init} \
 --train $DATA_DIR/$BIN.bin \
$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > $OUTPUT_DIR/run_logs/out_run$RUN_ID.txt
else
    echo "Previously trained SOM found, skipping epoch 1.";
fi

# Perform first mapping on data
if [ ! -f "$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin" ];
then
    echo "Start train mapping first file";
    # Start mapping binary to last result
    echo "CUDA_VISIBLE_DEVICES=$GPU_ID Pink  --euclidean-distance-type float \
	 --euclidean-distance-dimension {som.rotated_size} --neuron-dimension $RotatedSize --numrot 360 --progress 10 \
	--som-width $W --som-height $H --som-depth $D --layout $LAYOUT --map $DATA_DIR/$BIN.bin \
	$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin \
	$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin";
    CUDA_VISIBLE_DEVICES=$GPU_ID Pink  --euclidean-distance-type float \
	 --euclidean-distance-dimension {som.rotated_size} --neuron-dimension $RotatedSize --numrot 360 --progress 10 \
	--som-width $W --som-height $H --som-depth $D --layout $LAYOUT --map $DATA_DIR/$BIN.bin \
	$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin \
	$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > $OUTPUT_DIR/run_logs/out_map$i$RUN_ID.txt;
else
    echo "Previous mapping result found, skipping first mapping.";
fi"""

    with open(bash_path, "w") as f:
        f.write(initial_lines)
        f.write(first_loop)
        total_string += initial_lines
        total_string += first_loop
        gauss = som.gauss_start * som.gauss_decrease
        if verbose:
            print(
                "start, and end gauss, and first newgauss:",
                som.gauss_start,
                som.gauss_end,
                gauss,
            )
        learning_constraint_decrease = som.learning_constraint_decrease

        # Check number of runs needed to get to gauss_end
        i = 1
        end = 1
        g = som.gauss_start * som.gauss_decrease
        while g > som.gauss_end:
            g *= som.gauss_decrease
            end += 1

        # Get list of declining learning constraints
        learning_constraints = declining_learning_constraint(
            som.learning_constraint, som.learning_constraint_decrease, end - 1
        )

        # Loop until gauss declined so that gauss_end is met
        while gauss > som.gauss_end:
            learning_constraint = (
                learning_constraints[i - 1] * gauss * np.sqrt(2 * np.pi)
            )
            i += 1
            if verbose:
                print(
                    "newgauss:",
                    round(gauss, 2),
                    "new learning constraint:",
                    learning_constraint,
                    "start lc:",
                    som.learning_constraint,
                    "lc decrease:",
                    som.learning_constraint_decrease,
                    "Nt:",
                    learning_constraint / (gauss * np.sqrt(2 * np.pi)),
                )
            latter_loop = f"""
            i={i}
            END={end}
            # Train phase
                old_GAUSS=$GAUSS
                old_SIGMA=$SIGMA
                GAUSS={gauss} # Decrease gaussian width
                SIGMA={learning_constraint} # Decrease gaussian width

                if [ ! -f $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin ]; then
                    echo "Epoch $i/$END: Gaussian = $GAUSS";

                    CUDA_VISIBLE_DEVICES=$GPU_ID Pink --euclidean-distance-type float --seed \
                    {som.random_seed}  --euclidean-distance-dimension  {som.rotated_size} --dist-func gaussian {gauss} {learning_constraint} \
            --neuron-dimension $RotatedSize --numrot 360 --num-iter $EPOCHS_PER_EPOCH --progress 10 --som-width $W \
            --som-height $H --som-depth $D --layout $LAYOUT --init \
            $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$old_GAUSS$o$old_SIGMA$RUN_ID.bin  --train $DATA_DIR/$BIN.bin \
            $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > $OUTPUT_DIR/run_logs/out_run$i$RUN_ID.txt;
                else
                    echo "Previously trained SOM found, skipping epoch $i.";
                fi

            # Mapping phase
                if [ ! -f $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin ]; then
                    echo "Start train mapping $i/$END:";
                    # Start mapping binary to last result
                    CUDA_VISIBLE_DEVICES=$GPU_ID Pink   --euclidean-distance-dimension {som.rotated_size} --euclidean-distance-type float \
                        --neuron-dimension $RotatedSize --numrot 360 --progress 10 \
                        --som-width $W --som-height $H --som-depth $D --layout $LAYOUT --map $DATA_DIR/$BIN.bin \
                        $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin \
                        $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > $OUTPUT_DIR/run_logs/out_map$i$RUN_ID.txt;
                else
                    echo "Previous mapping result found, skipping final mapping.";
                fi
            """
            f.write(latter_loop)
            total_string += latter_loop
            gauss *= som.gauss_decrease

        final_string = f"""
        echo "Last SOM written to file: result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin";
        echo "Last mapping written to file as well.";
        echo "Done.";

        echo "CUDA_VISIBLE_DEVICES=$GPU_ID Pink --euclidean-distance-type float --euclidean-distance-dimension {som.rotated_size} --seed {som.random_seed} --dist-func gaussian {gauss} $SIGMA  \
        --neuron-dimension $RotatedSize --numrot 360 --num-iter $EPOCHS_PER_EPOCH --progress 10 --som-width $W \
        --som-height $H --som-depth $D --layout $LAYOUT --init \
         {som.init} --train $DATA_DIR/$BIN.bin \
        $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin";

        """
        f.write(final_string)
        total_string += final_string
    return total_string


def write_bash_script_to_run_pink_train_test(
    som, run_id, bin_filename, gpu_id, data_dir, output_dir, bash_path
):
    learning_constraint = som.learning_constraint * som.gauss_start * np.sqrt(2 * np.pi)
    initial_lines = """
#!/bin/bash
# Script to decrease width of gaussian

# Hyper-parameters, tune these
RUN_ID="_ID{}"
W={} # Width of the SOM
H={} # Height of the SOM
GAUSS={} # Initial gaussian width
SIGMA={}
GPU_ID={}
EPOCHS_PER_EPOCH={}
ALPHA={} # Should be between 0 and 1
FullSize={}
RotatedSize={}
LAYOUT={}
PBC={} # Periodic boundary conditions
BIN={}
train_text="_train"
test_text="_test"
TRAIN_BIN="$BIN$train_text"
TEST_BIN="$BIN$test_text"
DATA_DIR={}
OUTPUT_DIR={}
LEARNING_CONSTRAINT_START={}
LEARNING_CONSTRAINT_DECREASE={}
echo "Starting Pink script"
echo "Using binary file: $TRAIN_BIN.bin"
""".format(
        run_id,
        som.som_width,
        som.som_height,
        som.gauss_start,
        learning_constraint,
        gpu_id,
        som.epochs_per_epoch,
        som.gauss_decrease,
        som.fullsize,
        som.rotated_size,
        som.layout,
        som.pbc,
        bin_filename,
        data_dir,
        output_dir,
        som.learning_constraint,
        som.learning_constraint_decrease,
    )

    first_loop = """
clear
o="_"
x="x"
echo "Starting Pink script"
echo "Using binary file: $TRAIN_BIN.bin"
#echo "Output directory: $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/"
# Create output directory if it doesn't exist yet
if [ ! -d "$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/" ]; then
    mkdir $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID;
else
    echo "Output folder already exists, proceeding to training.";
fi

# Copy this script to the output directory, to archive the used parameters
cp ${{0}} $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/${{0##*/}}

# First cycle with full size neurons and no initialization
if [ ! -f $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin ]; then
    echo "Epoch 1/$END: Gaussian = $GAUSS"
    CUDA_VISIBLE_DEVICES=$GPU_ID Pink --seed {} --dist-func gaussian $GAUSS $SIGMA --inter-store overwrite \
--neuron-dimension $RotatedSize --numrot 360 --num-iter $EPOCHS_PER_EPOCH --progress 0.1 --som-width $W \
--som-height $H --layout $LAYOUT --pbc $PBC --init \
    {} --train $DATA_DIR/$TRAIN_BIN.bin \
$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > run_logs/out_run$RUN_ID.txt
else
    echo "Previously trained SOM found, skipping epoch 1.";
fi

# Perform first mapping on train data
if [ ! -f "$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin" ];
then
    echo "Start train mapping first file";
    # Start mapping binary to last result
    CUDA_VISIBLE_DEVICES=$GPU_ID Pink --inter-store overwrite \
	--neuron-dimension $RotatedSize --numrot 360 --progress 0.1 \
	--som-width $W --som-height $H --layout $LAYOUT --map $DATA_DIR/$TRAIN_BIN.bin \
	$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin \
	$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > run_logs/out_map$i$RUN_ID.txt;
else
    echo "Previous mapping result found, skipping first mapping.";
fi


# Perform first mapping on test data
if [ ! -f "$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$TEST_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin" ];
then
    echo "Start Test mapping first file";
    # Start mapping binary to last result
    CUDA_VISIBLE_DEVICES=$GPU_ID Pink --inter-store overwrite \
	--neuron-dimension $RotatedSize --numrot 360 --progress 0.1 \
	--som-width $W --som-height $H --layout $LAYOUT --map $DATA_DIR/$TEST_BIN.bin \
	$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$TEST_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin \
	$OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > run_logs/out_map$i$RUN_ID.txt;
else
    echo "Previous mapping result found, skipping first mapping.";
fi


""".format(
        som.random_seed, som.init
    )

    with open(bash_path, "w") as f:
        f.write(initial_lines)
        f.write(first_loop)
        gauss = som.gauss_start * som.gauss_decrease
        print(
            "start, and end gauss, and first newgauss:",
            som.gauss_start,
            som.gauss_end,
            gauss,
        )
        learning_constraint_decrease = som.learning_constraint_decrease

        # Check number of runs needed to get to gauss_end
        i = 1
        end = 1
        g = som.gauss_start * som.gauss_decrease
        while g > som.gauss_end:
            g *= som.gauss_decrease
            end += 1

        # Get list of declining learning constraints
        learning_constraints = declining_learning_constraint(
            som.learning_constraint, som.learning_constraint_decrease, end - 1
        )

        # Loop until gauss declined so that gauss_end is met
        while gauss > som.gauss_end:
            learning_constraint = (
                learning_constraints[i - 1] * gauss * np.sqrt(2 * np.pi)
            )
            i += 1
            print(
                "newgauss:",
                round(gauss, 2),
                "new learning constraint:",
                learning_constraint,
                "start lc:",
                som.learning_constraint,
                "lc decrease:",
                som.learning_constraint_decrease,
                "Nt:",
                learning_constraint / (gauss * np.sqrt(2 * np.pi)),
            )
            latter_loop = """
            i={}
            END={}
            # Train phase
                old_GAUSS=$GAUSS
                GAUSS={} # Decrease gaussian width

                if [ ! -f $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin ]; then
                    echo "Epoch $i/$END: Gaussian = $GAUSS";

                    CUDA_VISIBLE_DEVICES=$GPU_ID Pink --seed {} --dist-func gaussian $GAUSS {} --inter-store overwrite \
            --neuron-dimension $RotatedSize --numrot 360 --num-iter $EPOCHS_PER_EPOCH --progress 0.1 --som-width $W \
            --som-height $H --layout $LAYOUT --pbc $PBC --init \
            $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$old_GAUSS$o$SIGMA$RUN_ID.bin  --train $DATA_DIR/$TRAIN_BIN.bin \
            $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > run_logs/out_run$i$RUN_ID.txt;
                else
                    echo "Previously trained SOM found, skipping epoch $i.";
                fi

            # Mapping phase
                if [ ! -f $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin ]; then
                    echo "Start train mapping $i/$END:";
                    # Start mapping binary to last result
                    CUDA_VISIBLE_DEVICES=$GPU_ID Pink --inter-store overwrite \
                        --neuron-dimension $RotatedSize --numrot 360 --progress 0.1 \
                        --som-width $W --som-height $H --layout $LAYOUT --map $DATA_DIR/$TRAIN_BIN.bin \
                        $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin \
                        $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > run_logs/out_map$i$RUN_ID.txt;
                else
                    echo "Previous mapping result found, skipping final mapping.";
                fi



# map test
    if [ ! -f $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$TEST_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin ]; then
	echo "Start Test mapping $i/$END:";
	# Start mapping binary to last result
	CUDA_VISIBLE_DEVICES=$GPU_ID Pink --inter-store overwrite \
	    --neuron-dimension $RotatedSize --numrot 360 --progress 0.1 \
	    --som-width $W --som-height $H --layout $LAYOUT --map $DATA_DIR/$TEST_BIN.bin \
	    $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/mapping_result$TEST_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin \
	    $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin > run_logs/out_map$i$RUN_ID.txt;
    else
	echo "Previous mapping result found, skipping final mapping.";
    fi

            """.format(
                i, end, gauss, som.random_seed, learning_constraint
            )
            f.write(latter_loop)
            gauss *= som.gauss_decrease

        final_string = """
        echo "Last SOM written to file: result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin";
        echo "Last mapping written to file as well.";
        echo "Done.";

        echo "CUDA_VISIBLE_DEVICES=$GPU_ID Pink --seed {} --dist-func gaussian $GAUSS $SIGMA --inter-store overwrite \
        --neuron-dimension $RotatedSize --numrot 360 --num-iter $EPOCHS_PER_EPOCH --progress 0.1 --som-width $W \
        --som-height $H --layout $LAYOUT --pbc $PBC --init \
            {} --train $DATA_DIR/$TRAIN_BIN.bin \
        $OUTPUT_DIR/$BIN$o$W$x$H$RUN_ID/result$TRAIN_BIN$o$W$x$H$x$RotatedSize$o$GAUSS$o$SIGMA$RUN_ID.bin";

        """.format(
            som.random_seed, som.init
        )
        f.write(final_string)


def declining_learning_constraint(lc_start, lc_decline, n):
    """Generate list of length n with learning constraints of
    decreasing size."""
    learning_constraints = []
    tel = lc_start
    for i in range(n):
        learning_constraints.append(tel)
        tel *= lc_decline
    return learning_constraints


def map_to_som(
    som,
    trained_path,
    dataset_bin_path,
    mapping_path,
    gpu_id,
    run_id,
    overwrite=False,
    write_to_file=None,
):
    print("SUPERSEEDED BY map_dataset_to_trained_som function")
    """Print line to run pink."""

    if (not overwrite) and os.path.exists(mapping_path):
        print(
            "The mapping {}  already exists, use overwrite=True flag \
                to proceed anyway.".format(
                mapping_path
            )
        )
    else:
        map_string = """CUDA_VISIBLE_DEVICES={} Pink --inter-store overwrite \\ 
--neuron-dimension {} --numrot 360 --progress 1 \\
--som-width {} --som-height {} --layout {} --map \\ 
{} \\
{} \\
{} \\
        > run_logs/out_map_{} \
""".format(
            gpu_id,
            som.rotated_size,
            som.som_width,
            som.som_height,
            som.layout,
            dataset_bin_path,
            mapping_path,
            trained_path,
            run_id,
        )
        map_string = """CUDA_VISIBLE_DEVICES={} Pink --inter-store overwrite --neuron-dimension {} --numrot 360 --progress 1 --som-width {} --som-height {} --som-depth 1 --layout {} --map {} {} {}""".format(
            gpu_id,
            som.rotated_size,
            som.som_width,
            som.som_height,
            som.layout,
            dataset_bin_path,
            mapping_path,
            trained_path,
        )
        # As gpu's are often unavailable (check by using nvidia-smi),
        # executing this printed line has to be done manually by the user
        print(map_string)
        if not write_to_file is None:
            with open(write_to_file, "w") as f:
                f.write(map_string)
    return map_string


def find_index(lst, a):
    """return indices of elements in lst that match a"""
    indices = np.array([i for i, x in enumerate(lst) if x == a])
    return indices


def find_nearest(array, target):
    """given numpy array  and a number of targets target,
    returns the index of the values in the array closest to target"""
    array = np.asarray(array)
    idx = (np.abs(array - target)).argmin()
    return idx  # , array[idx]


def find_closest(A, target):
    """given numpy array A and a number of targets target,
    returns the index of the values in A closest to target"""
    # A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A) - 1)
    left = A[idx - 1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


def calculate_AQE_per_prototype(
    som_map,
    rotated_pixel_size,
    cutout_id,
    verbose=False,
    comparative_datamap=[],
    percentiles_color="red",
    som_height=None,
    ax=None,
):
    """Returns Quantization Error per prototype for specific source,
    which is defined as the average Euclidean distance from each input of the
    training set to each corresponding best matching node/prototype/neuron."""
    bmus = np.argmin(som_map, axis=1)
    # plt.hist(np.min(som_map, axis=1))
    # plt.show()
    min_values = np.min(som_map, axis=1)
    # plt.hist(min_values)
    # plt.show()
    # assert (min_values <= 1).all() and (min_values >= 0).all()

    # Get AQE per node (normalized by the number of pixels in the node
    QE_per_node = [[] for _ in range(len(som_map[0]))]
    comparative_QE_per_node = [[] for _ in range(len(som_map[0]))]
    for i, (bmu, min_value) in enumerate(zip(bmus, min_values)):
        QE_per_node[bmu].append(min_value)
    QE_per_node = np.array(QE_per_node)

    AQE_per_node = np.array(list(map(np.mean, QE_per_node)))
    AQE_std_per_node = np.array(list(map(np.std, QE_per_node)))

    if not comparative_datamap == []:
        # print('Red stands for remnant')
        comp_bmu = np.argmin(comparative_datamap)
        comp_min_value = np.min(comparative_datamap)
        # Do something
        # for i, (bmu, min_value) in enumerate(zip(comp_bmus, comp_min_values)):
        #    comparative_QE_per_node[bmu].append(min_value)
        # comparative_QE_per_node = np.array(comparative_QE_per_node)

        percentiles_color = "black"
        comp_color = "red"

    QE = QE_per_node[comp_bmu]
    AQ, AQ_std = AQE_per_node[comp_bmu], AQE_std_per_node[comp_bmu]

    n, bins, _ = ax.hist(QE, bins="sqrt", histtype="step", linewidth=3)
    # Plot median and percentiles
    percentiles = [5, 25, 50, 75, 95]
    x = np.percentile(QE, percentiles)
    median = np.percentile(QE, 50)

    ax.plot(x, 0.30 * max(n) * np.ones(len(x)), "o-", c=percentiles_color)
    ax.plot(median, 0.30 * max(n), "o", c=percentiles_color, markersize=10)
    [
        plt.text(xx, 0.37 * max(n), f"{p}th", color=percentiles_color)
        for xx, p in zip(x, percentiles)
    ]

    # Plot comparative map points
    plt.plot(comp_min_value, 0.60 * max(n), "o", c=comp_color, markersize=10)

    plt.xlabel("Euclidean norm to best matching neuron")
    plt.ylabel("Number of cutouts")

    return ax


def calculate_normalized_AQE_per_prototype(
    som_map,
    rotated_pixel_size,
    plot_mean=False,
    verbose=False,
    cutouts_normed=False,
    comparative_datamap=[],
    percentiles_color="red",
    plot_only_comparative=False,
    save=False,
    save_dir=None,
    save_name=None,
    additional_text_for_paper=False,
    som_height=None,
    version=1,
    xlabel="Summed Euclidean distance between cutout and its best matching prototype / number of pixels",
    ylabel="Cutouts",
):
    """Returns Average Quantization Error per prototype,
    which is defined as the average Euclidean distance from each input of the
    training set to each corresponding best matching node/prototype/neuron.
    The values are normalized by dividing by the number of pixels"""
    bmus = np.argmin(som_map, axis=1)
    # plt.hist(np.min(som_map, axis=1))
    # plt.show()
    if not cutouts_normed:
        min_values = np.min(som_map, axis=1)
    else:
        print("NOTE: Normalizing all datamap values")
        min_values = np.min(som_map, axis=1) / (rotated_pixel_size**2)

    # plt.hist(min_values)
    # plt.show()
    print("")
    # assert (min_values <= 1).all() and (min_values >= 0).all()

    # Get AQE per node (normalized by the number of pixels in the node
    QE_per_node = [[] for _ in range(len(som_map[0]))]
    comparative_QE_per_node = [[] for _ in range(len(som_map[0]))]
    for i, (bmu, min_value) in enumerate(zip(bmus, min_values)):
        QE_per_node[bmu].append(min_value)
    QE_per_node = np.array(QE_per_node)

    AQE_per_node = np.array(list(map(np.mean, QE_per_node)))
    AQE_std_per_node = np.array(list(map(np.std, QE_per_node)))

    if verbose:
        if not comparative_datamap == []:
            print("Red stands for remnant")
            comp_bmus = np.argmin(comparative_datamap, axis=1)
            # comp_min_values = np.min(comparative_datamap, axis=1)/(rotated_pixel_size**2)
            comp_min_values = np.min(comparative_datamap, axis=1)
            # Do something
            for i, (bmu, min_value) in enumerate(zip(comp_bmus, comp_min_values)):
                comparative_QE_per_node[bmu].append(min_value)
            comparative_QE_per_node = np.array(comparative_QE_per_node)

            percentiles_color = "black"
            comp_color = "red"

        for i, (QE, comp_QE, AQ, AQ_std) in enumerate(
            zip(QE_per_node, comparative_QE_per_node, AQE_per_node, AQE_std_per_node)
        ):
            if QE == []:
                print(i, "No best matching sources in the original data set")
                if not comp_QE == []:
                    print("Comparison data set:", comp_QE)
                continue
            elif comp_QE == [] and plot_only_comparative:
                # print(i, 'No best matching sources in the comparative data set')
                continue
            else:
                print(f"Neuron location: ({int(i%som_height)}, {int(i/som_height)})")
            plt.figure(figsize=(10, 2))
            n, bins, _ = plt.hist(QE, bins="sqrt", histtype="step", linewidth=3)
            if plot_mean:
                # Plot mean and std devs.
                sigmas = [-2, -1, 0, 1, 2]
                x = AQ + np.array(sigmas) * AQ_std
                plt.plot(x, 0.1 * max(n) * np.ones(len(x)), "o-", c="orange")
                plt.plot(AQ, 0.1 * max(n), "o", c="orange", markersize=10)
                plt.text(AQ, 0.17 * max(n), "mean", color="orange")
                [
                    plt.text(xx, 0.17 * max(n), f"{s}\u03C3", color="orange")
                    for xx, s in zip(x, sigmas)
                    if not s == 0
                ]
            # Plot median and percentiles
            percentiles = [5, 25, 50, 75, 95]
            x = np.percentile(QE, percentiles)
            median = np.percentile(QE, 50)

            plt.plot(x, 0.30 * max(n) * np.ones(len(x)), "o-", c=percentiles_color)
            plt.plot(median, 0.30 * max(n), "o", c=percentiles_color, markersize=10)
            [
                plt.text(
                    xx,
                    0.37 * max(n),
                    f"{p}th",
                    color=percentiles_color,
                    horizontalalignment="center",
                    bbox=dict(
                        boxstyle="Round,pad=0.0",
                        edgecolor="white",
                        facecolor="white",
                        alpha=0.5,
                    ),
                )
                if ip % 2 == 0
                else plt.text(
                    xx,
                    0.13 * max(n),
                    f"{p}th",
                    color=percentiles_color,
                    horizontalalignment="center",
                    bbox=dict(
                        boxstyle="Round,pad=0.0",
                        edgecolor="white",
                        facecolor="white",
                        alpha=0.5,
                    ),
                )
                for ip, (xx, p) in enumerate(zip(x, percentiles))
            ]

            # Plot comparative map points
            if not comp_QE == []:
                for qe in comp_QE:
                    plt.plot(qe, 0.60 * max(n), "|", c=comp_color, markersize=20)

            # Custom text for paper
            x, y = int(i / som_height), int(i % som_height)
            if additional_text_for_paper:
                if x == 0 and y == 0:
                    additional_text = "A)"
                elif x == 1 and y == 0:
                    additional_text = "B)"
                elif x == 4 and y == 0:
                    additional_text = "C)"
                elif x == 4 and y == 1:
                    additional_text = "D)"
                else:
                    additional_text = ""
                ax = plt.gca()
                plt.text(
                    -0.1,
                    0.9,
                    additional_text,
                    color="r",
                    size=20,
                    transform=ax.transAxes,
                )
            #########################
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            if save:
                plt.savefig(
                    os.path.join(save_dir, f"hist_{x}_{y}_" + save_name + ".pdf"),
                    bbox_inches="tight",
                )
            plt.show()
            plt.close()

            s3 = sum(QE > AQ + 3 * AQ_std)
            s5 = sum(QE > AQ + 5 * AQ_std)
            # print(f"{AQ:.5f}pm{AQ_std:.5f} 3sigma={AQ+3*AQ_std:.5f} {s3} {100*s3/len(QE):.1f}, " \
            #        f"5sigma={AQ+5*AQ_std:.5f} {s5} {100*s5/len(QE):.1f}")

    return AQE_per_node, AQE_std_per_node, QE_per_node


def plot_matrix_of_AQE_histograms(
    som,
    som_map,
    rotated_pixel_size,
    verbose=False,
    cutouts_normed=False,
    comparative_datamap=[],
    percentiles_color="red",
    save=False,
    save_path="",
    absolute=True,
    plot_only_comparative=False,
    version=1,
):
    """Plot matrix with same dimensions as the SOM worth of AQE histograms"""
    bmus = np.argmin(som_map, axis=1)
    if cutouts_normed:
        min_values = np.min(som_map, axis=1)
    else:
        min_values = np.min(som_map, axis=1) / (rotated_pixel_size**2)

    # Get AQE per node (normalized by the number of pixels in the node
    QE_per_node = [[] for _ in range(len(som_map[0]))]
    comparative_QE_per_node = [[] for _ in range(len(som_map[0]))]
    for i, (bmu, min_value) in enumerate(zip(bmus, min_values)):
        QE_per_node[bmu].append(min_value)
    QE_per_node = np.array(QE_per_node)

    AQE_per_node = np.array(list(map(np.mean, QE_per_node)))
    AQE_std_per_node = np.array(list(map(np.std, QE_per_node)))

    max_QE = np.max(np.max(QE_per_node))
    print(f"max qe is {max_QE}")
    """
    ns = []
    for QE in QE_per_node:
        n, bins, _ = plt.hist(QE,bins=20,histtype='step', linewidth=1)
        if np.max(QE) == max_QE:
            best_bins = bins
    for QE in QE_per_node:
        n, bins, _ = plt.hist(QE,bins=bins,histtype='step', linewidth=1)
        if np.max(QE) == max_QE:
            best_bins = bins
        ns.append(n)
    max_n = np.max(np.max(ns))
    print(f'max_n is {max_n}')
    #plt.close()
    
    """

    if not comparative_datamap == []:
        comp_bmus = np.argmin(comparative_datamap, axis=1)
        comp_min_values = np.min(comparative_datamap, axis=1) / (
            rotated_pixel_size**2
        )
        # Do something
        for i, (bmu, min_value) in enumerate(zip(comp_bmus, comp_min_values)):
            comparative_QE_per_node[bmu].append(min_value)
        comparative_QE_per_node = np.array(comparative_QE_per_node)

        percentiles_color = "black"
        comp_color = "orange"

    # create subplots
    f, axes = plt.subplots(som.som_width, som.som_height, figsize=(15, 15))
    # ax = plt.Axes(f, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # f.add_axes(ax)
    flax = axes.T.flatten()
    for i, (ax, QE, comp_QE, AQ, AQ_std) in enumerate(
        zip(flax, QE_per_node, comparative_QE_per_node, AQE_per_node, AQE_std_per_node)
    ):
        if QE == []:
            # print(i, 'No best matching sources in the original data set')
            # if not comp_QE == []:
            # print("Comparison data set:", comp_QE)
            continue
        elif comp_QE == [] and plot_only_comparative:
            # print(i, 'No best matching sources in the comparative data set')
            continue
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(False)
        # a.set_aspect('equal')
        if absolute:
            n, bins, _ = ax.hist(
                QE, bins="auto", histtype="step", density=True, linewidth=2, color="r"
            )
        else:
            n, bins, _ = ax.hist(
                QE, bins="auto", histtype="step", density=True, linewidth=2, color="r"
            )

        # Plot median and percentiles
        percentiles = [5, 25, 50, 75, 95]
        x = np.percentile(QE, percentiles)
        median = np.percentile(QE, 50)

        """
        ax.plot(x, 0.30*max(n)*np.ones(len(x)),'o-', c=percentiles_color)
        ax.plot(median, 0.30*max(n),'o', c=percentiles_color, markersize=10)
        [ax.text(xx, 0.37*max(n), f"{p}th", color=percentiles_color) for xx, p 
                in zip(x, percentiles)]
        """

        # Plot comparative map points
        if not comp_QE == []:
            for qe in comp_QE:
                if absolute:
                    ax.plot(qe, 0.75 * max(n), marker="|", c="yellow", markersize=20)
                else:
                    ax.plot(qe, 0.75 * max(n), marker="|", c="yellow", markersize=20)

        append = "_relative.png"
        if absolute:
            append = "_absolute.png"
            ax.set_xlim(-0.01 * max_QE, max_QE * 1.02)
        # ax.set_ylim(0,max_n)

        # plt.xlabel('Summed Euclidean distance between cutout and its best matching prototype / number of pixels')
        # plt.ylabel('Number of cutouts')
        # plt.show()
        # s3 = sum(QE > AQ+3*AQ_std)
        # s5 = sum(QE > AQ+5*AQ_std)
        # print(f"{AQ:.5f}pm{AQ_std:.5f} 3sigma={AQ+3*AQ_std:.5f} {s3} {100*s3/len(QE):.1f}, " \
        #        f"5sigma={AQ+5*AQ_std:.5f} {s5} {100*s5/len(QE):.1f}")

    f.subplots_adjust(wspace=0.0001, hspace=0.0001, left=0, right=1, top=1, bottom=0)
    if save:
        # plt.savefig(save_path+append,alpha=0.5,transparent=True, bbox_inches='tight', pad_inches=0)
        if absolute:
            plt.savefig(save_path + append, transparent=True)
        else:
            plt.savefig(save_path + append, transparent=True)
        # plt.savefig(save_path, pad_inches=0, transparent=True)
    else:
        plt.show()


def return_cutouts_sorted_per_prototype(
    p, som, cutouts, data_directory, som_map, QE_per_node, version=1
):
    """From best to least matching for each best matching unit"""
    # Go from datamap to list of all bmus
    bmus = np.argmin(som_map, axis=1)
    bmu_distances = np.min(som_map, axis=1)
    # Get all indexes of prototypes that have atleast one match
    QE_non_empty = [i for i, QE in enumerate(QE_per_node) if not QE == []]

    # Load the cutouts
    # cutouts = np.load(os.path.join(data_directory, som.training_dataset_name + '.npy'))
    # print(os.path.join(data_directory, som.training_dataset_name + '.npy'))

    # For all prototypes...
    for p_i in QE_non_empty:
        if not p_i == p:
            continue
        # get indices of cutouts that best match to this prototype
        ind = find_index(bmus, p_i)
        # sort these indices from good to worst matching (all are best matching)
        sorted_index = ind[np.argsort(bmu_distances[ind])]
        sorted_values = np.sort(bmu_distances[ind])

        # PLotting
        print(
            f"Prototype {p_i},"
            f"({int(p_i/som.som_height)},{int(p_i % som.som_width)})"
        )
        # Get prototype
        prot = np.array(
            som.data_som[int(p_i % som.som_width), int(p_i / som.som_height), 0]
        )
        # prot = np.array(som.data_som[int(i / som.som_height),int( i % som.som_width),  0])

        r = som.rotated_size
        f = som.fullsize
        t_old = -1
        prot_size = int(np.sqrt(len(prot)))
        crop = prot.reshape(prot_size, prot_size)
        if prot_size > r:
            crop = crop[
                int((prot_size - r) / 2) : int((prot_size + r) / 2),
                int((prot_size - r) / 2) : int((prot_size + r) / 2),
            ]

        plt.imshow(crop, cmap="viridis", origin="upper", interpolation="nearest")
        plt.grid(False)
        plt.show()

        for j, (t, v) in enumerate(zip(sorted_index, sorted_values)):

            print("index:", t)
            if version == 1:
                img2 = cutouts[t].reshape(f, f)
                crop2 = img2[
                    int((f - r) / 2) : int((f + r) / 2),
                    int((f - r) / 2) : int((f + r) / 2),
                ]
            else:
                crop2 = cutouts[t].reshape(r, r)
            plt.imshow(crop2, cmap="viridis", origin="upper", interpolation="nearest")
            plt.grid(False)
            plt.show()

        """
        plt.figure(figsize=(18,5))
        prot_size = int(np.sqrt(len(prot)))
        crop = prot.reshape(prot_size, prot_size)
        if prot_size > r:
            crop = crop[int((prot_size-r)/2):int((prot_size+r)/2),int((prot_size-r)/2):int((prot_size+r)/2)]
        minmax_interval = vis.MinMaxInterval()
        plt.text(0.1*r, 10, f"Prototype ", color='w')
        plt.text(0.3*r, r-20, f"SEDs:", color='w')

        # Plot the cutouts
        for j, (t, v) in enumerate(zip(sorted_index, sorted_values)):
            if t == t_old:
                crop2 = np.zeros((r,r))
                plt.text(r/2 +(j+1)*r, r/2, f"\"...\"", color='w')
            else:
                img2 = cutouts[t].reshape(f,f)
                crop2 = img2[int((f-r)/2):int((f+r)/2),int((f-r)/2):int((f+r)/2)]
            crop = np.hstack((crop,crop2))  
            t_old = t
            # Print caption
            plt.text(0.1*r+(j+1)*r, r-20, f"{v:.2g} {v/normed_prototype_sum:.2g}", color='w')
        plt.imshow(crop, cmap='viridis', origin='upper', interpolation="nearest")
        plt.grid(False)
        #plt.tick_params(
        #    axis='x',          # changes apply to the x-axis
        plt.show()
"""


def return_cutouts_per_percentile_per_prototype(
    som,
    data_directory,
    som_map,
    QE_per_node,
    percentiles=[5, 15, 25, 35, 45, 50, 55, 65, 75, 85, 95],
    cutouts_normed=False,
    threshold=0,
    version=1,
    debug=False,
    plot_only_subset_of_prototypes=False,
    subset_of_prototypes=None,
    save=False,
    figures_dir=None,
    pixval=None,
    pixval_squared=None,
    good_percentile=None,
    bad_percentile=None,
    outlier_thresholds=[0.1],
):
    """returns cutouts close to x times the AQE standard deviation where the values of x
    are taken from std_list."""
    bmus = np.argmin(som_map, axis=1)

    # Prepare the values
    QE_non_empty = [i for i, QE in enumerate(QE_per_node) if not QE == []]
    percentiles_lists = [
        np.percentile(QE, percentiles) for QE in QE_per_node[QE_non_empty]
    ]

    # For each prototype
    targets_per_prototype = [[] for _ in range(len(percentiles_lists))]
    values_per_prototype = [[] for _ in range(len(percentiles_lists))]

    for i_p, percentiles_list, target, values in zip(
        QE_non_empty, percentiles_lists, targets_per_prototype, values_per_prototype
    ):
        if i_p == 0 and debug:
            print("perc list", percentiles_list)
            # print("QE for that node", QE_per_node[i_p])
            plt.hist(QE_per_node[0])
            plt.show()
        # Find the indices of cutouts that best match the current prototype
        ind = find_index(bmus, i_p)
        # Find the values for these indexes
        # Note: if cutouts are normalized in preprocessing, dividing by the number of pixels
        # in the cutout is not needed
        if not cutouts_normed:
            min_values = np.min(som_map[ind], axis=1)
        else:
            min_values = np.min(som_map[ind], axis=1) / (som.rotated_size**2)
        for percent in percentiles_list:
            match_idx = find_nearest(min_values, percent)
            target.append(ind[match_idx])
            values.append(min_values[match_idx])

    # Load the cutouts
    cutouts = np.load(os.path.join(data_directory, som.training_dataset_name + ".npy"))
    # print(os.path.join(data_directory, som.training_dataset_name + '.npy'))

    # Plot the cutouts
    good = []
    bad = []
    for i, targets, values in zip(
        QE_non_empty, targets_per_prototype, values_per_prototype
    ):
        x, y, z = int(i / som.som_width), int(i % som.som_width), 0
        if plot_only_subset_of_prototypes:
            if not (x, y, z) in subset_of_prototypes:
                continue
        threshold_reached = False
        threshold_reached2 = False
        threshold2 = 0.003
        if debug:
            print(f"Prototype {i}, ({int(i/som.som_width)},{int(i % som.som_width)})")
        # Get prototype
        # prot = np.array(som.data_som[int(i % som.som_width),int( i / som.som_height),  0])
        prot = np.array(som.data_som[x, y, z])
        prototype_sum = np.sum(prot)

        r = som.rotated_size
        f = som.fullsize
        t_old = -1
        plt.figure(figsize=(18, 6))
        prot_size = int(np.sqrt(len(prot)))
        crop = prot.reshape(prot_size, prot_size)
        if prot_size > r:
            crop = crop[
                int((prot_size - r) / 2) : int((prot_size + r) / 2),
                int((prot_size - r) / 2) : int((prot_size + r) / 2),
            ]
        crop_segments = []
        if not pixval is None:
            print(f"proto min {np.min(crop)} {np.max(crop)}")
            # crop_segment = np.where(crop>(outlier_threshold*np.max(crop)),1,0)

            for outlier_threshold in outlier_thresholds:
                crop_segments.append(np.where(crop > (outlier_threshold), 1, 0))
        minmax_interval = vis.MinMaxInterval()
        plt.text(0.05 * r, 35, f"Neuron ({x},{y}) \n{prototype_sum:.3g}", color="w")
        plt.vlines(r, 0, r * 3, color="k")
        # plt.text(0.3*r, r-20, f"normed AQEs:", color='w')
        for j, (p, t, v) in enumerate(zip(percentiles, targets, values)):
            if not pixval is None:
                if p == good_percentile:
                    good.append(v)
                if p == bad_percentile:
                    bad.append(v)
            if t == t_old:
                crop2 = np.zeros((r, r))
                # plt.text(r/2 +(j+1)*r, r/2, f"\"...\"", color='w')
            else:
                if version == 1:
                    img2 = cutouts[t].reshape(f, f)
                    crop2 = img2[
                        int((f - r) / 2) : int((f + r) / 2),
                        int((f - r) / 2) : int((f + r) / 2),
                    ]
                else:
                    crop2 = cutouts[t].reshape(r, r)
            crop = np.hstack((crop, crop2))
            t_old = t
            # Print caption
            # outlier score
            # plt.text(0.1*r+(j+1)*r, r-30, f"{1000*v:.2g}", color='w')
            if not pixval is None:
                for ii, outlier_threshold in enumerate(outlier_thresholds):
                    cutout_area = np.sum(np.where(crop2 > outlier_threshold, 1, 0))
                    crop2_segment = np.where(crop2 > outlier_threshold, 1, 0)
                    crop_segments[ii] = np.hstack((crop_segments[ii], crop2_segment))

                # outlier flux
                # plt.text(0.1*r+(j+1)*r, r-20, f"{1000*v:.2g} {1000*v*(0.1*pixval+1):.2g} {1000*v*(0.1*pixval_squared+1):.2g}", color='w')
            """
            if threshold < v/prototype_sum and not threshold_reached:
                plt.vlines((1+j)*r, 0, r, color='red')
                threshold_reached = True
            if threshold2 < v and not threshold_reached2:
                plt.vlines((1+j)*r, 0, r, color='green',linestyle='dashed')
                threshold_reached2 = True
            """
            if p == good_percentile:
                plt.vlines((1 + j) * r, 0, r, color="green")
                plt.vlines((2 + j) * r - 5, 0, r, color="green")
            if p == bad_percentile:
                plt.vlines((1 + j) * r, 0, r, color="red", linestyle="dashed")
                plt.vlines((2 + j) * r - 5, 0, r, color="red", linestyle="dashed")
            # plt.text(0.1*r+(j+1)*r, r-50, f"{v/prototype_sum:.2g}", color='w')
            plt.text(0.1 * r + (j + 1) * r, 15, f"{p}th perc.", color="w")
        if not pixval is None:
            crop = np.vstack((crop, *crop_segments))
        plt.imshow(crop, cmap="viridis", origin="upper", interpolation="nearest")
        plt.grid(False)
        plt.yticks([])
        plt.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off
        plt.tick_params(
            axis="y",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(figures_dir, f"sort_along_prot_{x}_{y}_{z}"))
        plt.show()
    plt.close("all")
    if not pixval is None:
        return good, bad
    else:
        return targets_per_prototype


def check_percentage_below_threshold(som, som_map, threshold):
    min_values = np.min(som_map, axis=1) / (som.rotated_size**2)
    bmu_index = np.argmin(som_map, axis=1)

    prots = [
        np.array(som.data_som[int(i / som.som_height), int(i % som.som_width), 0])
        for i in bmu_index
    ]
    normed_prototype_sums = [np.sum(prot) / (som.rotated_size**2) for prot in prots]
    below_threshold = [
        threshold > min_value / normed_prototype_sum
        for min_value, normed_prototype_sum in zip(min_values, normed_prototype_sums)
    ]
    print(f"Total number of mapped cutouts is {len(som_map)}.")
    print(
        f"Total number below threshold is {sum(below_threshold)} ({sum(below_threshold)/len(som_map)*100:.3g}%)"
    )


def return_cutouts_per_std_per_prototype(
    som,
    som_map,
    AQE_per_node,
    AQE_std_per_node,
    cutouts,
    standard_deviations=[-2, -1, -0.5, 0, 0.5, 1, 2, 3, 4],
):
    """returns cutouts close to x times the AQE standard deviation where the values of x
    are taken from std_list."""
    bmus = np.argmin(som_map, axis=1)

    # Prepare the values
    std_lists = [
        aq + np.array(standard_deviations) * std
        for aq, std in zip(AQE_per_node, AQE_std_per_node)
    ]

    # For each prototype
    targets_per_prototype = [[] for _ in range(som.som_width * som.som_height)]
    for i_p, (std_list, target) in enumerate(zip(std_lists, targets_per_prototype)):
        # Find the indices of cutouts that best match the current prototype
        ind = find_index(bmus, i_p)
        # Find the values for these indexes
        min_values = np.min(som_map[ind], axis=1) / (som.rotated_size**2)
        for std in std_list:
            match_idx = find_nearest(min_values, std)
            target.append(ind[match_idx])

    # Plot the cutouts
    for i, targets in enumerate(targets_per_prototype):
        # Get prototype
        prot = np.array(
            som.data_som[int(i % som.som_width), int(i / som.som_height), 0]
        )
        prot = np.array(
            som.data_som[int(i / som.som_height), int(i % som.som_width), 0]
        )

        r = som.rotated_size
        f = som.fullsize
        crop = prot.reshape(r, r)
        # img = cutouts[targets[0]].reshape(som.fullsize,som.fullsize)
        # crop = img[int((f-r)/2):int((f+r)/2),int((f-r)/2):int((f+r)/2)]
        t_old = -1
        plt.figure(figsize=(18, 5))
        plt.text(0.1 * r, 10, "Prototype", color="w")
        for j, (standard_deviation, t) in enumerate(zip(standard_deviations, targets)):
            if t == t_old:
                crop2 = np.zeros((r, r))
                plt.text(r / 2 + (j + 1) * r, r / 2, f'"..."', color="w")
            else:
                img2 = cutouts[t].reshape(f, f)
                crop2 = img2[
                    int((f - r) / 2) : int((f + r) / 2),
                    int((f - r) / 2) : int((f + r) / 2),
                ]
            crop = np.hstack((crop, crop2))
            t_old = t
            # Print caption
            plt.text(
                0.1 * r + (j + 1) * r, 10, f"{standard_deviation} st. dev.", color="w"
            )
        plt.imshow(crop, interpolation="nearest", cmap="viridis", origin="lower")
        plt.grid(False)
        plt.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off
        plt.tight_layout()
        plt.show()
    plt.close("all")
    return targets_per_prototype


def calculate_AQEs(rotated_size, maps, verbose=True, version=1):
    """Returns Average Quantization Errors (Note: plural!),
    which is defined as the average Euclidean distance from each input of the
    training set to each corresponding best matching node/prototype/neuron."""

    if version == 1:
        aq = [np.sqrt(np.min(tm, axis=1)) for tm in maps]
    else:
        aq = [np.min(tm, axis=1) for tm in maps]
    """
    aq = [np.min(tm, axis=1) for tm in maps]
    """

    AQEs = [np.mean(a) for a in aq]
    AQEs_std = [np.std(a) for a in aq]
    AQEs_quart = [np.percentile(a, 25) for a in aq]
    AQEs_fifth = [np.percentile(a, 50) for a in aq]
    AQEs_sev = [np.percentile(a, 75) for a in aq]
    AQEs_norm = [np.mean(a / (rotated_size**2)) for a in aq]
    AQEs_norm_std = [np.std(a / (rotated_size**2)) for a in aq]
    """
    AQEs = [np.mean(np.sqrt(np.min(tm, axis=1))) for tm in maps]
    AQEs_std = [np.std(np.min(tm, axis=1)) for tm in maps]
    AQEs_quart = [np.percentile(np.min(tm, axis=1),25) for tm in maps]
    AQEs_fifth = [np.percentile(np.min(tm, axis=1),50) for tm in maps]
    AQEs_sev = [np.percentile(np.min(tm, axis=1),75) for tm in maps]
    AQEs_norm = [np.mean(np.min(tm, axis=1)/(rotated_size**2)) for tm in maps]
    AQEs_norm_std = [np.std(np.min(tm, axis=1)/(rotated_size**2)) for tm in maps]
    """

    # Check if aqe-deviations are normally distributed
    """
    plt.figure(figsize=(10,4))
    bins = plt.hist([np.min(tm,axis=1) for tm in maps],bins=100,density=False)
    height = max(bins[0])
    plt.plot([AQEs[0]-AQEs_std[0],AQEs[0],AQEs[0]+AQEs_std[0]], [height*0.2 for _ in
        range(3)],'r',markersize=10, marker='o')
    plt.plot([AQEs_quart[0],AQEs_fifth[0],AQEs_sev[0]], [height*0.5 for _ in
        range(3)],color='orange',marker='o',markersize=10)
    #plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0,20)
    plt.show()
    """
    return AQEs, AQEs_std, AQEs_norm, AQEs_norm_std, AQEs_quart, AQEs_fifth, AQEs_sev


def calculate_TEs(
    maps, som_width, som_height, periodic_boundary_condition, layout, verbose=False
):
    TEs_list = []
    """Return the Topological Errors [%] (Note: plural!),
    which is defined as the chance that the second best matching unit is not(!) the
    direct neighbour of the best matching unit."""

    for tm_counter, tm in enumerate(maps):
        TEs = 0
        for distances in tm:
            BMU_value, SBMU_value = heapq.nsmallest(2, distances)
            if any(np.isnan([BMU_value, SBMU_value])):
                print("Something went wrong with calculating the BMU and SBMU.")
                print(np.shape(distances), type(distances))
                print(distances)
                print("Counter:", tm_counter)
                sdfsdfs
            else:
                bmu = distances.tolist().index(BMU_value)
                sbmu = distances.tolist().index(SBMU_value)

            if layout == "quadratic" or layout == "cartesian":
                # Get BMU coordinates
                x, y = int(bmu % som_width), int(bmu / som_height)
                x2, y2 = int(sbmu % som_width), int(sbmu / som_height)

                # Calculate distance if pbc=True
                if periodic_boundary_condition:
                    if abs(x - x2) > som_width / 2.0:
                        if x < x2:
                            x += som_width
                        else:
                            x2 += som_width
                    if abs(y - y2) > som_height / 2.0:
                        if y < y2:
                            y += som_height
                        else:
                            y2 += som_height

                TE_difference = np.linalg.norm(np.array([x, y]) - np.array([x2, y2]))
                if verbose:
                    print(TE_difference)

                # Get quantization error
                if TE_difference > 1.9:
                    TEs += 1
            else:
                hex_size, hex_map = get_hex_size(som_width)
                neighbors = get_hex_neighbors(bmu, hex_map)
                if not (sbmu in neighbors):
                    TEs += 1

        TEs /= len(tm)
        TEs_list.append(100.0 * TEs)
    return TEs_list


def get_hex_size(hex_length):
    """Returns number of nodes in the hexmap (hex_size) with dimensions hex_length
    x hex_length.
    Also returns the indexes of the nodes per row of the hexmap (hex_map)"""
    hex_size = 0
    last = 0
    hex_map = []
    for t in range(int(hex_length / 2) + 1, hex_length + 1):
        hex_map.append(list(range(last, t + last)))
        hex_size += t
        last += t
    for t in range(hex_length - 1, int(hex_length / 2), -1):
        hex_map.append(list(range(last, t + last)))
        hex_size += t
        last += t
    return hex_size, hex_map


def get_hex_neighbors(node_id, hex_map):
    """Return the indexes of the direct neighbours to a node (node_id)
    of a given hexagonal map (hex_map)."""
    node_x, node_y = [
        (a.index(node_id), a_i) for a_i, a in enumerate(hex_map) if node_id in a
    ][0]
    neighbor_ids = []
    if node_y < int(len(hex_map) / 2):
        neighbor_coordinates = [
            (node_x - 1, node_y - 1),
            (node_x, node_y - 1),
            (node_x - 1, node_y),
            (node_x + 1, node_y),
            (node_x, node_y + 1),
            (node_x + 1, node_y + 1),
        ]
    elif node_y == int(len(hex_map) / 2):
        neighbor_coordinates = [
            (node_x - 1, node_y - 1),
            (node_x, node_y - 1),
            (node_x - 1, node_y),
            (node_x + 1, node_y),
            (node_x - 1, node_y + 1),
            (node_x, node_y + 1),
        ]
    else:
        neighbor_coordinates = [
            (node_x, node_y - 1),
            (node_x + 1, node_y - 1),
            (node_x - 1, node_y),
            (node_x + 1, node_y),
            (node_x - 1, node_y + 1),
            (node_x, node_y + 1),
        ]
    for n_x, n_y in neighbor_coordinates:
        try:
            if n_x > -1 and n_y > -1:
                neighbor_ids.append(hex_map[n_y][n_x])
        except:
            pass
    return neighbor_ids


def load_som_mapping(
    mapping_path, som, version=1, compress=False, replace_nans=False, verbose=True
):
    """Load distances of cut-outs to a trained SOM, also returning
    the number of cut-outs mapped to the SOM, and the width, height and depth
    of this SOM. Requires the file path to the mapping binary file and the
    layout of the trained SOM (either quadratic or hexagonal)"""
    # Create list of indexes to retrieve coordinates of the cut-outs
    cut_out_index = []

    # Unpack SOM mapping
    if not os.path.exists(mapping_path):
        raise FileNotFoundError
    assert version == 1 or version == 2

    with open(mapping_path, "rb") as inputStream:
        if version == 1:
            numberOfImages = struct.unpack("i", inputStream.read(4))[0]
            som_width = struct.unpack("i", inputStream.read(4))[0]
            som_height = struct.unpack("i", inputStream.read(4))[0]
            som_depth = struct.unpack("i", inputStream.read(4))[0]
            failed = 0
            assert som.som_width == som_width, f"{som.som_width}, {som_width}"
            assert som.som_height == som_height, f"{som.som_height}, {som_height}"
            assert som.som_depth == som_depth, f"{som.som_depth}, {som_depth}"

            if som.layout == "hexagonal":
                map_size, _ = get_hex_size(som_width)
            else:
                map_size = som_width * som_height * som_depth

            data_map = np.ones((numberOfImages, map_size))
            for i in range(numberOfImages):
                for t in range(map_size):
                    try:
                        data_map[i, t] = struct.unpack_from("f", inputStream.read(4))[0]
                    except:
                        failed += 1
                        print(
                            "Loading SOM mapping failed for:",
                            mapping_path,
                            f"For image {i} and som_node_index {t}.",
                        )
                    if failed == 0:
                        cut_out_index.append(i)  # add index
            data_map = data_map[: len(cut_out_index)]
            if failed > 0:
                print("Failed:", int(1.0 * failed / (map_size)))
            return data_map, numberOfImages, som_width, som_height, som_depth
        elif version == 2:
            # <file format version> 2 <data-type> <number of entries> <som layout> <data>
            (
                version,
                file_type,
                data_type,
                numberOfImages,
                som_layout,
                som_dimensionality,
            ) = struct.unpack("i" * 6, inputStream.read(4 * 6))
            som_dimensions = struct.unpack(
                "i" * som_dimensionality, inputStream.read(4 * som_dimensionality)
            )
            if verbose:

                print("version:", version)
                print("file_type:", file_type)
                print("data_type:", data_type)
                print("numberOfImages:", numberOfImages)
                print("som_layout:", som_layout)
                print("som dimensionality:", som_dimensionality)
                print("som dimensions:", som_dimensions)

            som_width = som_dimensions[0]
            som_height = som_dimensions[1] if som_dimensionality > 1 else 1
            som_depth = som_dimensions[2] if som_dimensionality > 2 else 1

            failed = 0
            maps = []
            if verbose:
                print(
                    "unpacked som width height depth", som_width, som_height, som_depth
                )
            assert file_type == 2  # 2 equals mapping
            assert som.som_width == som_width
            assert som.som_height == som_height
            assert som.som_depth == som_depth

            if som.layout == "hexagonal":
                map_size, _ = get_hex_size(som_width)
            else:
                map_size = som_width * som_height * som_depth

            data_map = np.ones((numberOfImages, map_size))
            for i in range(numberOfImages):
                for t in range(map_size):
                    try:
                        data_map[i, t] = struct.unpack_from("f", inputStream.read(4))[0]
                        if t == 0:
                            cut_out_index.append(i)  # add index
                    except:
                        failed += 1
            data_map = data_map[: len(cut_out_index)]
            if failed > 0:
                print("Failed:", int(1.0 * failed / (map_size)))
            if replace_nans:
                data_map = np.nan_to_num(data_map, nan=1e9)

            if compress:
                assert (
                    som.som_width == som.som_height
                ), "compression only implemented for square SOMs"
                assert (
                    som.layout != "hexagonal"
                ), "Compression only implemented for cartesian SOMs"
                print(
                    "WARNING: Compress taking place at load_mapping stage. Don't do another compress later on!"
                )
                i_compress, new_dimension = return_sparce_indices(som.som_width)
                data_map = data_map[:, i_compress]
                som_width = new_dimension
                som_height = new_dimension

            return (
                copy.deepcopy(data_map),
                numberOfImages,
                som_width,
                som_height,
                som_depth,
            )
            """
            start = inputStream.tell()      
            if os.path.getsize(mapping_path) < numberOfImages * som_width * som_height * som_depth * 4 + start:
                shape = "hex"
            else:
                shape = "box"

            #Unpacks data
            hexSize=int(1.0 + 6.0 * (( (som_width-1)/2.0 + 1.0) * (som_height-1)/ 4.0))
            try:
                while True:
                    if shape == "box":
                        data = np.ones(som_width * som_height * som_depth)
                        for i in range(som_width * som_height *som_depth):
                            data[i] = struct.unpack_from("f", inputStream.read(4))[0]
                        maps.append(data)
                    else:
                        
                        data = np.ones(hexSize)
                        for i in range(hexSize):
                            data[i] = struct.unpack_from("f", inputStream.read(4))[0]
                        maps.append(data)

            except:
                pass
        maps = np.array(maps)


        if verbose:
            print('''Loaded distances of {} cut-outs to a SOM with a width, height 
                and depth equal to {},{},{}.'''.format(numberOfImages,
                som_width, som_height, som_depth))
        return maps, numberOfImages, som_width, som_height, som_depth
        """


def load_som_mapping_old(mapping_path, som, verbose=True):
    """Load distances of cut-outs to a trained SOM, also returning
    the number of cut-outs mapped to the SOM, and the width, height and depth
    of this SOM. Requires the file path to the mapping binary file and the
    layout of the trained SOM (either quadratic or hexagonal)"""
    # Create list of indexes to retrieve coordinates of the cut-outs
    cut_out_index = []

    # Unpack SOM mapping
    if not os.path.exists(mapping_path):
        print("This file does not exist:", mapping_path)
    with open(mapping_path, "rb") as inputStream:
        numberOfImages = struct.unpack("i", inputStream.read(4))[0]
        som_width = struct.unpack("i", inputStream.read(4))[0]
        som_height = struct.unpack("i", inputStream.read(4))[0]
        som_depth = struct.unpack("i", inputStream.read(4))[0]
        failed = 0
        assert som.som_width == som_width
        assert som.som_height == som_height
        assert som.som_depth == som_depth

        if som.layout == "hexagonal":
            map_size, _ = get_hex_size(som_width)
        else:
            map_size = som_width * som_height * som_depth

        data_map = np.ones((numberOfImages, map_size))
        for i in range(numberOfImages):
            for t in range(map_size):
                try:
                    data_map[i, t] = struct.unpack_from("f", inputStream.read(4))[0]
                    if t == 0:
                        cut_out_index.append(i)  # add index
                except:
                    failed += 1
        data_map = data_map[: len(cut_out_index)]
        if failed > 0:
            print("Failed:", int(1.0 * failed / (map_size)))
    if verbose:
        print(
            """Loaded distances of {} cut-outs to a SOM with a width, height 
            and depth equal to {},{},{}.""".format(
                numberOfImages, som_width, som_height, som_depth
            )
        )
    return data_map, numberOfImages, som_width, som_height, som_depth


def unpack_trained_som(
    trained_path, layout, verbose=True, version=1, replace_nans=False
):
    """Unpacks a trained SOM, returns the SOM with hexagonal layout
    in a flattened format. Requires the file path to the trained SOM and
    its layout (either quadratic or hexagonal)"""
    assert version == 1 or 2
    with open(trained_path, "rb") as inputStream:
        if version == 1:
            failures = 0
            # File structure: (som_width, som_height, som_depth, number_of_channels,
            # neuron_width, neuron_height) float
            number_of_channels = struct.unpack("i", inputStream.read(4))[0]
            som_width = struct.unpack("i", inputStream.read(4))[0]
            som_height = struct.unpack("i", inputStream.read(4))[0]
            som_depth = struct.unpack("i", inputStream.read(4))[0]
            neuron_width = struct.unpack("i", inputStream.read(4))[0]
            neuron_height = struct.unpack("i", inputStream.read(4))[0]

            if layout == "quadratic":
                data_som = np.ones(
                    (
                        som_width,
                        som_height,
                        som_depth,
                        number_of_channels * neuron_width * neuron_height,
                    )
                )
                for i in range(som_width):
                    for ii in range(som_height):
                        for iii in range(som_depth):
                            for iv in range(
                                number_of_channels * neuron_width * neuron_height
                            ):
                                try:
                                    data_som[i, ii, iii, iv] = struct.unpack_from(
                                        "f", inputStream.read(4)
                                    )[0]
                                except:
                                    failures += 1.0
            else:
                som_size, _ = get_hex_size(som_width)
                data_som = np.ones(
                    (som_size, number_of_channels * neuron_width * neuron_height)
                )
                for i in range(som_size):
                    for ii in range(number_of_channels * neuron_width * neuron_height):
                        try:
                            data_som[i, ii] = struct.unpack_from(
                                "f", inputStream.read(4)
                            )[0]
                        except:
                            failures += 1.0
            if failures > 0:
                print(
                    "Failures:",
                    int(failures / (number_of_channels * neuron_width * neuron_height)),
                )
        elif version == 2:
            failures = 0
            # File structure: <file format version> 1 <data-type> <som layout> <neuron layout>
            # <data>
            (
                file_format_version,
                file_type,
                data_type,
                som_layout,
                som_dimensionality,
            ) = struct.unpack("i" * 5, inputStream.read(4 * 5))
            assert file_format_version == 2
            assert file_type == 1
            assert (
                data_type == 0
            ), "Other values than 0 are not supported for PINK versions <=2.5"
            assert som_dimensionality == 2 or som_dimensionality == 3
            som_dimensions = struct.unpack(
                "i" * som_dimensionality, inputStream.read(4 * som_dimensionality)
            )
            neuron_layout, neuron_dimensionality = struct.unpack(
                "i" * 2, inputStream.read(4 * 2)
            )
            neuron_dimensions = struct.unpack(
                "i" * neuron_dimensionality, inputStream.read(4 * neuron_dimensionality)
            )
            if verbose:
                print("som_layout:", som_layout)
                print("som dimensionality:", som_dimensionality)
                print("som dimensions:", som_dimensions)
                print("neuron_layout:", neuron_layout)
                print("neuron dimensionality:", neuron_dimensionality)
                print("neuron dimensions:", neuron_dimensions)

            som_width = som_dimensions[0]
            som_height = som_dimensions[1] if som_dimensionality > 1 else 1
            som_depth = som_dimensions[2] if som_dimensionality > 2 else 1
            number_of_channels = (
                neuron_dimensions[0] if neuron_dimensionality > 2 else 1
            )
            if neuron_dimensionality == 1:
                neuron_width = neuron_dimensions[0]
                neuron_height = 1
            if neuron_dimensionality == 2:
                neuron_width = neuron_dimensions[0]
                neuron_height = neuron_dimensions[1]
            if neuron_dimensionality == 3:
                neuron_width = neuron_dimensions[1]
                neuron_height = neuron_dimensions[2]
            if verbose:
                print(
                    f"SOM wxhxd {som_width}x{som_height}x{som_depth},",
                    f" neuron n_channels x width x height",
                    f" {number_of_channels}x{neuron_width}x{neuron_height}",
                )
            assert neuron_width == neuron_height, "Not implemented (at all)"
            """
            file_format_version = struct.unpack("i", inputStream.read(4))[0]
            assert file_format_version == version
            file_type = struct.unpack("i", inputStream.read(4))[0]
            assert file_type == 1
            data_type = struct.unpack("i", inputStream.read(4))[0]
            assert data_type == 0 
            som_layout = struct.unpack("i", inputStream.read(4))[0]
            som_dimensionality = struct.unpack("i", inputStream.read(4))[0]
            assert som_dimensionality == 2 or som_dimensionality == 3
            if som_dimensionality == 2:
                som_width = struct.unpack("i", inputStream.read(4))[0]
                som_height = struct.unpack("i", inputStream.read(4))[0]
                som_depth = 1
            if som_dimensionality == 3:
                som_depth = struct.unpack("i", inputStream.read(4))[0]
                som_width = struct.unpack("i", inputStream.read(4))[0]
                som_height = struct.unpack("i", inputStream.read(4))[0]
                raise NotImplementedError
            neuron_layout = struct.unpack("i", inputStream.read(4))[0]
            neuron_dimensionality = struct.unpack("i", inputStream.read(4))[0]
            assert neuron_dimensionality == 2 or neuron_dimensionality == 3
            if neuron_dimensionality == 2:
                neuron_width = struct.unpack("i", inputStream.read(4))[0]
                neuron_height = struct.unpack("i", inputStream.read(4))[0]
                number_of_channels = 1
            if neuron_dimensionality == 3:
                number_of_channels = struct.unpack("i", inputStream.read(4))[0]
                neuron_width = struct.unpack("i", inputStream.read(4))[0]
                neuron_height = struct.unpack("i", inputStream.read(4))[0]
            """

            # TEst
            if verbose:
                print("layout", layout)
            if layout == "quadratic" or layout == "cartesian":
                data_som = np.ones(
                    (
                        som_width,
                        som_height,
                        som_depth,
                        number_of_channels * neuron_width * neuron_height,
                    )
                )
                for i in range(som_width):
                    for ii in range(som_height):
                        for iii in range(som_depth):
                            for iv in range(
                                number_of_channels * neuron_width * neuron_height
                            ):
                                try:
                                    data_som[i, ii, iii, iv] = struct.unpack_from(
                                        "f", inputStream.read(4)
                                    )[0]
                                except:
                                    failures += 1.0
                print(f"failures: {failures}")
            else:
                raise NotImplementedError(
                    "Still need to implement the unpacking of hex shaped v2" " SOMs"
                )

            # Unpacks data
            nee = """
            data_som = []
            try:
                while True:
                    data = np.ones(neuron_width * neuron_height * number_of_channels)
                    for i in range(neuron_width * neuron_height * number_of_channels):
                        data[i] = struct.unpack_from("f", inputStream.read(4))[0]
                    data_som.append(data)
            except:
                pass
            data_som = np.array(data_som)
    
            print (str(len(data_som)) + " neurons loaded")
            """
        if replace_nans:
            print(
                "Warning: Replacing NaNs in trained SOM. PINK messed up during training..."
            )
            data_som = np.nan_to_num(data_som, copy=True, nan=0)
        return (
            data_som,
            som_width,
            som_height,
            som_depth,
            neuron_width,
            neuron_height,
            number_of_channels,
        )


def get_quadratic_neighbor_indexes(i, j, som_width, som_height, pbc):
    """Return the indexes of a rectangular or quadratic SOM.
    Requires the index of the SOM (i,j) the SOM width and height
    and finally whether periodic boundary conditions are enabled or disabled
    (pbc=True or pbc=False)."""
    n_i = [
        [i - 1, j - 1],
        [i, j - 1],
        [i + 1, j - 1],
        [i - 1, j],
        [i + 1, j],
        [i - 1, j + 1],
        [i, j + 1],
        [i + 1, j + 1],
    ]
    if pbc:
        return [[n_x % som_width, n_y % som_height] for n_x, n_y in n_i]
    else:
        return [
            [n_x, n_y]
            for n_x, n_y in n_i
            if (n_x > -1 and n_x < som_width and n_y > -1 and n_y < som_height)
        ]


def populate_hex_map(filler, som_width, som_height):
    hex_size, hex_map = get_hex_size(som_width)
    filler_map = np.empty([som_width, som_height], dtype=type(filler[0]))
    filler_mask = np.ones([som_width, som_height], dtype=np.bool)

    mapY = 0
    mapX = abs((som_height - 1) / 2 - mapY)
    mapX = mapX / 2 + mapX % 2 - 1
    for f in filler:
        mapX = mapX + 1
        off = abs((som_height - 1) / 2 - mapY)
        if mapX >= som_width - np.floor(off / 2) - off % 2 * (mapY) % 2:
            mapY = mapY + 1
            mapX = abs((som_height - 1) / 2 - mapY)
            mapX = np.floor(mapX / 2) + mapX % 2 * (1 - mapY) % 2
        # print(int(mapX),mapY, proto_count)
        filler_map[int(mapX), int(mapY)] = f
        filler_mask[int(mapX), int(mapY)] = False
    return filler_map, filler_mask


def u_matrify(data_map, layout, som_width, som_height, pbc=False, verbose=True):
    """Given a 1D mapped SOM data-array and its layout type,
    returns the U-matrix!"""

    if layout == "hexagonal":
        raise NotImplementedError
    else:
        u_matrix = np.zeros([som_width, som_height])
        t = 0
        for j in range(som_height):
            for i in range(som_width):
                neighbors = get_quadratic_neighbor_indexes(
                    i, j, som_width, som_height, pbc
                )
                ding = data_map[t].reshape(som_width, som_height)
                uh_i = np.sum(
                    [ding[neighbor[1], neighbor[0]] for neighbor in neighbors]
                )
                uh_i /= len(neighbors)
                u_matrix[i, j] = uh_i
                t += 1
    if verbose:
        print("u-matrix, rounded to two decimal")
        print(
            np.array(list(map(lambda x: round(x, 2), u_matrix.flatten()))).reshape(
                som_width, som_height
            )
        )
    return u_matrix


def u_matrify_old(trained_som, layout, som_width, som_height, pbc=False):
    """Given a 1D trained SOM data-array and its layout type,
    returns the U-matrix!"""

    if layout == "hexagonal":
        hex_size, hex_map = get_hex_size(som_width)
        u_matrix = np.zeros(hex_size)
        for m_i, m in enumerate(trained_som):
            neighbors = get_hex_neighbors(m_i, hex_map)
            uh_i = np.sum(
                [
                    scipy.spatial.distance.euclidean(m, trained_som[neighbor])
                    for neighbor in neighbors
                ]
            )
            uh_i /= len(neighbors)
            u_matrix[m_i] = uh_i
    else:
        u_matrix = np.zeros([som_width, som_height])
        for i in range(som_width):
            for j in range(som_height):
                neighbors = get_quadratic_neighbor_indexes(
                    i, j, som_width, som_height, pbc
                )
                uh_i = np.sum(
                    [
                        scipy.spatial.distance.euclidean(
                            trained_som[i, j], trained_som[neighbor[0], neighbor[1]]
                        )
                        for neighbor in neighbors
                    ]
                )
                uh_i /= len(neighbors)
                u_matrix[i, j] = uh_i
    return u_matrix


def plot_pie_chart(
    labels,
    sizes,
    save=True,
    save_dir=None,
    save_name=None,
    explode=None,
    angle=9,
    figsize_x=2,
    figsize_y=2,
):
    """Pie chart, where the slices will be ordered and plotted counter-clockwise"""

    fig1, ax1 = plt.subplots()
    ax1.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct="%1.0f%%",
        shadow=False,
        startangle=angle,
    )
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis("equal")
    # plt.title(title)
    plt.tight_layout()
    fig1 = plt.gcf()
    fig1.set_size_inches(figsize_x, figsize_y)  # or (4,4) or (5,5) or whatever
    if save:
        plt.savefig(
            os.path.join(save_dir, save_name + ".pdf"), dpi=300, bbox_inches="tight"
        )
    plt.show()
    plt.close()


def flatten(l):
    """Flatten a list or numpy array"""
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def return_cutout(bin_path, cutout_id, version=1, verbose=False):
    """Open bin_path, return cut-out with id=cutout_id"""
    with open(bin_path, "rb") as file:
        number_of_channels = 1
        if version == 1:
            if verbose:
                print(cutout_id)
            number_of_images, number_of_channels, width, height = struct.unpack(
                "i" * 4, file.read(4 * 4)
            )
            data_dimensionality = 2
        if version == 2:
            # <file format version> 0 <data-type> <number of entries> <data layout> <data>
            (
                version,
                file_type,
                data_type,
                number_of_images,
                data_layout,
                data_dimensionality,
            ) = struct.unpack("i" * 6, file.read(4 * 6))
            data_dimensions = struct.unpack(
                "i" * data_dimensionality, file.read(4 * data_dimensionality)
            )
            if data_dimensionality < 3:
                width = data_dimensions[0]
                height = data_dimensions[1]
            else:
                number_of_channels = data_dimensions[0]
                width = data_dimensions[1]
                height = data_dimensions[2]
            # print("Data dimensionality:", data_dimensionality)
            # print("Data_dimensions:", data_dimensions)
        if cutout_id > number_of_images:
            raise Exception(
                f"Requested image ID {cutout_id} is larger than the number of images."
            )
        size = width * height
        if data_dimensionality == 3:
            file.seek((cutout_id * number_of_channels + 0) * size * 4, 1)
        else:
            file.seek((cutout_id * 1 + 0) * size * 4, 1)

        array = np.array(
            struct.unpack(
                "f" * number_of_channels * size,
                file.read(number_of_channels * size * 4),
            )
        )
        if data_dimensionality < 3:
            cutout = np.ndarray([width, height], "float", array)
        else:
            cutout = np.ndarray([number_of_channels, width, height], "float", array)

    return cutout


def pretty_plot(
    som,
    bin_path,
    id,
    i,
    clip_threshold=False,
    plot_border=True,
    debug=False,
    x=0,
    y=0,
    apply_clipping=False,
    overwrite=False,
    save=False,
    save_full=False,
    outliers_subpath="",
    outliers_path="",
    index_label=False,
    custom_save_name="",
    enable_degree_label=False,
    enable_hmsdms_label=False,
    version=1,
):

    # plot outlier and save to its outliers subfolder
    s = som.rotated_size
    w = som.fullsize  # Size needed for cut-out to be able to rotate it
    fig = plt.figure()
    fig.set_size_inches(3, 3)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    image = return_cutout(bin_path, id, version=version)
    if apply_clipping:
        image_clip = np.clip(image, clip_threshold * np.std(image), 1e10)
        image = np.hstack((image, image_clip))
    ax.imshow(
        image, aspect="equal", interpolation="nearest", origin="lower", cmap="viridis"
    )
    if index_label:
        ax.text(w - 2, 2, str(i), color="white", horizontalalignment="right")
    if enable_hmsdms_label:
        hmsdms_label = SkyCoord(
            ra=pd_catalogue["RA"].iloc[id],
            dec=pd_catalogue["DEC"].iloc[id],
            unit=u.degree,
        ).to_string("hmsdms")
        ax.text(w - 2, 2, hmsdms_label, color="white", horizontalalignment="right")
    elif enable_degree_label:
        ax.text(
            w - 2,
            2,
            "({ra:.{digits}f}, {dec:.{digits}f})".format(
                ra=pd_catalogue["RA"].iloc[id],
                dec=pd_catalogue["DEC"].iloc[id],
                digits=3,
            ),
            color="white",
            horizontalalignment="right",
        )
    # plot border around the cropped center of the image
    if plot_border:
        ax.plot([(w - s) / 2, w - (w - s) / 2], [(w - s) / 2, (w - s) / 2], "r")
        ax.plot([(w - s) / 2, (w - s) / 2], [(w - s) / 2, w - (w - s) / 2], "r")
        ax.plot([w - (w - s) / 2, w - (w - s) / 2], [(w - s) / 2, w - (w - s) / 2], "r")
        ax.plot([(w - s) / 2, w - (w - s) / 2], [w - (w - s) / 2, w - (w - s) / 2], "r")
    if debug:
        print("Prototype coordinates: " + str(x) + "," + str(y))
    if save:
        plt.savefig(os.path.join(outliers_subpath, f"{i:04d}_{custom_save_name}.png"))
    elif save_full:
        plt.savefig(os.path.join(outliers_path, f"{i:04d}.png"))
    else:
        plt.show()
    plt.close()


def plot_and_save_outliers(
    som,
    outliers_path,
    bin_path,
    number_of_outliers_to_show,
    pd_catalogue,
    closest_prototype_id,
    distance_to_bmu_sorted_down_id,
    clip_threshold=False,
    plot_border=True,
    not_in_xy=None,
    debug=False,
    apply_clipping=False,
    overwrite=False,
    save=True,
    save_full=False,
    plot_subset=None,
    subset_pickle_path=None,
    index_label=False,
    enable_degree_label=False,
    enable_hmsdms_label=False,
    version=1,
):
    """Save first number_of_outliers_to_show outliers to outliers_path in
    separate directories per prototype.
    Also save the outlier coordinates and catalogue entries."""
    # Width of cropped image used in the training

    outliers_to_show = []
    if plot_subset is None:
        # [:number_of_outliers_to_show]
        imlist = distance_to_bmu_sorted_down_id
    else:
        assert not subset_pickle_path is None
        imlist = distance_to_bmu_sorted_down_id[:number_of_outliers_to_show][
            plot_subset
        ]
        pd_catalogue_subset = pd_catalogue.iloc[imlist]
        pd_catalogue_subset.to_pickle(subset_pickle_path)
    for i, id in enumerate(imlist):
        skip = False
        if debug:
            if plot_subset is None:
                print(i)
            else:
                print(i, "outlier number:", plot_subset[i])
            print(pd_catalogue["RA"].iloc[id], pd_catalogue["DEC"].iloc[id])
            # print(id, pd_catalogue.iloc[id])
        x = int(closest_prototype_id[id] % som.som_width)
        y = int(closest_prototype_id[id] / som.som_width)
        if not not_in_xy is None:
            for xx, yy in not_in_xy:
                if x == xx and y == yy:
                    skip = True
        if skip or len(outliers_to_show) > number_of_outliers_to_show:
            continue
        outliers_to_show.append(i)
        # print('x,y=', x,y)

        # Make image dir for outliers
        outliers_subpath = os.path.join(outliers_path, str(x) + "_" + str(y))
        if overwrite or not os.path.exists(outliers_subpath):
            if save and not os.path.exists(outliers_subpath):
                os.makedirs(outliers_subpath)

            pretty_plot(
                som,
                bin_path,
                id,
                i,  # **kwargs)
                clip_threshold=clip_threshold,
                plot_border=plot_border,
                # plot_subset=None,
                debug=debug,
                apply_clipping=apply_clipping,
                overwrite=overwrite,
                save=save,
                save_full=save_full,
                outliers_subpath=outliers_subpath,
                outliers_path=outliers_path,
                enable_degree_label=enable_degree_label,
                index_label=index_label,
                enable_hmsdms_label=enable_hmsdms_label,
                version=version,
            )

    # Write source locations to website text file
    ra = pd_catalogue["RA"].iloc[distance_to_bmu_sorted_down_id[outliers_to_show]]
    dec = pd_catalogue["DEC"].iloc[distance_to_bmu_sorted_down_id[outliers_to_show]]
    with open(os.path.join(outliers_path, "loc.txt"), "w") as loc_f:
        for a, b in zip(ra, dec):
            loc_f.write("{};{}\n".format(a, b))

    # Write first 100 outliers to csv
    rows = pd_catalogue.iloc[distance_to_bmu_sorted_down_id[outliers_to_show]]
    # Add distances to bmu as separate column
    rows.to_csv(os.path.join(outliers_path, "outliers_catalogue.csv"), index=True)
    rows.to_hdf(os.path.join(outliers_path, "outliers_catalogue.h5"), "df")
    return rows


def plot_distance_to_bmu_histogram(
    data_map,
    number_of_outliers_to_show,
    outliers_path,
    save=False,
    xmax=200,
    run_id="",
    visualize_sizes=False,
):

    # Get distances to bmu
    distance_to_bmu = np.min(data_map, axis=1)

    # Get the maj-size that contains 90% of the sources
    sorted_extremes = np.sort(distance_to_bmu)

    # Linear
    fig = plt.figure(figsize=(8, 4))
    # fig.set_size_inches(9, 4.5)
    ax = fig.add_subplot(111)
    bins = ax.hist(
        distance_to_bmu, bins=50, histtype="step", linewidth=3
    )  # 'auto')#400)
    height = max(bins[0])
    sections = [0.9, 0.99, 0.999]
    cutoff = [sorted_extremes[int(len(distance_to_bmu) * x)] for x in sections]

    # Plot red line for outliers shown
    red_x = sorted_extremes[-number_of_outliers_to_show]
    red_y = height * 0.7
    plt.text(
        red_x + 1,
        red_y / 8,
        str(number_of_outliers_to_show) + " most \noutlying objects",
        color="r",
        fontsize=12,
    )
    # plt.text(0.5,0.6, str(number_of_outliers_to_show)+" biggest outliers shown below",
    #        color='r', transform=ax.transAxes)
    plt.vlines(red_x, ymax=2 * height, ymin=0, color="r", linestyle="dashed")
    plt.arrow(
        red_x,
        red_y / 2,
        3,
        0,
        shape="full",
        length_includes_head=True,
        head_width=height * 0.1,
        head_length=1,
        fc="r",
        ec="r",
    )

    # Visualize the size-distribution
    if visualize_sizes:
        hh = [height * 0.1, height * 0.1, height * 0.1]
        for c, s, h in zip(cutoff, sections, hh):
            plt.vlines(c, ymax=h * 0.95, ymin=0)
            print(
                "Cut-off that includes {0}% of the sources: {1} (= {2} x median)".format(
                    s, round(c, 1), round(c / np.median(distance_to_bmu), 1)
                )
            )
            plt.text(c, h, str(s * 100) + "% of sources")
            plt.arrow(
                c,
                h * 0.5,
                -10,
                0,
                shape="full",
                length_includes_head=True,
                head_width=height * 0.01,
                head_length=5,
                fc="k",
                ec="k",
            )

    if xmax == "":
        xmax = max(bins[1])
    info_x = xmax * 0.65
    plt.text(
        info_x,
        red_y / 15,
        """Euclidean norms 
Median: {}
Mean: {}
Std. dev.: {} 
Max.: {}(={}xmedian)""".format(
            str(round(np.median(distance_to_bmu), 1)),
            str(round(np.mean(distance_to_bmu), 1)),
            str(round(np.std(distance_to_bmu), 1)),
            str(round(max(distance_to_bmu), 1)),
            str(int(max(distance_to_bmu) / np.median(distance_to_bmu))),
        ),
        fontsize=12,
        bbox=dict(edgecolor="black", facecolor="None"),
    )
    # plt.ylim(0,height*1.05)
    plt.xlim(-1, xmax)
    plt.yscale("log")
    plt.ylim(0.6, height * 1.2)
    # plt.title('Histogram of distance to closest prototype')
    plt.xlabel("Euclidian norm to best matching prototype")
    plt.ylabel("Number of radio-sources per bin")
    plt.tight_layout()
    if save:
        if run_id == "":
            plt.savefig(outliers_path + "/outliers_histogram.png", transparent=True)
        else:
            plt.savefig(
                outliers_path + "/outliers_histogram_ID{}.eps".format(run_id),
                transparent=True,
            )
    plt.show()
    plt.close()


def plot_u_matrix(
    som,
    trained_path,
    output_directory,
    cbar=True,
    save=False,
    save_dir=None,
    mask=False,
    mask_threshold=3,
    max_tiles=0,
    version=1,
    overwrite=True,
    replace_nans=False,
    force_chosen_som_index=None,
):
    """
    Plot the u matrix of a som. Save it if you like. Mask (part of the) u-matrix using
    either a threshold or a maximum number of tiles. In the latter case, the tiles with the
    smallest 'max_tiles' u-value will be masked.
    """
    # Turn SOM file into data binary
    som_data_bin = os.path.join(
        output_directory,
        som.trained_subdirectory,
        f"umatrix_data_binary_ID{som.run_id}_chosen_som_index_{force_chosen_som_index}.bin",
    )
    map_bin = os.path.join(
        output_directory,
        som.trained_subdirectory,
        f"umatrix_map_binary_ID{som.run_id}_chosen_som_index_{force_chosen_som_index}.bin",
    )
    write_som_to_data_binary(
        som, som_data_bin, verbose=True, overwrite=overwrite, version=version
    )

    # Map each neuron to all other neurons to get summed euclidean distances
    alternate_neuron_dimension = None
    if version == 2:
        alternate_neuron_dimension = int(
            np.sqrt(np.shape(som.data_som)[-1] / som.number_of_channels)
        )
    exec_string = map_dataset_to_trained_som(
        som,
        som_data_bin,
        map_bin,
        trained_path,
        0,
        use_gpu=True,
        verbose=True,
        version=version,
        alternate_neuron_dimension=alternate_neuron_dimension,
    )

    # Unpack mapping
    try:
        umat_data_map, _, _, _, _ = load_som_mapping(
            map_bin, som, version=version, replace_nans=replace_nans, verbose=True
        )
    except FileNotFoundError:
        print("Now execute the PINK command above to get the mapping for the u-matrix")
        return None

    print(f"som pbc {som.pbc}")
    u_matrix = u_matrify(
        umat_data_map, som.layout, som.som_width, som.som_height, pbc=som.pbc
    )
    if version == 1:
        pass  # u_matrix = u_matrix.T
    fig = plt.figure(figsize=[6, 6])
    if som.layout == "hexagonal":
        raise NotImplementedError
        u_matrix_map, u_matrix_mask = populate_hex_map(
            u_matrix, som.som_width, som.som_height
        )
        sns.heatmap(
            u_matrix_map.T,
            annot=False,
            square=True,
            fmt="d",
            cbar=True,
            mask=u_matrix_mask.T,
            cmap="inferno",
        )  # , linewidths=.1)
    else:
        sns.heatmap(
            u_matrix, annot=False, square=True, fmt="d", cbar=True, cmap="inferno"
        )
    title = "U-matrix of" + som.som_label
    plt.title(title)
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(os.path.join(save_dir, title + ".png"), bbox="tight", dpi=300)
    # plt.close()

    # Return indexes with mask_threshold
    if mask:
        title = "U-matrix of" + som.som_label
        plt.title(title)
        if max_tiles > 0:
            assert som.layout == "quadratic"
            if mask_threshold != 3:
                print(
                    "Once used, the max_tiles parameter deprecates the mask_threshold parameter."
                )
            mask_threshold = np.sort(u_matrix.reshape(som.som_width * som.som_height))[
                max_tiles
            ]
        u_matrix_mask2 = u_matrix < mask_threshold
        sns.heatmap(
            u_matrix,
            mask=u_matrix_mask2,
            annot=False,
            square=True,
            fmt="d",
            cbar=True,
            cmap="inferno",
        )
        plt.show()
        plt.close()
        a, b = np.where(u_matrix < mask_threshold)
        return [[i, j] for j, i in zip(a, b)]


def show_prototypes(som, indexes):
    """Given a som and indexes plots the corresponding prototypes"""
    for a, b in indexes:
        print(a, b)

        neuron_size = int(np.sqrt(som.data_som[a, b, 0].size))
        plt.matshow(
            som.data_som[a, b, 0].reshape(neuron_size, neuron_size), cmap="viridis"
        )
        plt.grid(False)
        plt.colorbar()
        plt.show()
    # plt.close()
    return som.data_som[a, b, 0].reshape(neuron_size, neuron_size)


def return_sparce_indices(dimension, debug=False):
    """Given a squared matrix (specifically a heatmap or som with
    shape equal to dimensionxdimension
    returns the indices needed to reduce the matrix to a squared
    int(dimension/2) + 1 x int(dimension/2)+1"""

    assert dimension > 2, "dimension must be at least 3"
    assert dimension % 2 == 1, "dimension must be odd"

    new_dimension = int(dimension / 2) + 1
    i = 0
    i_stored = []
    # Get indices for only odd rows and odd columns
    for row in np.arange(0, dimension, 1):
        for col in np.arange(0, dimension, 1):
            if row % 2 == 0 and col % 2 == 0:
                i_stored.append(i)
            i += 1
    if debug:
        # Show example of how sparce indices compress matrix a
        a = np.random.randint(1, 10, size=dimension * dimension)
        print("Matrix with old dimensions")
        print(a.reshape(dimension, dimension))
        print("Matrix with new dimensions")
        print(a[i_stored].reshape(new_d, new_d))
    return i_stored, new_dimension


def plot_som_bmu_heatmap(
    som,
    closest_prototype_id,
    data_map=None,
    cbar=False,
    save=False,
    compress=False,
    ax=None,
    return_n_closest=1,
    save_dir=None,
    save_name="heatmap",
    cmap="viridis",
    highlight=[],
    legend=False,
    legend_list=None,
    highlight_colors=None,
    gap=0,
    fontsize=20,
    debug=False,
):
    """
    Plot heatmap of trained SOM.
    With closest_prototype_id equal to
    data_map, _,_,_,_ = load_som_mapping(mapping_path,som)
    closest_prototype_id = np.argmin(data_map, axis=1)

    or insert data_map and we will ignore closest_prototype given
    """
    assert not data_map is None, "please insert data_map kwarg"
    # Flip SOM to align multiple different SOMs
    data_map = flip_datamap(
        data_map,
        som.som_width,
        som.som_height,
        flip_axis0=som.flip_axis0,
        flip_axis1=som.flip_axis1,
        rot90=som.rot90,
    )
    if compress:
        assert not data_map is None, "please insert data_map kwarg"
        assert (
            som.som_width == som.som_height
        ), "compression only implemented for square SOMs"
        assert (
            som.layout != "hexagonal"
        ), "Compression only implemented for cartesian SOMs"
        i_compress, new_dimension = return_sparce_indices(som.som_width)
        data_map = data_map[:, i_compress]
        som_width = new_dimension
        som_height = new_dimension
        if debug:
            print("new_dimension:", new_dimension)
            print("datamap shape:", data_map.shape)
    else:
        som_width = som.som_width
        som_height = som.som_height

    # Count how many cut-outs most resemble each prototype
    if not data_map is None:
        # closest_prototype_id = np.argmin(data_map, axis=1)
        closest_prototype_id = np.array([], dtype="int")
        sorted_map = np.argsort(data_map, axis=1)
        for i in range(1, return_n_closest + 1):
            tussen = np.array([l[:i] for l in sorted_map]).flatten()
            closest_prototype_id = np.concatenate([closest_prototype_id, tussen])
        # print("shape of close", np.shape(closest_prototype_id), closest_prototype_id)
        # closest_prototype_id = np.array([l[:return_n_closest] for l in np.argsort(data_map,axis=1)]).flatten()

    closest_to_prototype_count = np.bincount(closest_prototype_id)
    if debug:
        print("Total of the bincount is: ", np.sum(closest_to_prototype_count))
        print("closest_to_prototype_count len ", len(closest_to_prototype_count))
    closest_to_prototype_count = np.concatenate(
        [
            closest_to_prototype_count,
            np.zeros(
                som_width * som_height - len(closest_to_prototype_count), dtype=int
            ),
        ]
    )
    # old_count = copy.deepcopy(closest_to_prototype_count).reshape([som_width,som_height])

    # plot and save the heatmap
    if ax is None:
        fig = plt.figure(figsize=(14, 14))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
    else:
        ax.set_axis_off()

    if som.layout == "hexagonal":
        heatmap_map, heatmap_mask = populate_hex_map(
            closest_to_prototype_count, som_width, som_height
        )
        # Note the transposition of closest_to_prototype_count
        ax = sns.heatmap(
            heatmap_map.T,
            annot=True,
            mask=heatmap_mask.T,
            fmt="d",
            cbar=cbar,
            cmap=cmap,
            annot_kws={"size": fontsize},
        )
    else:
        closest_to_prototype_count = closest_to_prototype_count.reshape(
            [som_width, som_height]
        )
        sns.heatmap(
            closest_to_prototype_count.T,
            annot=True,
            fmt="d",
            ax=ax,
            cbar=cbar,
            cmap=cmap,
            square=True,
            linecolor="k",
            linewidths=0.5,
            annot_kws={"size": fontsize, "va": "center", "ha": "center"},
            cbar_kws={"shrink": 0.8},
        )
    # Highlight is a list of sublists, every sublist can contain a number of prototypes
    # all prototypes within a sublist are grouped. This way you can manually
    # color-code your SOM
    if not highlight == []:
        print("Entering highlight mode")
        appearance_list = []
        if not legend:
            legend_list = np.ones(len(highlight))
        else:
            assert not legend_list == []
        unit_size = 1
        linewidth = 8
        gap = 0
        for group_index, (group, col, legend_label) in enumerate(
            zip(highlight, highlight_colors, legend_list)
        ):
            legend_flag = True
            for h in group:
                ss = "solid"
                if legend_flag:
                    ax.add_patch(
                        Rectangle(
                            (
                                h[0] * (unit_size + gap) + gap / 2,
                                h[1] * (unit_size + gap) + gap / 2,
                            ),
                            unit_size,
                            unit_size,
                            alpha=1,
                            facecolor="None",
                            edgecolor=col,
                            linewidth=linewidth,
                            label=legend_label,
                            zorder=100,
                        )
                    )  # group_index+1))
                    if h in appearance_list:
                        print(
                            "To enable dashes move",
                            h,
                            "to later part in the sequence (or else the \
                            legend will be ugly)",
                        )
                    legend_flag = False
                else:
                    if appearance_list.count(h) == 1:
                        # voor 2
                        # print('2-categories appearance', h)
                        ss = (0, (6, 6))
                    elif appearance_list.count(h) == 2:
                        # voor 3
                        # print('3-categories appearance', h)
                        ss = (3, (3, 6))
                    ax.add_patch(
                        Rectangle(
                            (
                                h[0] * (unit_size + gap) + gap / 2,
                                h[1] * (unit_size + gap) + gap / 2,
                            ),
                            unit_size,
                            unit_size,
                            alpha=1,
                            facecolor="None",
                            edgecolor=col,
                            linewidth=linewidth,
                            label="_nolegend_",
                            linestyle=ss,
                            zorder=100,
                        )
                    )
                appearance_list.append(h)
        if legend:
            ax.legend(
                bbox_to_anchor=(1.04, 0.5),
                loc="center left",
                ncol=1,
                # ax.legend(bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=1,
                prop={"size": 24},
            )
    if save:
        plt.savefig(os.path.join(save_dir, save_name + ".png"))
    if not ax is None:
        return ax, closest_to_prototype_count
    else:
        plt.show()
        plt.close()
        return None, closest_to_prototype_count


def query_SIMBAD(ra, dec, object_search_radius_in_arcsec=10, verbose=False):
    """Query SIMBAD to find out if an object exists, returns its type.
    querying a list of ras and decs wont work as not found objects are not returned :("""
    customSimbad = Simbad()
    customSimbad.add_votable_fields("otype", "otypes", "dim")
    customSimbad.remove_votable_fields("coordinates")
    a = customSimbad.query_region(
        SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="fk5"),
        radius=object_search_radius_in_arcsec * u.arcsec,
    )
    maj_arcsec = None
    if a is None:
        object_type = None
        object_types = None
        status_message = (
            f"SIMBAD did not find an object near the RA;DEC of {ra:.3f} {dec:.3f}"
        )
    else:
        object_name = a[0][0].decode("utf-8")
        object_type = a[0][1].decode("utf-8")
        object_types = a[0][2].decode("utf-8")
        maj_arcsec = a[0][3] * 60
        if not isinstance(maj_arcsec, float):
            maj_arcsec = None
            status_message = (
                f"{object_name} is a {object_type} located near the RA;DEC "
                f"of {ra:.3f} {dec:.3f}"
            )
        else:
            status_message = (
                f"{object_name} is a {object_type} located near the RA;DEC "
                f"of {ra:.3f} {dec:.3f} \nwith an angular diameter of {maj_arcsec:.2g} arcsec"
            )
    if verbose:
        print(status_message)
    return object_type, object_types, maj_arcsec, status_message


def get_SIMBAD_object_type_slow_webscrape(
    ra, dec, object_search_radius_in_arcsec=10, verbose=True
):
    """Query SIMBAD to find out if an object exists, returns its type"""
    url = f"https://simbad.u-strasbg.fr/simbad/sim-coo?output.format=ASCII&Coord={ra}+{dec}&Radius={object_search_radius_in_arcsec}&Radius.unit=arcsec"
    url += "&output.max=1&otypedisp=V&obj.spsel=off&obj.fluxsel=off&obj.bibsel=off&obj.messel=off&obj.notesel=off&obj.rvsel=off&obj.plxsel=off"
    query_result = urlopen(url).read().decode("ascii").split("\n")
    object_type = None
    status_message = ""
    # print(query_result)
    if len(query_result) > 5:
        object_name, object_type, _, _ = map(
            lambda x: x.strip(), query_result[5].split(" --- ")
        )
        object_name = object_name[7:]
        if object_type.lower().startswith(("a", "o", "u", "i")):
            status_message = f"{object_name} is an {object_type} located near the RA;DEC of {ra:.3f} {dec:.3f}"
        else:
            status_message = f"{object_name} is a {object_type} located near the RA;DEC of {ra:.3f} {dec:.3f}"
    else:
        status_message = (
            f"SIMBAD did not find an object near the RA;DEC of {ra:.3f} {dec:.3f}"
        )

    if verbose:
        print(status_message)
    return object_type, status_message


def return_PANSTARRS_cutout_images(ra, dec, size=240, filters="grizy"):
    """Query ps1filenames.py service to get a list of images

    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    assert isinstance(ra, float)
    assert isinstance(dec, float)

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = (
        "{service}?ra={ra}&dec={dec}&size={size}&format=fits" "&filters={filters}"
    ).format(**locals())
    table = None
    try:
        table = Table.read(url, format="ascii")
    except:
        print(f"Unable to read url:", url)
    return table


def return_PANSTARRS_cutout_url(
    ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False
):
    """Get URL for images in the table

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """

    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = return_PANSTARRS_cutout_images(ra, dec, size=size, filters=filters)
    if table is None:
        return None
    url = (
        "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
        "ra={ra}&dec={dec}&size={size}&format={format}"
    ).format(**locals())
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table["filter"]]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0, len(table) // 2, len(table) - 1]]
        for i, param in enumerate(["red", "green", "blue"]):
            url = url + "&{}={}".format(param, table["filename"][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table["filename"]:
            url.append(urlbase + filename)
    return url


def return_PANSTARRS_cutout(
    ra,
    dec,
    size_in_arcsec,
    filters="i",
    return_zeros_if_outside_of_field=False,
    overwrite=False,
    save_dir="/home/rafael/data/mostertrij/data",
    arcsec_per_pixel=0.25,
):
    """Returns a PANSTARRS cutout for the specified filter (only one possible filter at a time)

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel) must be int
    filters = string with filters to include possible filters are: "grizy"
    """
    assert isinstance(size_in_arcsec, int) or isinstance(size_in_arcsec, float)
    assert isinstance(ra, float)
    assert isinstance(dec, float)

    size = int(round(size_in_arcsec / arcsec_per_pixel))
    save_dir = os.path.join(save_dir, "PANSTARRS")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ra{ra}dec{dec}size{size}filter{filters}.npy")
    if not overwrite and os.path.exists(save_path):
        fim = np.load(save_path)
    else:

        fitsurl = return_PANSTARRS_cutout_url(
            ra, dec, size=size, filters=filters, format="fits"
        )
        if fitsurl is None:
            if return_zeros_if_outside_of_field:
                return np.zeros([size, size])
            else:
                return None
        try:

            with fits.open(fitsurl[0], memmap=True) as fh:
                # fh = fits.open(fitsurl[0])
                fim = fh[0].data
                # replace NaN values with zero for display
                fim[np.isnan(fim)] = 0.0
                np.save(save_path, fim)
        except:
            return None

    return fim


def plot_FIRST_or_PANSTARRS(
    ra,
    dec,
    size_in_arcsec,
    filters="i",
    figsize=3,
    ax=None,
    plot_reticle=True,
    title="FIRST",
    overwrite=False,
    asinh_transform=False,
    plot_FIRST=True,
    plot_PANSTARRS=False,
    cmap="gray",
    save=False,
    save_path=None,
    **kwarg,
):
    """Plots a PANSTARRS cutout for the specified fiter (only one possible filter at a time)

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel) must be int
    filters = string with filters to include possible filters are: "grizy"
    """
    assert not plot_FIRST == plot_PANSTARRS

    if plot_FIRST:
        fim = return_FIRST_cutout(
            ra, dec, size_in_arcsec, overwrite=overwrite, mode="partial", **kwarg
        )
    if plot_PANSTARRS:
        fim = return_PANSTARRS_cutout(
            ra,
            dec,
            size_in_arcsec,
            filters="i",
            return_zeros_if_outside_of_field=False,
            overwrite=overwrite,
            **kwarg,
        )

    if fim is None or fim.size == 1:
        return None

    # set contrast to something reasonable
    if asinh_transform:
        transform = vis.AsinhStretch() + vis.PercentileInterval(99.5)
        fim = transform(fim)

    if ax is None:
        fig = plt.figure(figsize=(figsize, figsize))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.title(title)
        ax.imshow(
            fim, cmap=cmap, aspect="equal", interpolation="nearest", origin="lower"
        )
        if plot_reticle:
            ax.plot(np.shape(fim)[0] / 2, np.shape(fim)[1] / 2, "r+")
            x_offset, y_offset = 10, 10
            if plot_FIRST:
                scale_bar_length = (
                    10 / 1.8
                )  # 10arcsec/0.25arcsec_per_pixel_resolution=40
            if plot_PANSTARRS:
                scale_bar_length = (
                    10 / 0.25
                )  # 10arcsec/0.25arcsec_per_pixel_resolution=40
            scale_bar_color = "white"
            ax.plot(
                [x_offset, x_offset + scale_bar_length],
                [y_offset, y_offset],
                "-",
                color=scale_bar_color,
                linewidth=3,
            )
            bf = ax.text(
                x_offset, y_offset + 8, '10"', color=scale_bar_color, fontsize=18
            )
        if save:
            plt.savefig(save_path)
        else:
            plt.show()
    else:
        ax.title.set_text(title)
        ax.axis("off")
        bf = ax.imshow(
            fim, cmap=cmap, aspect="equal", interpolation="nearest", origin="lower"
        )
        if plot_reticle:
            bf = ax.plot(np.shape(fim)[0] / 2, np.shape(fim)[1] / 2, "r+")
            if plot_FIRST:
                x_offset, y_offset = 3, 3
                scale_bar_length = (
                    10 / 1.8
                )  # 10arcsec/0.25arcsec_per_pixel_resolution=40
            if plot_PANSTARRS:
                x_offset, y_offset = 10, 10
                scale_bar_length = (
                    10 / 0.25
                )  # 10arcsec/0.25arcsec_per_pixel_resolution=40
            scale_bar_color = "white"
            ax.plot(
                [x_offset, x_offset + scale_bar_length],
                [y_offset, y_offset],
                "-",
                color=scale_bar_color,
                linewidth=3,
            )
            bf = ax.text(
                x_offset, y_offset + y_offset, '10"', color=scale_bar_color, fontsize=18
            )
        return bf


def check_for_FIRST_catalogue_entry(ra, dec, object_search_radius_in_arcsec=10):
    """Use Vizier to query FIRST.
    ra and dec are expected to be in degree"""

    assert isinstance(ra, float)
    assert isinstance(dec, float)
    result = Vizier.query_region(
        SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="fk5"),
        radius=object_search_radius_in_arcsec * u.arcsec,
        catalog="VIII/92/first14",
    )
    if len(result) > 0:
        for k in result.keys():
            rr = result[k].to_pandas()
            source_name = f"FIRST {rr['FIRST'][0].decode('utf-8')}"
            diameter_arcsec = rr["Maj"].values[0]
            integrated_flux_in_mJy = rr["Fint"].values[0]
            verbose_message = (
                f"Found {source_name}: int. flux is {integrated_flux_in_mJy}mJy at "
                f'1.4GHz; angular diam. is {diameter_arcsec:.2g}"'
            )
    else:
        verbose_message = "Not present in FIRST catalogue."
        source_name, diameter_arcsec, integrated_flux_in_mJy = None, None, None
    return source_name, integrated_flux_in_mJy, diameter_arcsec, verbose_message


def get_2MASX_angular_diameter_arcsec(
    ra, dec, object_search_radius_in_arcsec=10, overwrite=False
):
    """Use Vizier to query 2MASX.
    ra and dec are expected to be in degree"""

    assert isinstance(ra, float)
    assert isinstance(dec, float)

    store_path = f"ra{ra}dec{dec}searchradius{object_search_radius_in_arcsec}.npy"
    if overwrite or not os.path.exists(store_path):

        v = Vizier(columns=["*", "r.K20e"], catalog="2MASX")
        result = v.query_region(
            SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="fk5"),
            radius=object_search_radius_in_arcsec * u.arcsec,
            catalog="2MASX",
        )
        if len(result) > 0:
            for k in result.keys():
                rr = result[k].to_pandas()
                # print("2MASX result:",rr)
                if isinstance(rr["_2MASX"][0], str):
                    source_name = f"2MASX J{rr['_2MASX'][0]}"
                else:
                    source_name = f"2MASX J{rr['_2MASX'][0].decode('utf-8')}"
                diameter_arcsec = rr["r.K20e"].values[0]
                verbose_message = (
                    f"2MASX J{source_name} reports an angular diameter of "
                    f"{diameter_arcsec:.2g} arcsec."
                )
                # print(diameter_arcsec, type(diameter_arcsec))
                # if not isinstance(diameter_arcsec, float):
                #    diameter_arcsec = None
        else:
            verbose_message = "Not present in 2MASX catalogue."
            source_name, diameter_arcsec = None, None
        np.save(store_path, (source_name, diameter_arcsec, verbose_message))
    else:
        source_name, diameter_arcsec, verbose_message = np.load(
            store_path, allow_pickle=True
        )
        if not diameter_arcsec is None:
            diameter_arcsec = float(diameter_arcsec)

    return source_name, diameter_arcsec, verbose_message


def return_cutouts_closest_to_prototype(
    som,
    x,
    y,
    z,
    data_map,
    include_worse_matches=False,
    adapted_outlier_score=False,
    compress=False,
    correction_contribution=0.033,
    filter_on_outlier_threshold=None,
):
    """return the cutouts that best match the
    prototype with the given coordinates (x,y,z)"""
    assert som.layout in ["quadratic", "cartesian"]
    if compress:
        assert (
            som.som_width == som.som_height
        ), "compression only implemented for square SOMs"
        assert (
            som.layout != "hexagonal"
        ), "Compression only implemented for cartesian SOMs"
        i_compress, new_dimension = return_sparce_indices(som.som_width)
        data_map = data_map[:, i_compress]
        som_width = new_dimension
        som_height = new_dimension
    else:
        som_width = som.som_width
        som_height = som.som_height

    si = (x * som_height) + y
    # Also calculate outlier score improved by summed pixel value of prototype
    closest_resemblances = []
    if adapted_outlier_score:
        max_summed = []
        for i in range(som_width):
            for j in range(som_height):
                for k in range(som.som_depth):
                    max_summed.append(np.sum(som.data_som[i, j, k]))
        max_summed = np.max(max_summed)
        summed_pixelvalue_correction = max_summed / np.sum(som.data_som[x, y, z])
    if include_worse_matches:
        assert (
            adapted_outlier_score == False
        ), "Not implemented in combination with worse matches."
        # Get distances to prototype
        distances = data_map[:, si]
        distance_index_sorted = np.argsort(distances)
        closest_index = np.array(range(len(distances)))[distance_index_sorted]
        closest_distances = distances[closest_index]
    else:
        # Sort the euclidian distances to this prototype
        closest = np.argsort(data_map, axis=1)[:, 0]
        closest_index = [i for i, som_index in enumerate(closest) if som_index == si]
        closest_distances = np.min(data_map, axis=1)[closest_index]
        sort_index = np.argsort(closest_distances)
        closest_index = np.array(closest_index)[sort_index]
        closest_distances = np.array(closest_distances)[sort_index]
        # Also calculate outlier score improved by summed pixel value of prototype
        if adapted_outlier_score:
            closest_resemblances = np.array(
                closest_distances
                * (1 + summed_pixelvalue_correction * correction_contribution)
            )

            if not filter_on_outlier_threshold is None:
                bools = [closest_resemblances < filter_on_outlier_threshold]
                return (
                    closest_index[bools],
                    closest_distances[bools],
                    closest_resemblances[bools],
                )

    return closest_index, closest_distances, closest_resemblances


def filter_cutouts_present_in_FIRST_cat(
    cutout_ids,
    pandas_catalogue,
    object_search_radius_in_arcsec=10,
    take_half_LGZ_Size_as_search_radius=False,
    verbose=True,
):
    """Given a set of cutouts discard those that are present in the FIRST catalogue.
    Using Vizier to query FIRST catalogue.
    """
    accepted_ids, rejected_ids = [], []
    for cutout_id in cutout_ids:

        # Get source ra, dec and size
        ra, dec = (
            pandas_catalogue["RA"].iloc[cutout_id],
            pandas_catalogue["DEC"].iloc[cutout_id],
        )
        lotss_diam_arcsec = pandas_catalogue["LGZ_Size"].iloc[cutout_id]
        if take_half_LGZ_Size_as_search_radius:
            object_search_radius_in_arcsec = lotss_diam_arcsec / 2

        (
            source_name,
            integrated_flux_in_mJy,
            diameter_arcsec,
            verbose_message,
        ) = check_for_FIRST_catalogue_entry(
            ra, dec, object_search_radius_in_arcsec=object_search_radius_in_arcsec
        )

        if source_name is None:
            accepted_ids.append(cutout_id)
        else:
            if verbose:
                print("    Source rejected:", verbose_message)
            rejected_ids.append(cutout_id)
    if verbose:
        print(
            (
                f"{len(rejected_ids)} out of {len(cutout_ids)} sources were rejected because the"
                " they were found to be present in the FIRST catalogue"
                f" using a search radius of {object_search_radius_in_arcsec} arcsec."
            )
        )
    return accepted_ids, rejected_ids


def filter_cutouts_with_galaxy_scale_emission(
    cutout_ids,
    pandas_catalogue,
    radio_to_optical_extent_ratio=2,
    object_search_radius_in_arcsec=10,
    verbose=True,
):
    """Given a set of cutouts discard those that have galaxy scale emission. Based on 2MASX and
    SIMBAD query.
    """
    accepted_ids, rejected_ids = [], []
    for cutout_id in cutout_ids:

        # Get source ra, dec and size
        ra, dec = (
            pandas_catalogue["RA"].iloc[cutout_id],
            pandas_catalogue["DEC"].iloc[cutout_id],
        )
        lotss_diam_arcsec = pandas_catalogue["LGZ_Size"].iloc[cutout_id]
        MASX_objectname, MASX_diam_arcsec, message = get_2MASX_angular_diameter_arcsec(
            ra, dec, object_search_radius_in_arcsec=object_search_radius_in_arcsec
        )

        # if verbose:
        #    print(f'LGZsize {lotss_diam_arcsec:.2f} optical size {MASX_diam_arcsec}')
        if MASX_diam_arcsec is None or MASX_diam_arcsec < 6:
            accepted_ids.append(cutout_id)
        else:
            if lotss_diam_arcsec / MASX_diam_arcsec > radio_to_optical_extent_ratio:
                accepted_ids.append(cutout_id)
            else:
                if verbose:
                    print("    Source rejected:", message)
                    print(
                        (
                            f"    Radio {lotss_diam_arcsec:.2f}, 2MASX {MASX_diam_arcsec:.2f}, ratio"
                            f" is {lotss_diam_arcsec/MASX_diam_arcsec:.2f}"
                        )
                    )
                rejected_ids.append(cutout_id)
    if verbose:
        print(
            (
                f"{len(rejected_ids)} out of {len(cutout_ids)} sources were rejected because the"
                " angular extent of the optical emission is larger than"
                f" {radio_to_optical_extent_ratio} times the extend of the radio emission."
            )
        )
    return accepted_ids, rejected_ids


def plot_LoTSS(
    lotss_fullsize,
    cutout_ids,
    pandas_catalogue,
    bin_path,
    version=1,
    save=False,
    save_dir=None,
    save_index=None,
    query_SIMBAD_for_source_description=False,
    object_search_radius_in_arcsec=10,
    print_radio_to_optical_extent=False,
    overwrite=False,
    print_FIRST_cat_message=False,
    arcsec_per_pixel_lotss=1.5,
    plot_reticle=True,
):
    """Plot LoTSS and FIRST and PANSTARRS cutouts"""
    for cutout_id in cutout_ids:

        # Get source ra, dec and size
        ra, dec = (
            pandas_catalogue["RA"].iloc[cutout_id],
            pandas_catalogue["DEC"].iloc[cutout_id],
        )
        size_in_arcsec = int(
            np.ceil(pandas_catalogue["LGZ_Size"].iloc[cutout_id] * np.sqrt(2))
        )
        # Calculate rescale factor for the lotss scalebar
        rescale_factor = (lotss_fullsize * arcsec_per_pixel_lotss) / size_in_arcsec

        # Get 2MASX angular size
        if print_radio_to_optical_extent:
            (
                MASX_objectname,
                MASX_diam_arcsec,
                message,
            ) = get_2MASX_angular_diameter_arcsec(ra, dec)
            if MASX_diam_arcsec is None:
                extent_message = f"No 2MASX catalogue entry found."
            else:
                extent_message = (
                    f"Radio (LGZ_Size) to optical (2MASX) extent ratio is:"
                    f" {pandas_catalogue['LGZ_Size'].iloc[cutout_id]/MASX_diam_arcsec:.1f}"
                )

        # get the simbad object type for this ra and dec
        if query_SIMBAD_for_source_description:
            object_type, object_types, maj_arcsec, status_message = query_SIMBAD(
                ra, dec, verbose=True
            )
        if print_FIRST_cat_message:
            (
                FIRST_source_name,
                FIRST_integrated_flux_in_mJy,
                FIRST_diameter_arcsec,
                FIRST_verbose_message,
            ) = check_for_FIRST_catalogue_entry(
                ra, dec, object_search_radius_in_arcsec=object_search_radius_in_arcsec
            )

        # Create figure that will contain lotss, first and panstarrs
        f, ax1 = plt.subplots(1, 1, figsize=(3, 3))
        plot_cutout_in_subplot(
            ax1,
            cutout_id,
            bin_path,
            title="LoTSS DR1",
            version=version,
            figsize=3,
            plot_reticle=True,
            rescale_factor=rescale_factor,
        )
        if query_SIMBAD_for_source_description and not object_type is None:
            plt.suptitle(status_message, y=0.05, x=0.15, ha="left")
        elif print_radio_to_optical_extent:
            plt.suptitle(extent_message, y=0.05, x=0.15, ha="left")
        elif print_FIRST_cat_message:
            plt.suptitle(FIRST_verbose_message, y=0.05, x=0.15, ha="left")
        plt.grid(False)
        if save:
            if save_index is None:
                save_name = f"cutoutid{cutout_id}.png"
            else:
                save_name = f"nr{save_index:03d}_cutoutid{cutout_id}.png"
            plt.savefig(os.path.join(save_dir, save_name), bbox_inches="tight")
        else:
            plt.show()
        plt.close()


def plot_LoTSS_FIRST_and_PANSTARRS_remnant_inspect(
    lotss_fullsize,
    cutout_ids,
    pandas_catalogue,
    bin_path,
    version=1,
    save=False,
    save_dir=None,
    save_index=None,
    overwrite=False,
    dimensions_normal=True,
    sqrt_stretch=True,
    lower_sigma_limit=0,
    upper_sigma_limit=1e9,
    apply_clipping=False,
    cutouts=None,
    lotss_data_dir="/data/mostertrij/pink-basics/data_LoTSS_DR1",
    data_maps=[],
    store_directory="/data/mostertrij/data/LoTSS_DR2_temp",
    lotss_field_image_name="mosaic-blanked",
    som_objects=[],
    rotation_label=None,
    closest_prototype_ids=[],
    som_image_dir=None,
    som_titles=[],
    zoom_in=True,
    query_SIMBAD_for_source_description=False,
    take_half_LGZ_Size_as_search_radius=False,
    object_search_radius_in_arcsec=10,
    cutout_size_in_arcsec=1000,
    print_radio_to_optical_extent=False,
    print_FIRST_cat_message=False,
    arcsec_per_pixel_lotss=1.5,
    plot_reticle=True,
):
    """Plot LoTSS and FIRST and PANSTARRS cutouts"""
    failed_indices = []
    for t, cutout_id in enumerate(cutout_ids):
        # print(t, cutout_id)
        if save:
            if save_index is None:
                save_name = f"cutoutid{cutout_id}"
            elif save_index == "enumerate":
                save_name = f"nr{t:03d}_cutoutid{cutout_id}"
            else:
                save_name = f"nr{save_index:03d}_cutoutid{cutout_id}"
            if os.path.exists(os.path.join(save_dir, save_name)) and (
                overwrite == False
            ):
                continue

        # Get source ra, dec and size
        ra, dec = (
            pandas_catalogue["RA"].iloc[cutout_id],
            pandas_catalogue["DEC"].iloc[cutout_id],
        )
        rms = pandas_catalogue["cutout_rms"].iloc[cutout_id]
        field = pandas_catalogue["Mosaic_ID"].iloc[cutout_id]
        field_path = os.path.join(lotss_data_dir, field)
        size_in_arcsec = int(
            np.ceil(pandas_catalogue["LGZ_Size"].iloc[cutout_id] * np.sqrt(2))
        )
        if size_in_arcsec == 0:
            size_in_arcsec = int(
                np.ceil(pandas_catalogue["source_size"].iloc[cutout_id] * np.sqrt(2))
            )
        fullsize = size_in_arcsec / arcsec_per_pixel_lotss
        # Calculate rescale factor for the lotss scalebar
        rescale_factor = (lotss_fullsize * arcsec_per_pixel_lotss) / size_in_arcsec

        # Get 2MASX angular size
        if print_radio_to_optical_extent:
            (
                MASX_objectname,
                MASX_diam_arcsec,
                message,
            ) = get_2MASX_angular_diameter_arcsec(ra, dec)
            if MASX_diam_arcsec is None:
                extent_message = f"No 2MASX catalogue entry found."
            else:
                extent_message = (
                    f"Radio (LGZ_Size) to optical (2MASX) extent ratio is:"
                    f" {pandas_catalogue['LGZ_Size'].iloc[cutout_id]/MASX_diam_arcsec:.1f}"
                )

        # get the simbad object type for this ra and dec
        if take_half_LGZ_Size_as_search_radius:
            object_search_radius_in_arcsec = (
                pandas_catalogue["LGZ_Size"].iloc[cutout_id] / 2
            )
        if query_SIMBAD_for_source_description:
            object_type, object_types, maj_arcsec, status_message = query_SIMBAD(
                ra, dec, verbose=True
            )
        if print_FIRST_cat_message:
            (
                FIRST_source_name,
                FIRST_integrated_flux_in_mJy,
                FIRST_diameter_arcsec,
                FIRST_verbose_message,
            ) = check_for_FIRST_catalogue_entry(
                ra, dec, object_search_radius_in_arcsec=object_search_radius_in_arcsec
            )

        image_list = []
        cutout_size = cutout_size_in_arcsec / arcsec_per_pixel_lotss
        extract_attempt_counter = 0
        while (
            (len(image_list) == 0)
            and (cutout_size > 0.8 * fullsize)
            and (extract_attempt_counter < 10)
        ):

            single_source_catalogue = pd.DataFrame(
                data={"RA": [ra], "DEC": [dec], "cutout_rms": [rms]}
            )
            (
                image_list,
                single_source_catalogue,
            ) = single_fits_to_numpy_cutouts_using_astropy_better(
                cutout_size,
                single_source_catalogue,
                "RA",
                "DEC",
                field_path,
                lotss_field_image_name + ".fits",
                apply_clipping=apply_clipping,
                apply_mask=False,
                verbose=False,
                store_directory=store_directory,
                mode="partial",
                store_file=f"LoTSS_DR2_cutout_id{cutout_id}",
                dimensions_normal=dimensions_normal,
                variable_size=False,
                hdf=True,
                rescale=False,
                sqrt_stretch=False,
                destination_size=None,
                lower_sigma_limit=lower_sigma_limit,
                upper_sigma_limit=upper_sigma_limit,
                arcsec_per_pixel=arcsec_per_pixel_lotss,
                overwrite=overwrite,
            )

            try:

                single_source_catalogue = pd.DataFrame(
                    data={"RA": [ra], "DEC": [dec], "cutout_rms": [rms]}
                )
                (
                    image_list,
                    single_source_catalogue,
                ) = single_fits_to_numpy_cutouts_using_astropy_better(
                    cutout_size,
                    single_source_catalogue,
                    "RA",
                    "DEC",
                    field_path,
                    lotss_field_image_name + ".fits",
                    apply_clipping=apply_clipping,
                    apply_mask=False,
                    verbose=False,
                    store_directory=store_directory,
                    mode="partial",
                    store_file=f"LoTSS_DR2_cutout_id{cutout_id}",
                    dimensions_normal=dimensions_normal,
                    variable_size=False,
                    hdf=True,
                    rescale=False,
                    sqrt_stretch=False,
                    destination_size=None,
                    lower_sigma_limit=lower_sigma_limit,
                    upper_sigma_limit=upper_sigma_limit,
                    arcsec_per_pixel=arcsec_per_pixel_lotss,
                    overwrite=overwrite,
                )
            except:
                print(extract_attempt_counter)
                failed_indices.append(cutout_id)
                image_list = []
                cutout_size *= 0.8
                extract_attempt_counter += 1
                continue
            cutout_size *= 0.8
            extract_attempt_counter += 1

        if np.shape(image_list)[0] == 0:
            print(
                (
                    f"extract_counter {extract_attempt_counter} cutout_size={cutout_size}"
                    f"lotss_fullsize {fullsize}"
                )
            )
            continue
        # Create figure that will contain lotss, first and panstarrs
        # Plot figure with subplots of different sizes
        s = 4
        # set up subplot grid
        if len(som_objects) == 1:
            col = 8
            row = 3
        else:
            col = 9
            row = 3
        fig = plt.figure(figsize=(col * s, row * s))
        gridspec.GridSpec(row, col)

        # large subplot
        ax1 = plt.subplot2grid((row, col), (0, 1), colspan=3, rowspan=3)
        # Supertitle
        plt.suptitle(save_name)
        ax1.axis("off")
        image = image_list[0] * 1000  # to turn the values into mJy
        # Display the exact same thing as the above plot
        norm = vis.ImageNormalize(
            image, interval=vis.MinMaxInterval(), stretch=vis.SqrtStretch()
        )
        plt.imshow(
            image, aspect="equal", norm=norm, interpolation="nearest", origin="lower"
        )
        plt.title(
            f"LoTSS DR2:  RA, DEC {pandas_catalogue.iloc[cutout_id].RA:.3f}, {pandas_catalogue.iloc[cutout_id].DEC:.3f} [deg]",
            fontsize=16,
        )

        # Insert flux colorbar
        cbaxes = inset_axes(ax1, width="40%", height="3%", loc=4)
        ax1.tick_params(axis="y", colors="white")
        ax1.tick_params(axis="x", colors="white")
        plt.colorbar(cax=cbaxes, orientation="horizontal", label="mJy")

        if plot_reticle:
            x_offset, y_offset = 10, 10
            scale_bar_length = (
                50 / arcsec_per_pixel_lotss
            ) * rescale_factor  # 1.5arcsec_per_pixel_resolution
            # /(cutout_size_in_arcsec/arcsec_per_pixel_lotss) # 1.5arcsec_per_pixel_resolution
            scale_bar_length = 50 / arcsec_per_pixel_lotss
            scale_bar_color = "white"
            ax1.plot(
                [x_offset, x_offset + scale_bar_length],
                [y_offset, y_offset],
                "-",
                color=scale_bar_color,
                linewidth=3,
            )
            ax1.text(x_offset, y_offset + 5, '50"', color=scale_bar_color, fontsize=18)
            ax1.plot(np.shape(image)[0] / 2, np.shape(image)[1] / 2, "w+")

            f = len(image)
            ax1.plot(
                [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                [f / 2 + fullsize / 2, f / 2 + fullsize / 2],
                "-w",
                linewidth=1,
            )
            ax1.plot(
                [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                [f / 2 - fullsize / 2, f / 2 - fullsize / 2],
                "-w",
                linewidth=1,
            )
            ax1.plot(
                [f / 2 + fullsize / 2, f / 2 + fullsize / 2],
                [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                "-w",
                linewidth=1,
            )
            ax1.plot(
                [f / 2 - fullsize / 2, f / 2 - fullsize / 2],
                [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                "-w",
                linewidth=1,
            )

        # small subplot 1
        ax2 = plt.subplot2grid((row, col), (0, 0))
        if cutouts is None:
            plot_cutout_in_subplot(
                ax2,
                cutout_id,
                bin_path,
                title="LoTSS DR1",
                version=version,
                figsize=3,
                plot_reticle=True,
                rescale_factor=rescale_factor,
            )
        else:
            plt.title("LoTSS DR2 zoom-in \n(flux < 1.5 sigma removed)")
            ax2.axis("off")
            image = cutouts[cutout_id]
            bp = ax2.imshow(
                image, aspect="equal", interpolation="nearest", origin="lower"
            )

            x_offset, y_offset = 10, 10
            # 1.5arcsec_per_pixel_resolution
            scale_bar_length = (10 / 1.5) * rescale_factor
            scale_bar_color = "white"
            ax2.plot(
                [x_offset, x_offset + scale_bar_length],
                [y_offset, y_offset],
                "-",
                color=scale_bar_color,
                linewidth=3,
            )
            ax2.plot(np.shape(image)[0] / 2, np.shape(image)[1] / 2, "k+")
            bp = ax2.text(
                x_offset, y_offset + 5, '10"', color=scale_bar_color, fontsize=18
            )

        # small subplot 2
        ax3 = plt.subplot2grid((row, col), (1, 0))
        plot_FIRST_or_PANSTARRS(
            ra,
            dec,
            size_in_arcsec,
            ax=ax3,
            filters="i",
            cmap="gray",
            figsize=3,
            plot_reticle=True,
            asinh_transform=True,
            plot_FIRST=False,
            plot_PANSTARRS=True,
            title="PAN-STARRS filter=i",
            overwrite=False,
        )

        # small subplot 3
        ax4 = plt.subplot2grid((row, col), (2, 0))
        plot_FIRST_or_PANSTARRS(
            ra,
            dec,
            size_in_arcsec,
            ax=ax4,
            filters="i",
            cmap="viridis",
            figsize=3,
            plot_reticle=True,
            plot_FIRST=True,
            plot_PANSTARRS=False,
            title="FIRST",
            overwrite=False,
        )

        if len(som_objects) == 1:
            for ii, (som, data_map, som_title) in enumerate(
                zip(som_objects, data_maps, som_titles)
            ):
                #  Load som image and closest prototype ids
                ax = plt.subplot2grid((row, col), (ii, 4), colspan=2, rowspan=2)
                ax.axis("off")
                som_image = np.load(
                    os.path.join(som_image_dir, f"trained_SOM_image_ID{som.run_id}.npy")
                )
                som_pixel_w = np.shape(som_image)[0]
                som_pixel_h = np.shape(som_image)[1]
                # data_map_remnants = data_map[cutout_ids]#np.load(os.path.join(som_image_dir,f'datamap_to_remnants_ID{som.run_id}.npy'))
                closest_prototype_id = np.argmin(data_map, axis=1)

                # Plot trained SOM image
                ax.imshow(
                    som_image,
                    aspect="equal",
                    interpolation="nearest",
                    origin="upper",
                    cmap="viridis",
                )  # '_nolegend_')
                # Plot red square indicating the position of the source on the SOM
                y = closest_prototype_id[cutout_id] % som.som_width
                x = int(closest_prototype_id[cutout_id] / som.som_width)
                ax.add_patch(
                    Rectangle(
                        (
                            x * som_pixel_w / som.som_width,
                            y * som_pixel_h / som.som_height,
                        ),
                        som_pixel_w / som.som_width,
                        som_pixel_h / som.som_height,
                        linewidth=1,
                        edgecolor="r",
                        facecolor="none",
                    )
                )
                plt.title(som_title)

                # Plot small BMU to the right of the trained SOMS
                ax = plt.subplot2grid((row, col), (ii, 6))
                ax.axis("off")
                z = 0
                proto = som.data_som[x, y, z, :]
                # proto = som.data_som[x,y,z,:]
                proto_size = int(np.sqrt(len(proto)))
                proto = proto.reshape((proto_size, proto_size))
                r = proto_size / np.sqrt(2)
                if not rotation_label is None:
                    # Load rotation and flip
                    flips = np.load(
                        os.path.join(
                            som_image_dir, f"{rotation_label}_flip_ID{som.run_id}.npy"
                        )
                    )
                    rotations = np.load(
                        os.path.join(
                            som_image_dir,
                            f"{rotation_label}_rotation_ID{som.run_id}.npy",
                        )
                    )
                    proto = rotate(
                        proto, np.rad2deg(rotations[cutout_id]), reshape=False
                    )
                    # print(flips[cutout_id], np.rad2deg(rotations[cutout_id]))
                if zoom_in:
                    proto = proto[
                        int((proto_size - r) / 2) : int((proto_size + r) / 2),
                        int((proto_size - r) / 2) : int((proto_size + r) / 2),
                    ]
                ax.imshow(
                    proto,
                    aspect="equal",
                    interpolation="nearest",
                    origin="lower",
                    cmap="viridis",
                )
                # add red square as border of image
                # ax.add_patch( Rectangle((-0.5,-0.5),
                ax.add_patch(
                    Rectangle(
                        (-0.4, 0.0),
                        r - 1,
                        r - 1,
                        linewidth=3,
                        edgecolor="r",
                        facecolor="none",
                    )
                )
                plt.title("Best matching neuron")

                # Plot small BMU to the right of the trained SOMS
                ax = plt.subplot2grid((row, col), (ii + 2, 4), colspan=3)
                # ax.axis('off')
                calculate_AQE_per_prototype(
                    data_map,
                    som.rotated_size,
                    cutout_id,
                    verbose=False,
                    comparative_datamap=data_map[cutout_id],
                    percentiles_color="red",
                    ax=ax,
                    som_height=None,
                )
                plt.grid(False)
        else:
            # Plot small trained SOM images to the right of large subplot
            assert len(som_objects) < 4
            assert len(som_objects) == len(som_titles)
            for ii, (som, data_map, som_title) in enumerate(
                zip(som_objects, data_maps, som_titles)
            ):
                #  Load som image and closest prototype ids
                ax = plt.subplot2grid((row, col), (ii, 4))
                ax.axis("off")
                som_image = np.load(
                    os.path.join(som_image_dir, f"trained_SOM_image_ID{som.run_id}.npy")
                )
                som_pixel_w = np.shape(som_image)[0]
                som_pixel_h = np.shape(som_image)[1]
                # data_map_remnants = data_map[cutout_ids]#np.load(os.path.join(som_image_dir,f'datamap_to_remnants_ID{som.run_id}.npy'))
                closest_prototype_id = np.argmin(data_map, axis=1)

                # Plot trained SOM image
                ax.imshow(
                    som_image,
                    aspect="equal",
                    interpolation="nearest",
                    origin="upper",
                    cmap="viridis",
                )  # '_nolegend_')
                # Plot red square indicating the position of the source on the SOM
                y = closest_prototype_id[cutout_id] % som.som_width
                x = int(closest_prototype_id[cutout_id] / som.som_width)
                ax.add_patch(
                    Rectangle(
                        (
                            x * som_pixel_w / som.som_width,
                            y * som_pixel_h / som.som_height,
                        ),
                        som_pixel_w / som.som_width,
                        som_pixel_h / som.som_height,
                        linewidth=1,
                        edgecolor="r",
                        facecolor="none",
                    )
                )
                plt.title(som_title)

                # Plot small BMU to the right of the trained SOMS
                ax = plt.subplot2grid((row, col), (ii, 5))
                ax.axis("off")
                z = 0
                proto = som.data_som[x, y, z, :]
                # proto = som.data_som[x,y,z,:]
                proto_size = int(np.sqrt(len(proto)))
                proto = proto.reshape((proto_size, proto_size))
                r = proto_size / np.sqrt(2)
                if not rotation_label is None:
                    # Load rotation and flip
                    flips = np.load(
                        os.path.join(
                            som_image_dir, f"{rotation_label}_flip_ID{som.run_id}.npy"
                        )
                    )
                    rotations = np.load(
                        os.path.join(
                            som_image_dir,
                            f"{rotation_label}_rotation_ID{som.run_id}.npy",
                        )
                    )
                    proto = rotate(
                        proto, np.rad2deg(rotations[cutout_id]), reshape=False
                    )
                    # print(flips[cutout_id], np.rad2deg(rotations[cutout_id]))
                if zoom_in:
                    proto = proto[
                        int((proto_size - r) / 2) : int((proto_size + r) / 2),
                        int((proto_size - r) / 2) : int((proto_size + r) / 2),
                    ]
                ax.imshow(
                    proto,
                    aspect="equal",
                    interpolation="nearest",
                    origin="lower",
                    cmap="viridis",
                )
                # add red square as border of image
                # ax.add_patch( Rectangle((-0.5,-0.5),
                ax.add_patch(
                    Rectangle(
                        (-0.4, 0.0),
                        r - 1,
                        r - 1,
                        linewidth=3,
                        edgecolor="r",
                        facecolor="none",
                    )
                )
                plt.title("Best matching neuron")

                # Plot small BMU to the right of the trained SOMS
                ax = plt.subplot2grid((row, col), (ii, 6), colspan=3)
                # ax.axis('off')
                calculate_AQE_per_prototype(
                    data_map,
                    som.rotated_size,
                    cutout_id,
                    verbose=False,
                    comparative_datamap=data_map[cutout_id],
                    percentiles_color="red",
                    ax=ax,
                    som_height=None,
                )
                plt.grid(False)

        if query_SIMBAD_for_source_description and not object_type is None:
            plt.suptitle(status_message, y=0.05, x=0.15, ha="left")
        elif print_radio_to_optical_extent:
            plt.suptitle(extent_message, y=0.05, x=0.15, ha="left")
        elif print_FIRST_cat_message:
            plt.suptitle(FIRST_verbose_message, y=0.05, x=0.15, ha="left")

        plt.grid(False)
        if save:
            plt.savefig(os.path.join(save_dir, save_name + ".jpg"), bbox_inches="tight")
        else:
            plt.show()
        plt.close()


def plot_LoTSS_FIRST_and_PANSTARRS_inspect(
    lotss_fullsize,
    cutout_ids,
    pandas_catalogue,
    bin_path,
    version=1,
    save=False,
    save_dir=None,
    save_index=None,
    overwrite=False,
    dimensions_normal=True,
    sqrt_stretch=True,
    lower_sigma_limit=0,
    upper_sigma_limit=1e9,
    apply_clipping=False,
    cutouts=None,
    lotss_data_dir="/data/mostertrij/pink-basics/data_LoTSS_DR1",
    lotss_field_image_name="mosaic-blanked",
    query_SIMBAD_for_source_description=False,
    take_half_LGZ_Size_as_search_radius=False,
    object_search_radius_in_arcsec=10,
    cutout_size_in_arcsec=1000,
    print_radio_to_optical_extent=False,
    print_FIRST_cat_message=False,
    arcsec_per_pixel_lotss=1.5,
    plot_reticle=True,
):
    """Plot LoTSS and FIRST and PANSTARRS cutouts"""
    failed_indices = []
    for t, cutout_id in enumerate(cutout_ids):
        print(t, cutout_id)
        if save:
            if save_index is None:
                save_name = f"cutoutid{cutout_id}.jpg"
            else:
                save_name = f"nr{save_index:03d}_cutoutid{cutout_id}.jpg"
            if os.path.exists(os.path.join(save_dir, save_name)) and (
                overwrite == False
            ):
                continue

        # Get source ra, dec and size
        ra, dec = (
            pandas_catalogue["RA"].iloc[cutout_id],
            pandas_catalogue["DEC"].iloc[cutout_id],
        )
        rms = pandas_catalogue["cutout_rms"].iloc[cutout_id]
        field = pandas_catalogue["Mosaic_ID"].iloc[cutout_id]
        field_path = os.path.join(lotss_data_dir, field)
        size_in_arcsec = int(
            np.ceil(pandas_catalogue["LGZ_Size"].iloc[cutout_id] * np.sqrt(2))
        )
        if size_in_arcsec == 0:
            size_in_arcsec = int(
                np.ceil(pandas_catalogue["source_size"].iloc[cutout_id] * np.sqrt(2))
            )
        fullsize = size_in_arcsec / arcsec_per_pixel_lotss
        # Calculate rescale factor for the lotss scalebar
        rescale_factor = (lotss_fullsize * arcsec_per_pixel_lotss) / size_in_arcsec

        # Get 2MASX angular size
        if print_radio_to_optical_extent:
            (
                MASX_objectname,
                MASX_diam_arcsec,
                message,
            ) = get_2MASX_angular_diameter_arcsec(ra, dec)
            if MASX_diam_arcsec is None:
                extent_message = f"No 2MASX catalogue entry found."
            else:
                extent_message = (
                    f"Radio (LGZ_Size) to optical (2MASX) extent ratio is:"
                    f" {pandas_catalogue['LGZ_Size'].iloc[cutout_id]/MASX_diam_arcsec:.1f}"
                )

        # get the simbad object type for this ra and dec
        if take_half_LGZ_Size_as_search_radius:
            object_search_radius_in_arcsec = (
                pandas_catalogue["LGZ_Size"].iloc[cutout_id] / 2
            )
        if query_SIMBAD_for_source_description:
            object_type, object_types, maj_arcsec, status_message = query_SIMBAD(
                ra, dec, verbose=True
            )
        if print_FIRST_cat_message:
            (
                FIRST_source_name,
                FIRST_integrated_flux_in_mJy,
                FIRST_diameter_arcsec,
                FIRST_verbose_message,
            ) = check_for_FIRST_catalogue_entry(
                ra, dec, object_search_radius_in_arcsec=object_search_radius_in_arcsec
            )

        image_list = []
        cutout_size = cutout_size_in_arcsec / arcsec_per_pixel_lotss
        extract_attempt_counter = 0
        while (
            (len(image_list) == 0)
            and (cutout_size > 0.8 * fullsize)
            and (extract_attempt_counter < 10)
        ):

            try:

                single_source_catalogue = pd.DataFrame(
                    data={"RA": [ra], "DEC": [dec], "cutout_rms": [rms]}
                )
                (
                    image_list,
                    single_source_catalogue,
                ) = single_fits_to_numpy_cutouts_using_astropy_better(
                    cutout_size,
                    single_source_catalogue,
                    "RA",
                    "DEC",
                    field_path,
                    lotss_field_image_name + ".fits",
                    apply_clipping=apply_clipping,
                    apply_mask=False,
                    verbose=False,
                    store_directory="/data/mostertrij/data/LoTSS_DR2_temp",
                    mode="partial",
                    store_file=f"LoTSS_DR2_cutout_id{cutout_id}",
                    dimensions_normal=dimensions_normal,
                    variable_size=False,
                    hdf=True,
                    rescale=False,
                    sqrt_stretch=False,
                    destination_size=None,
                    lower_sigma_limit=lower_sigma_limit,
                    upper_sigma_limit=upper_sigma_limit,
                    arcsec_per_pixel=arcsec_per_pixel_lotss,
                    overwrite=overwrite,
                )
            except:
                print(extract_attempt_counter)
                failed_indices.append(cutout_id)
                image_list = []
                cutout_size *= 0.8
                extract_attempt_counter += 1
                continue
            cutout_size *= 0.8
            extract_attempt_counter += 1

        if np.shape(image_list)[0] == 0:
            print(
                (
                    f"extract_counter {extract_attempt_counter} cutout_size={cutout_size}"
                    f"lotss_fullsize {fullsize}"
                )
            )
            continue
        # Create figure that will contain lotss, first and panstarrs
        # Plot figure with subplots of different sizes
        s = 4
        fig = plt.figure(figsize=(4 * s, 3 * s))
        # set up subplot grid
        gridspec.GridSpec(3, 4)

        # large subplot
        ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
        ax1.axis("off")
        image = image_list[0] * 1000  # to turn the values into mJy
        # Display the exact same thing as the above plot
        norm = vis.ImageNormalize(
            image, interval=vis.MinMaxInterval(), stretch=vis.SqrtStretch()
        )
        plt.imshow(
            image, aspect="equal", norm=norm, interpolation="nearest", origin="lower"
        )
        plt.title(
            f"LoTSS DR2:  RA, DEC {pandas_catalogue.iloc[cutout_id].RA:.3f}, {pandas_catalogue.iloc[cutout_id].DEC:.3f} [deg]",
            fontsize=16,
        )
        # ax1.set_title(f'RA, DEC {pandas_catalogue.iloc[cutout_id].RA:.3f},',
        #        f'{pandas_catalogue.iloc[cutout_id].DEC:.3f}')
        # plt.imshow(image, aspect='equal',interpolation="nearest", origin='lower' )

        # fig.colorbar(image)

        cbaxes = inset_axes(ax1, width="40%", height="3%", loc=4)
        ax1.tick_params(axis="y", colors="white")
        ax1.tick_params(axis="x", colors="white")
        plt.colorbar(cax=cbaxes, orientation="horizontal", label="mJy")

        if plot_reticle:
            x_offset, y_offset = 10, 10
            scale_bar_length = (
                50 / arcsec_per_pixel_lotss
            ) * rescale_factor  # 1.5arcsec_per_pixel_resolution
            # /(cutout_size_in_arcsec/arcsec_per_pixel_lotss) # 1.5arcsec_per_pixel_resolution
            scale_bar_length = 50 / arcsec_per_pixel_lotss
            scale_bar_color = "white"
            ax1.plot(
                [x_offset, x_offset + scale_bar_length],
                [y_offset, y_offset],
                "-",
                color=scale_bar_color,
                linewidth=3,
            )
            ax1.text(x_offset, y_offset + 5, '50"', color=scale_bar_color, fontsize=18)
            ax1.plot(np.shape(image)[0] / 2, np.shape(image)[1] / 2, "w+")

            f = len(image)
            ax1.plot(
                [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                [f / 2 + fullsize / 2, f / 2 + fullsize / 2],
                "-w",
                linewidth=1,
            )
            ax1.plot(
                [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                [f / 2 - fullsize / 2, f / 2 - fullsize / 2],
                "-w",
                linewidth=1,
            )
            ax1.plot(
                [f / 2 + fullsize / 2, f / 2 + fullsize / 2],
                [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                "-w",
                linewidth=1,
            )
            ax1.plot(
                [f / 2 - fullsize / 2, f / 2 - fullsize / 2],
                [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                "-w",
                linewidth=1,
            )

        # small subplot 1
        ax2 = plt.subplot2grid((3, 4), (0, 3))
        if cutouts is None:
            plot_cutout_in_subplot(
                ax2,
                cutout_id,
                bin_path,
                title="LoTSS DR1",
                version=version,
                figsize=3,
                plot_reticle=True,
                rescale_factor=rescale_factor,
            )
        else:
            plt.title("LoTSS DR2 zoom-in \n(flux < 1.5 sigma removed)")
            ax2.axis("off")
            image = cutouts[cutout_id]
            bp = ax2.imshow(
                image, aspect="equal", interpolation="nearest", origin="lower"
            )

            x_offset, y_offset = 10, 10
            # 1.5arcsec_per_pixel_resolution
            scale_bar_length = (10 / 1.5) * rescale_factor
            scale_bar_color = "white"
            ax2.plot(
                [x_offset, x_offset + scale_bar_length],
                [y_offset, y_offset],
                "-",
                color=scale_bar_color,
                linewidth=3,
            )
            ax2.plot(np.shape(image)[0] / 2, np.shape(image)[1] / 2, "k+")
            bp = ax2.text(
                x_offset, y_offset + 5, '10"', color=scale_bar_color, fontsize=18
            )

        # small subplot 2
        ax3 = plt.subplot2grid((3, 4), (1, 3))
        plot_FIRST_or_PANSTARRS(
            ra,
            dec,
            size_in_arcsec,
            ax=ax3,
            filters="i",
            cmap="gray",
            figsize=3,
            plot_reticle=True,
            asinh_transform=True,
            plot_FIRST=False,
            plot_PANSTARRS=True,
            title="PAN-STARRS filter=i",
            overwrite=False,
        )

        # small subplot 3
        ax4 = plt.subplot2grid((3, 4), (2, 3))
        plot_FIRST_or_PANSTARRS(
            ra,
            dec,
            size_in_arcsec,
            ax=ax4,
            filters="i",
            cmap="viridis",
            figsize=3,
            plot_reticle=True,
            plot_FIRST=True,
            plot_PANSTARRS=False,
            title="FIRST",
            overwrite=False,
        )

        if query_SIMBAD_for_source_description and not object_type is None:
            plt.suptitle(status_message, y=0.05, x=0.15, ha="left")
        elif print_radio_to_optical_extent:
            plt.suptitle(extent_message, y=0.05, x=0.15, ha="left")
        elif print_FIRST_cat_message:
            plt.suptitle(FIRST_verbose_message, y=0.05, x=0.15, ha="left")

        plt.grid(False)
        if save:
            plt.savefig(os.path.join(save_dir, save_name), bbox_inches="tight")
        else:
            plt.show()
        plt.close()


def simple_plot_LoTSS_FIRST_and_PANSTARRS(
    ra,
    dec,
    size_in_arcsec,
    bin_path,
    cutout_id,
    version=2,
    save=False,
    temp_save_dir=None,
    save_dir=None,
    save_name=None,
    plot_reticle=True,
    **kwarg,
):
    """Just LoTSS and FIRST and PANSTARRS cutouts
    (No simbad lookup or scalebars)
    """

    # Create figure that will contain lotss, first and panstarrs
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
    plot_cutout_in_subplot(
        ax1,
        cutout_id,
        bin_path,
        title="LoTSS-DR2",
        version=version,
        figsize=3,
        plot_reticle=True,
    )
    plot_FIRST_or_PANSTARRS(
        ra,
        dec,
        size_in_arcsec,
        ax=ax2,
        filters="i",
        cmap="gray",
        figsize=3,
        plot_reticle=True,
        asinh_transform=True,
        plot_FIRST=False,
        plot_PANSTARRS=True,
        save_dir=temp_save_dir,
        title="PAN-STARRS filter=i",
        overwrite=False,
        **kwarg,
    )
    plot_FIRST_or_PANSTARRS(
        ra,
        dec,
        size_in_arcsec,
        ax=ax3,
        filters="i",
        cmap="viridis",
        figsize=3,
        plot_reticle=True,
        plot_FIRST=True,
        plot_PANSTARRS=False,
        title="FIRST",
        overwrite=False,
        save_dir=temp_save_dir,
        **kwarg,
    )

    plt.grid(False)
    if save:
        plt.savefig(os.path.join(save_dir, save_name), bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_LoTSS_FIRST_and_PANSTARRS(
    lotss_fullsize,
    cutout_ids,
    pandas_catalogue,
    bin_path,
    version=1,
    save=False,
    save_dir=None,
    save_index=None,
    query_SIMBAD_for_source_description=False,
    take_half_LGZ_Size_as_search_radius=False,
    object_search_radius_in_arcsec=10,
    print_radio_to_optical_extent=False,
    print_FIRST_cat_message=False,
    arcsec_per_pixel_lotss=1.5,
    plot_reticle=True,
):
    """Plot LoTSS and FIRST and PANSTARRS cutouts"""
    for cutout_id in cutout_ids:

        # Get source ra, dec and size
        ra, dec = (
            pandas_catalogue["RA"].iloc[cutout_id],
            pandas_catalogue["DEC"].iloc[cutout_id],
        )
        size_in_arcsec = int(
            np.ceil(pandas_catalogue["LGZ_Size"].iloc[cutout_id] * np.sqrt(2))
        )
        # Calculate rescale factor for the lotss scalebar
        rescale_factor = (lotss_fullsize * arcsec_per_pixel_lotss) / size_in_arcsec

        # Get 2MASX angular size
        if print_radio_to_optical_extent:
            (
                MASX_objectname,
                MASX_diam_arcsec,
                message,
            ) = get_2MASX_angular_diameter_arcsec(ra, dec)
            if MASX_diam_arcsec is None:
                extent_message = f"No 2MASX catalogue entry found."
            else:
                extent_message = (
                    f"Radio (LGZ_Size) to optical (2MASX) extent ratio is:"
                    f" {pandas_catalogue['LGZ_Size'].iloc[cutout_id]/MASX_diam_arcsec:.1f}"
                )

        # get the simbad object type for this ra and dec
        if take_half_LGZ_Size_as_search_radius:
            object_search_radius_in_arcsec = (
                pandas_catalogue["LGZ_Size"].iloc[cutout_id] / 2
            )
        if query_SIMBAD_for_source_description:
            object_type, object_types, maj_arcsec, status_message = query_SIMBAD(
                ra, dec, verbose=True
            )
        if print_FIRST_cat_message:
            (
                FIRST_source_name,
                FIRST_integrated_flux_in_mJy,
                FIRST_diameter_arcsec,
                FIRST_verbose_message,
            ) = check_for_FIRST_catalogue_entry(
                ra, dec, object_search_radius_in_arcsec=object_search_radius_in_arcsec
            )

        # Create figure that will contain lotss, first and panstarrs
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
        plot_cutout_in_subplot(
            ax1,
            cutout_id,
            bin_path,
            title="LoTSS DR1",
            version=version,
            figsize=3,
            plot_reticle=True,
            rescale_factor=rescale_factor,
        )
        plot_FIRST_or_PANSTARRS(
            ra,
            dec,
            size_in_arcsec,
            ax=ax2,
            filters="i",
            cmap="gray",
            figsize=3,
            plot_reticle=True,
            asinh_transform=True,
            plot_FIRST=False,
            plot_PANSTARRS=True,
            title="PAN-STARRS filter=i",
            overwrite=False,
        )
        plot_FIRST_or_PANSTARRS(
            ra,
            dec,
            size_in_arcsec,
            ax=ax3,
            filters="i",
            cmap="viridis",
            figsize=3,
            plot_reticle=True,
            plot_FIRST=True,
            plot_PANSTARRS=False,
            title="FIRST",
            overwrite=False,
        )
        if query_SIMBAD_for_source_description and not object_type is None:
            plt.suptitle(status_message, y=0.05, x=0.15, ha="left")
        elif print_radio_to_optical_extent:
            plt.suptitle(extent_message, y=0.05, x=0.15, ha="left")
        elif print_FIRST_cat_message:
            plt.suptitle(FIRST_verbose_message, y=0.05, x=0.15, ha="left")

        plt.grid(False)
        if save:
            if save_index is None:
                save_name = f"cutoutid{cutout_id}.png"
            else:
                save_name = f"nr{save_index:03d}_cutoutid{cutout_id}.png"
            plt.savefig(os.path.join(save_dir, save_name), bbox_inches="tight")
        else:
            plt.show()
        plt.close()


def plot_LoTSS_and_PANSTARRS(
    lotss_fullsize,
    cutout_ids,
    pandas_catalogue,
    bin_path,
    version=1,
    save=False,
    save_dir=None,
    save_index=None,
    query_SIMBAD_for_source_description=False,
    overwrite=False,
    print_radio_to_optical_extent=False,
    title_lotss=None,
    title_panstarrs=None,
    print_FIRST_cat_message=False,
    model_maj_min_angle_degree=False,
    arcsec_per_pixel_lotss=1.5,
    arcsec_per_pixel_PANSTARRS=0.25,
    plot_reticle=True,
):
    """Plot LoTSS and PANSTARRS cutouts"""
    for cutout_id in cutout_ids:

        # Get source ra, dec and size
        cutout = pandas_catalogue.iloc[cutout_id]
        # pandas_catalogue['RA'].iloc[cutout_id], pandas_catalogue['DEC'].iloc[cutout_id]
        ra, dec = cutout.RA, cutout.DEC
        size_in_arcsec = int(np.ceil(cutout.LGZ_Size * np.sqrt(2)))
        # Convert to PANSTARRS pixelsize
        size_PANSTARRS = int(size_in_arcsec / arcsec_per_pixel_PANSTARRS)
        # Calculate rescale factor for the lotss scalebar
        rescale_factor = (lotss_fullsize * arcsec_per_pixel_lotss) / size_in_arcsec
        if model_maj_min_angle_degree:
            model_maj_min_angle_degree = (
                cutout.LGZ_Size,
                cutout.LGZ_Width,
                cutout.LGZ_PA,
            )
            print(model_maj_min_angle_degree)
        else:
            model_maj_min_angle_degree = None

        # Get 2MASX angular size
        if print_radio_to_optical_extent:
            (
                MASX_objectname,
                MASX_diam_arcsec,
                message,
            ) = get_2MASX_angular_diameter_arcsec(ra, dec)
            if MASX_diam_arcsec is None:
                extent_message = f"No 2MASX catalogue entry found."
            else:
                extent_message = (
                    f"Radio (LGZ_Size) to optical (2MASX) extent ratio is:"
                    f" {pandas_catalogue['LGZ_Size'].iloc[cutout_id]/MASX_diam_arcsec:.1f}"
                )

        # get the simbad object type for this ra and dec
        if query_SIMBAD_for_source_description:
            object_type, object_types, maj_arcsec, status_message = query_SIMBAD(
                ra, dec, verbose=True
            )
        #    continue
        # Create figure that will contain both lotss and panstarrs
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
        if title_lotss is None:
            title_lotss = "LoTSS DR1"
        if title_panstarrs is None:
            title_panstarrs = "PAN-STARRS filter=i"
        plot_cutout_in_subplot(
            ax1,
            cutout_id,
            bin_path,
            title=title_lotss,
            version=version,
            figsize=3,
            plot_reticle=True,
            rescale_factor=rescale_factor,
            model_maj_min_angle_degree=model_maj_min_angle_degree,
        )
        plot_FIRST_or_PANSTARRS(
            ra,
            dec,
            size_in_arcsec,
            ax=ax2,
            filters="i",
            cmap="gray",
            figsize=3,
            plot_reticle=True,
            asinh_transform=True,
            plot_FIRST=False,
            plot_PANSTARRS=True,
            title=title_panstarrs,
            overwrite=False,
        )
        if query_SIMBAD_for_source_description and not object_type is None:
            plt.suptitle(status_message, y=0.05, x=0.15, ha="left")
        elif print_radio_to_optical_extent:
            plt.suptitle(extent_message, y=0.05, x=0.15, ha="left")
        elif print_FIRST_cat_message:
            plt.suptitle(FIRST_verbose_message, y=0.05, x=0.15, ha="left")
        plt.grid(False)
        if save:
            if save_index is None:
                save_name = f"cutoutid{cutout_id}.png"
            else:
                save_name = f"nr{save_index:03d}_cutoutid{cutout_id}.png"
            plt.savefig(os.path.join(save_dir, save_name), bbox_inches="tight")
        else:
            plt.show()
            plt.close()


def plot_prototype(
    som,
    x,
    y,
    z,
    figsize=3,
    compress=False,
    show=True,
    save=False,
    website=False,
    save_path=None,
    return_ax=False,
):
    # Plot the choosen prototype
    data_som = copy.deepcopy(som.data_som)

    if compress:
        i_compress, new_dimension = return_sparce_indices(som.som_width)
        data_som = data_som.reshape(
            som.som_width * som.som_height, data_som.shape[-2], data_som.shape[-1]
        )
        data_som = data_som[i_compress].reshape(
            new_dimension, new_dimension, data_som.shape[-2], data_som.shape[-1]
        )

    proto = data_som[x, y, z]
    if som.number_of_channels == 1:
        neuron_size = int(np.sqrt(proto.size))
        proto = proto.reshape((som.number_of_channels, neuron_size, neuron_size))
    else:
        neuron_size = som.fullsize
        proto = proto.reshape((som.number_of_channels, neuron_size, neuron_size))

    for c in range(som.number_of_channels):
        fig = plt.figure()
        fig.set_size_inches(figsize, figsize)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        if som.number_of_channels == 1:
            if len(proto.shape) == 3:
                proto = proto[0]
            ax.imshow(proto, aspect="equal", cmap="viridis", interpolation="nearest")
        else:
            ax.imshow(proto[c], aspect="equal", cmap="viridis", interpolation="nearest")
        if return_ax:
            return fig, ax
        if save:
            if website:
                plt.savefig(os.path.join(save_path, "prototype.png"))
            else:
                plt.savefig(os.path.join(save_path, f"prototype_{x}_{y}.png"))
        else:
            plt.show()
        plt.close()


def plot_cutouts_closest_to_prototype(
    som,
    x,
    y,
    z,
    max_number_of_images_to_show,
    data_map,
    bin_path,
    website=False,
    object_search_radius_in_arcsec=10,
    save_path=None,
    figsize=4,
    plot_border=True,
    apply_clipping=False,
    compress=False,
    save=True,
    include_worse_matches=False,
    output_coordinates_only=False,
    plot_PANSTARRS_too=False,
    get_SIMBAD_type=True,
    variable_size=False,
    print_peak_flux=False,
    print_rms=False,
    plot_cutout_index=True,
    pandas_catalogue=None,
    plot_histogram=False,
    version=1,
    arcsec_per_pixel_lotss=1.5,
    arcsec_per_pixel_PANSTARRS=0.25,
    clip_threshold=1.5,
    **kwargs,
):
    """Plot the max_number_of_images_to_show cutouts that best match the
    prototype with the given coordinates (x,y,z)"""

    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("Created save directory:", save_path)

    """
    if som.layout == 'hexagonal':
        data_som_index, _ = populate_hex_map(list(range(len(som.data_som))),
                som.som_width, som.som_height)        
        data_som, data_som_mask = populate_hex_map(som.data_som, som.som_width, som.som_height)

        if data_som_mask[x,y]:
            print('Prototype coordinates (x,y)=({},{}) are out of bound.'.format(x,
                y))
            sdfsdf
        proto = data_som[x,y]
        si = data_som_index[x,y]
    else:
        data_som = copy.deepcopy(som.data_som)
        if compress:
            i_compress, new_dimension = return_sparce_indices(som.som_width)
            data_som = data_som.reshape(som.som_width*som.som_height,data_som.shape[-2],data_som.shape[-1])
            data_som = data_som[i_compress].reshape(new_dimension,new_dimension,data_som.shape[-2],data_som.shape[-1])

        #proto = data_som[x,y,z]
        #si = (x*som.som_height)+y
        #si = (y*som.som_height)+x
    """

    # Plot the choosen prototype
    plot_prototype(
        som,
        x,
        y,
        z,
        figsize=3,
        compress=compress,
        website=website,
        save=save,
        save_path=save_path,
    )

    (
        closest_index,
        closest_distances,
        closest_resemblances,
    ) = return_cutouts_closest_to_prototype(
        som,
        x,
        y,
        z,
        data_map,
        compress=compress,
        include_worse_matches=include_worse_matches,
    )

    # Plot histogram of compared differences
    if plot_histogram:
        # Get the maj-size that contains 90% of the sources
        fig = plt.figure()
        fig.set_size_inches(10, 5)
        bins = plt.hist(closest_resemblances, bins=50)
        height = max(bins[0])
        summed_heatmaps = closest_resemblances
        # Plot red line for outliers shown
        # red_x = sorted_extremes[-number_of_outliers_to_show]
        # red_y = height*0.7
        # plt.text(red_x, red_y, str(number_of_outliers_to_show)+" biggest outliers shown below", color='r')
        # plt.vlines(red_x, ymax=red_y-0.02*height, ymin=0, color='r')
        # plt.arrow(red_x, red_y/2, 20, 0, shape='full', length_includes_head=True, head_width=height*0.05, head_length = 10, fc='r', ec='r')

        # Visualize the size-distribution
        hh = [height * 0.3, height * 0.2, height * 0.1]
        # for c, s, h in zip(cutoff, sections, hh):
        #    plt.vlines(c, ymax=h*0.95, ymin=0)
        #    print('Cut-off that includes {0}% of the sources: {1} (= {2} x median)'.format(s,
        #        round(c,1), round(c/np.median(summed_heatmaps),1)))
        #    plt.text(c, h, str(s*100)+'% of sources')
        #    plt.arrow(c, h*0.5, -20, 0, shape='full', length_includes_head=True, head_width=height*0.05, head_length = 10, fc='k', ec='k')

        # info_x = max(summed_heatmaps)*0.65
        plt.text(
            min(summed_heatmaps) * 1.1,
            height * 0.25,
            """Median: {}
Mean: {}
Std. dev.: {} 
""".format(
                str(round(np.median(summed_heatmaps), 3)),
                str(round(np.mean(summed_heatmaps), 3)),
                str(round(np.std(summed_heatmaps), 3)),
                str(round(max(summed_heatmaps), 3)),
            ),
        )

        # plt.xlim([min(summed_heatmaps),1])
        plt.yscale("log")
        plt.xscale("log")
        plt.title("Histogram of distance to closest prototype")
        plt.xlabel("Euclidian norm to closest prototype")
        plt.ylabel("Number of radio-sources per bin")
        plt.tight_layout()
        plt.show()
        plt.close()
    # Plot cut-outs (closest to choosen prototype)
    # Size needed for cut-out to be able to rotate it

    result_strings = []
    rms_value, peak_flux_value = None, None
    for i, cutout_id in enumerate(closest_index[:max_number_of_images_to_show]):

        # Debugprint
        print(
            "RA DEC {} {}".format(
                pandas_catalogue["RA"].iloc[cutout_id],
                pandas_catalogue["DEC"].iloc[cutout_id],
            )
        )

        if output_coordinates_only:
            if len(closest_resemblances) > i:
                result_string = "{} {} Relative resemblance: {}".format(
                    pandas_catalogue["RA"].iloc[cutout_id],
                    pandas_catalogue["DEC"].iloc[cutout_id],
                    closest_resemblances[i],
                )
            else:
                result_string = "{} {}".format(
                    pandas_catalogue["RA"].iloc[cutout_id],
                    pandas_catalogue["DEC"].iloc[cutout_id],
                )
            result_strings.append(result_string)
            print(result_string)
        else:
            if plot_PANSTARRS_too:
                plot_LoTSS_and_PANSTARRS(
                    som.fullsize,
                    [cutout_id],
                    pandas_catalogue,
                    bin_path,
                    version=version,
                    save=save,
                    save_index=i,
                    save_dir=save_path,
                    arcsec_per_pixel_lotss=1.5,
                    arcsec_per_pixel_PANSTARRS=0.25,
                    **kwargs,
                )

            else:
                # Only plot LoTSS
                if print_rms:
                    rms_value = pandas_catalogue.cutout_rms.iloc[cutout_id] * 1000
                else:
                    rms_value = None
                if print_peak_flux:
                    peak_flux_value = pandas_catalogue.Peak_flux.iloc[cutout_id]
                    total_flux_value = pandas_catalogue.Total_flux.iloc[cutout_id]
                else:
                    peak_flux_value = None
                    total_flux_value = None
                plot_cutout(
                    cutout_id,
                    bin_path,
                    save_path,
                    str(i),
                    som.rotated_size,
                    figsize=figsize,
                    save=save,
                    plot_border=plot_border,
                    rms_value=rms_value,
                    peak_flux_value=peak_flux_value,
                    clip_threshold=clip_threshold,
                    total_flux_value=total_flux_value,
                    plot_cutout_index=plot_cutout_index,
                    version=version,
                )
    return (
        closest_index[:max_number_of_images_to_show],
        closest_resemblances[:max_number_of_images_to_show],
        result_strings,
    )


def plot_cutout(
    cutout_id,
    bin_path,
    save_path,
    title,
    rotated_size,
    figsize=4,
    save=True,
    plot_border=False,
    plot_second_border=False,
    plot_cutout_index=False,
    apply_clipping=False,
    cmap="viridis",
    rms_value=None,
    peak_flux_value=None,
    return_fig_only=False,
    total_flux_value=None,
    plot_text=None,
    channel_recursion=-9,
    clip_threshold=3,
    version=1,
):
    """Plot cutout to screen or save in save_dir"""

    image = return_cutout(bin_path, cutout_id, version=version)
    # Check if returned cutout has multiple channels
    if len(np.shape(image)) == 3:
        # If so, recursively call this function
        number_of_channels = np.shape(image)[0]
        if channel_recursion < 0:
            channel_recursion = number_of_channels - 1
        else:
            channel_recursion -= 1
        if channel_recursion > 0:
            plot_cutout(
                cutout_id,
                bin_path,
                save_path,
                title,
                rotated_size,
                figsize=figsize,
                save=save,
                plot_border=plot_border,
                plot_second_border=plot_second_border,
                plot_cutout_index=plot_cutout_index,
                apply_clipping=apply_clipping,
                cmap=cmap,
                rms_value=rms_value,
                peak_flux_value=peak_flux_value,
                total_flux_value=total_flux_value,
                plot_text=plot_text,
                channel_recursion=channel_recursion,
                clip_threshold=clip_threshold,
                version=version,
            )
        image = image[channel_recursion]

    fig = plt.figure(figsize=(figsize, figsize))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    if apply_clipping:
        image_clip = np.clip(image, clip_threshold * np.std(image), 1e10)
        image = np.hstack((image, image_clip))

    ax.imshow(image, aspect="equal", cmap=cmap, interpolation="nearest", origin="lower")

    if plot_border:
        # s = rotated_size
        # w = int(np.ceil(s*np.sqrt(2)))
        w = np.shape(image)[0]
        s = int(w / np.sqrt(2))
        # plot border around the 128x128 center of the image
        ax.plot([(w - s) / 2, w - (w - s) / 2], [(w - s) / 2, (w - s) / 2], "r")
        ax.plot([(w - s) / 2, (w - s) / 2], [(w - s) / 2, w - (w - s) / 2], "r")
        ax.plot([w - (w - s) / 2, w - (w - s) / 2], [(w - s) / 2, w - (w - s) / 2], "r")
        ax.plot([(w - s) / 2, w - (w - s) / 2], [w - (w - s) / 2, w - (w - s) / 2], "r")
    if plot_second_border:
        w = np.shape(image)[0]
        s = int(w / np.sqrt(2))
        s = (w + s) / 2
        color = "orange"
        linestyle = "dashed"
        ax.plot(
            [(w - s) / 2, w - (w - s) / 2],
            [(w - s) / 2, (w - s) / 2],
            color=color,
            linestyle=linestyle,
        )
        ax.plot(
            [(w - s) / 2, (w - s) / 2],
            [(w - s) / 2, w - (w - s) / 2],
            color=color,
            linestyle=linestyle,
        )
        ax.plot(
            [w - (w - s) / 2, w - (w - s) / 2],
            [(w - s) / 2, w - (w - s) / 2],
            color=color,
            linestyle=linestyle,
        )
        ax.plot(
            [(w - s) / 2, w - (w - s) / 2],
            [w - (w - s) / 2, w - (w - s) / 2],
            color=color,
            linestyle=linestyle,
        )

    if plot_cutout_index:
        s = rotated_size
        w = int(np.ceil(s * np.sqrt(2)))
        # plot index in lower left corner
        ax.text(w - 2, 2, str(cutout_id), color="white", horizontalalignment="right")
    text_size = 14
    value_size = 16
    if not peak_flux_value is None and not total_flux_value is None:
        ax.text(
            0.01,
            0.13,
            f"p. flux",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            color="lightgray",
            fontsize=text_size,
        )
        ax.text(
            0.17,
            0.13,
            f"{peak_flux_value:.2g}",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            color="w",
            fontsize=value_size,
        )
        ax.text(
            0.01,
            0.19,
            f"t. flux",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            color="lightgray",
            fontsize=text_size,
        )
        ax.text(
            0.17,
            0.19,
            f"{total_flux_value:.2g}",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            color="w",
            fontsize=value_size,
        )
    if not plot_text is None:
        ax.text(
            0.01,
            0.07,
            plot_text,
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            color="w",
            fontsize=text_size,
        )
    if not rms_value is None:
        ax.text(
            0.01,
            0.07,
            f"rms",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            color="lightgray",
            fontsize=text_size,
        )
        ax.text(
            0.12,
            0.07,
            f"{rms_value:.2g}",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            color="w",
            fontsize=value_size,
        )
    if not peak_flux_value is None or not rms_value is None:
        ax.text(
            0.01,
            0.01,
            f"[mJy(/beam)]",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            color="lightgray",
            fontsize=text_size,
        )

    if return_fig_only:
        return fig
    if save:
        plt.savefig(os.path.join(save_path, title + ".jpg"))
    else:
        plt.show()
    plt.close()


def return_tight_box_around_source(maj_degree, min_degree, angle_degree):
    """Return RA and DEC offset in (both in and output in degree)
    to get the tightest rectangle around a source."""

    ddec = max(
        abs(np.cos(np.deg2rad(angle_degree)) * maj_degree / 2),
        abs(np.cos(np.deg2rad(angle_degree - 90)) * min_degree / 2),
    )
    dra = max(
        abs(np.sin(np.deg2rad(angle_degree)) * maj_degree / 2),
        abs(np.sin(np.deg2rad(angle_degree - 90)) * min_degree / 2),
    )
    return dra, ddec


def plot_cutout_in_subplot(
    ax,
    cutout_id,
    bin_path,
    figsize=3,
    rescale_factor=1,
    model_maj_min_angle_degree=None,
    arcsec_per_pixel=1.5,
    apply_clipping=False,
    plot_reticle=False,
    title=None,
    clip_threshold=3,
    version=1,
):
    """Plot cutout on given axis"""

    if ax is None:
        ax = plt.gca()
    if not title is None:
        ax.title.set_text(title)
    image = return_cutout(bin_path, cutout_id, version=version)
    if apply_clipping:
        image_clip = np.clip(image, clip_threshold * np.std(image), 1e10)
        image = np.hstack((image, image_clip))
    ax.axis("off")
    bp = ax.imshow(image, aspect="equal", interpolation="nearest", origin="lower")

    # Plot box around LGZ_size*LGZ_width to visually check the box
    if not model_maj_min_angle_degree is None:
        maja, mina, angle_degree = model_maj_min_angle_degree
        maja = maja / arcsec_per_pixel * rescale_factor
        mina = mina / arcsec_per_pixel * rescale_factor

        ddec = max(
            abs(np.cos(np.deg2rad(angle_degree)) * maja / 2),
            abs(np.cos(np.deg2rad(angle_degree - 90)) * mina / 2),
        )
        dra = max(
            abs(np.sin(np.deg2rad(angle_degree)) * maja / 2),
            abs(np.sin(np.deg2rad(angle_degree - 90)) * mina / 2),
        )
        ax.add_patch(
            Rectangle(
                (image.shape[0] / 2 - dra, image.shape[0] / 2 - ddec),
                dra * 2,
                ddec * 2,
                linewidth=1,
                edgecolor="g",
                facecolor="none",
            )
        )

        print(maja, mina, angle_degree)
        ax.add_patch(
            Ellipse(
                (image.shape[0] / 2, image.shape[0] / 2),
                mina,
                maja,
                angle=angle_degree,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )

    if plot_reticle:
        x_offset, y_offset = 10, 10
        # 1.5arcsec_per_pixel_resolution
        scale_bar_length = (10 / 1.5) * rescale_factor
        scale_bar_color = "white"
        ax.plot(
            [x_offset, x_offset + scale_bar_length],
            [y_offset, y_offset],
            "-",
            color=scale_bar_color,
            linewidth=3,
        )
        ax.plot(np.shape(image)[0] / 2, np.shape(image)[1] / 2, "r+")
        bp = ax.text(x_offset, y_offset + 5, '10"', color=scale_bar_color, fontsize=18)

    return bp


def save_all_prototypes_and_cutouts(
    som,
    bin_path,
    website_path,
    max_number_of_images_to_show,
    catalogue,
    web_catalogue,
    data_map,
    highlight_cutouts=None,
    highlight_cutouts2=None,
    correction_contribution=0.033,
    figsize=5,
    save=True,
    plot_border=True,
    version=1,
    print_peak_flux=True,
    print_rms=True,
    peak_flux_value=None,
    total_flux_value=None,
    resemblance_division_line=None,
    plot_cutout_index=False,
    apply_clipping=False,
    clip_threshold=3,
    overwrite=False,
):
    if som.layout == "hexagonal":
        print("Hexagonal layout is not implemented yet.")
        return
    for x in range(som.som_width):
        for y in range(som.som_height):
            print(x, y)
            for z in range(som.som_depth):
                # Make image dir for prototype and website paths
                proto_path = os.path.join(
                    website_path, "prototype" + str(x) + "_" + str(y) + "_" + str(z)
                )
                image_paths = [
                    os.path.exists(os.path.join(proto_path, str(i) + ".jpg"))
                    for i in range(max_number_of_images_to_show)
                ]
                text_path_exists = os.path.exists(os.path.join(proto_path, "loc.txt"))
                os.makedirs(proto_path, exist_ok=True)

                if (
                    not os.path.exists(os.path.join(proto_path, "prototype.jpg"))
                    or overwrite
                ):
                    # Plot the choosen prototype
                    proto = som.data_som[x, y, z, :]
                    proto_size = int(np.sqrt(len(proto)))
                    proto = proto.reshape((proto_size, proto_size))
                    fig = plt.figure()
                    fig.set_size_inches(6, 6)
                    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(
                        proto,
                        aspect="equal",
                        interpolation="nearest",
                        origin="lower",
                        cmap="viridis",
                    )
                    plt.savefig(proto_path + "/prototype.jpg")
                    plt.close()

                adapted_outlier_score = False
                if not resemblance_division_line is None:
                    adapted_outlier_score = True

                if not text_path_exists or not all(image_paths) or overwrite:
                    # Sort the euclidian distances to this prototype
                    (
                        closest_index,
                        closest_distances,
                        closest_resemblances,
                    ) = return_cutouts_closest_to_prototype(
                        som,
                        x,
                        y,
                        z,
                        data_map,
                        include_worse_matches=False,
                        adapted_outlier_score=adapted_outlier_score,
                        correction_contribution=correction_contribution,
                    )

                    if not resemblance_division_line is None:
                        # Get indexnumber at which the resemblance division line is crossed
                        resemblance_division_index = 0
                        for ci, cr in enumerate(closest_resemblances):
                            if cr > resemblance_division_line:
                                resemblance_division_index = ci
                                # If crossed, write it to file
                                with open(
                                    os.path.join(proto_path, "division_index.txt"), "w"
                                ) as f:
                                    f.write(f"{resemblance_division_index}")
                                break

                rms_value = None
                if not all(image_paths) or overwrite:
                    # Plot cut-outs (closest to choosen prototype)
                    plot_text = ""
                    for i, cutout_id in enumerate(
                        closest_index[:max_number_of_images_to_show]
                    ):
                        if print_rms:
                            rms_value = catalogue.cutout_rms.iloc[cutout_id] * 1000
                        if print_peak_flux:
                            peak_flux_value = catalogue.Peak_flux.iloc[cutout_id]
                            total_flux_value = catalogue.Total_flux.iloc[cutout_id]
                        if not resemblance_division_line is None:
                            plot_text = f"{closest_resemblances[i]:.3g}"
                        if not highlight_cutouts is None:
                            if cutout_id in highlight_cutouts:
                                plot_border = True
                            else:
                                plot_border = False
                        plot_second_border = False
                        if not highlight_cutouts2 is None:
                            if cutout_id in highlight_cutouts2:
                                plot_second_border = True
                            else:
                                plot_second_border = False
                        plot_cutout(
                            cutout_id,
                            bin_path,
                            proto_path,
                            str(i),
                            som.rotated_size,
                            figsize=figsize,
                            save=save,
                            plot_border=plot_border,
                            plot_second_border=plot_second_border,
                            peak_flux_value=peak_flux_value,
                            total_flux_value=total_flux_value,
                            rms_value=rms_value,
                            plot_cutout_index=plot_cutout_index,
                            apply_clipping=apply_clipping,
                            plot_text=plot_text,
                            clip_threshold=clip_threshold,
                            version=version,
                        )

                # Write locations to text files for website
                if not text_path_exists or overwrite:
                    ra = catalogue["RA"].iloc[
                        closest_index[:max_number_of_images_to_show]
                    ]
                    dec = catalogue["DEC"].iloc[
                        closest_index[:max_number_of_images_to_show]
                    ]
                    with open(os.path.join(proto_path, "loc.txt"), "w") as loc_f:
                        for a, b in zip(ra, dec):
                            loc_f.write("{};{}\n".format(a, b))

                # Write catalogue lines to csv files
                proto_path = os.path.join(
                    website_path, "prototype" + str(x) + "_" + str(y) + "_" + str(z)
                )
                if not os.path.exists(proto_path):
                    os.makedirs(proto_path)
                csv_name = (
                    som.trained_subdirectory
                    + "_prototype_"
                    + str(x)
                    + "_"
                    + str(y)
                    + ".csv"
                )
                csv_path_exists = os.path.exists(os.path.join(proto_path, csv_name))
                if not csv_path_exists or overwrite:
                    print(proto_path)
                    condition = (web_catalogue["Closest_prototype_x"] == x) & (
                        web_catalogue["Closest_prototype_y"] == y
                    )
                    pd_to_csv = web_catalogue[condition]
                    pd_to_csv.to_csv(os.path.join(proto_path, csv_name), index=False)
    print("Done.")


def return_cutouts_not_matching_prototypes(som, best_matching_prototypes, catalogue):
    """When given a som and corresponding corrected pandas catalogue, returns catalogue
    subset that only contains entries that do not match best_matching_prototypes"""
    assert som.layout == "quadratic"
    catalogue["corrected_id"] = range(len(catalogue))
    bmu_ids = [(y * som.som_width) + x for y, x in best_matching_prototypes]
    subset = catalogue[~catalogue["Closest_prototype"].isin(bmu_ids)]
    print(
        "New training set contains {} cutouts, {} fewer than the original training"
        " set.".format(len(subset), len(catalogue) - len(subset))
    )
    return subset


def return_cutouts_matching_prototypes(
    som, best_matching_prototypes, catalogue, verbose=True
):
    """When given a som and corresponding corrected pandas catalogue, returns catalogue
    subset that only contains entries that match the best_matching_prototypes"""
    assert som.layout == "quadratic"
    catalogue["corrected_id"] = range(len(catalogue))
    bmu_ids = [(y * som.som_width) + x for y, x in best_matching_prototypes]
    subset = catalogue[catalogue["Closest_prototype"].isin(bmu_ids)]
    if verbose:
        print(
            "New training set contains {} cutouts, {} fewer than the original training"
            " set.".format(len(subset), len(catalogue) - len(subset))
        )
    return subset


def return_cutouts_matching_weighted_prototypes_count(
    som, data_map, best_matching_prototypes, best_matching_occurence_count
):
    """NOTE: Paper speficif function, might not be reusable.
    When given a som and corresponding corrected pandas catalogue, returns counts of
    entries that match the best_matching_prototypes, weighted by the occurence of the
    prototype (best_matching_occurence_count which should be a Counter dictionary)"""
    count = 0
    for bmu in best_matching_prototypes:
        bmo_count = best_matching_occurence_count[tuple(bmu)]
        indexes, _, _ = return_cutouts_closest_to_prototype(
            som, bmu[0], bmu[1], 0, data_map, include_worse_matches=False
        )
        count += len(indexes) / bmo_count

    return count


# @profile
def write_numpy_to_binary_v2(
    bin_output_path, data, layout, verbose=True, overwrite=False, data_type=0
):
    """Write numpy array to binary file v2,
    binary file format version 2
    file_type: 0 is data file, 1 is trained SOM, 2 is mapping, 3 is rotations
    data_type: 0 is float 32 (only supported type currently)
    """
    version = 2
    file_type = 0

    if layout == "quadratic" or layout == "cartesian":
        layout = 0
    elif layout == "hexagonal":
        layout = 1
    else:
        raise Exception("Data layout not recognized")

    if overwrite or not os.path.exists(bin_output_path):
        with open(bin_output_path, "wb") as file:
            n_channels = 1

            if verbose:
                print("cutouts shape", np.shape(data))
                print("numberOfImages", np.shape(data)[0])

            # For 2D image
            if len(data.shape) == 3:
                if verbose:
                    print("width", np.shape(data)[1])
                    print("height", np.shape(data)[2])
                dimensionality = 2
                image_size = np.shape(data)[1]
            # For more than 2D image
            elif len(data.shape) == 4:
                n_channels = np.shape(data)[1]
                if verbose:
                    print("width", np.shape(data)[2])
                    print("height", np.shape(data)[3])
                    print("number of channels", n_channels)
                dimensionality = 3
                image_size = np.shape(data)[2]
            elif len(data.shape) == 2:
                # asumme that these cutouts are not reshaped
                dimensionality = 2
                image_size = int(np.sqrt(np.shape(data)[1]))
            else:
                raise NotImplementedError

            # <file format version> 0 <data-type> <number of entries> <data layout> <data>
            file.write(
                struct.pack(
                    "i" * 6,
                    version,
                    file_type,
                    data_type,
                    np.shape(data)[0],
                    layout,
                    dimensionality,
                )
            )
            if len(data.shape) == 4:
                file.write(struct.pack("i", n_channels))
            file.write(struct.pack("i", image_size))
            file.write(struct.pack("i", image_size))

            # file.write(struct.pack('f' * data.size, *data.flatten()))
            # print("debugprint:", np.shape(data), data[0])
            for image in data:
                image.astype("f").tofile(file)

            if verbose:
                print(
                    f"version {version}; file_type {file_type}; data_type {data_type};",
                    f"number of entries/images {np.shape(data)[0]}; layout {layout}; dimensionality",
                    f" {dimensionality}; image size {image_size}x{image_size}; number_of_channels"
                    f" {n_channels}",
                )
        if not verbose:
            del data

        # Check binary filesize
        binsize = os.path.getsize(bin_output_path)
        if verbose:
            print(
                "Binary file size: {} MB. Expected file size {} MB".format(
                    round(binsize / 1024.0 / 1024.0, 2),
                    round(
                        (9 * 4 + 4.0 * n_channels * image_size**2 * len(data))
                        / 1024.0
                        / 1024.0,
                        2,
                    ),
                )
            )

    else:
        if verbose:
            print(
                "Binary file with the name '{0}' already exists".format(bin_output_path)
            )


def merge_channels(*args):
    """Requires multiple lists of cutouts as input and returns the merged result.
    For example given the np.array cutouts_channel1_radio and cutouts_channel2_optical,
    one should call merge_channels(cutouts_channel1_radio,cutouts_channel2_optical)"""
    assert len(args) > 1, "You provided a single input. I need more."
    return np.array([list(c) for c in zip(*args)])


def unpack_data_binary(
    bin_path, file_type=0, data_type=0, som_layout=0, verbose=True, channels=1
):
    """Unpack a given version 2 data binary"""
    cutouts = []
    assert file_type == 0

    with open(bin_path, "rb") as input:
        if channels == 1:
            # 2, 0, 0, nb_images, 0, 2, width, height)
            header = struct.unpack("i" * 8, input.read(4 * 8))
        else:
            # 2, 0, 0, nb_images, 0, 3, width, height, nb_channels)
            header = struct.unpack("i" * 9, input.read(4 * 9))

        # nb_images, nb_channels, width, height = struct.unpack('i' * 4, input.read(4 * 4))
        size = channels * header[6] * header[7]  # width * height
        if verbose:
            print(
                "Number of images, channels, width, height:",
                header[3],
                channels,
                header[6],
                header[7],
            )

        # <file format version> 0 <data-type> <number of entries> <data layout> <data>

        for _ in range(header[3]):
            cutouts.append(input.read(size * 4))
    return header, cutouts


def binary_v1_to_binary_v2(
    bin_v1_path, bin_v2_path, file_type=1, data_type=0, som_layout=0, verbose=True
):
    if file_type == 0:
        with open(bin_v1_path, "rb") as input, open(bin_v2_path, "wb") as output:

            nb_images, nb_channels, width, height = struct.unpack(
                "i" * 4, input.read(4 * 4)
            )
            size = nb_channels * width * height

            # <file format version> 0 <data-type> <number of entries> <data layout> <data>

            if nb_channels == 1:
                output.write(
                    struct.pack("i" * 8, 2, 0, 0, nb_images, 0, 2, width, height)
                )
            else:
                output.write(
                    struct.pack(
                        "i" * 9, 2, 0, 0, nb_images, 0, 3, width, height, nb_channels
                    )
                )

            for _ in range(nb_images):
                output.write(input.read(size * 4))
    elif file_type == 1:
        with open(bin_v1_path, "rb") as input, open(bin_v2_path, "wb") as output:
            # SOM file
            # <file format version> 1 <data-type> <som layout> <neuron layout> <data>
            (
                number_of_channels,
                som_width,
                som_height,
                som_depth,
                neuron_width,
                neuron_height,
            ) = struct.unpack("i" * 6, input.read(4 * 6))
            size = number_of_channels * neuron_width * neuron_height
            nb_neurons = som_width * som_height * som_depth

            version = 2  # binary file format version 2
            # file_type 0 is data file, 1 is trained SOM, 2 is mapping, 3 is rotations
            data_type = 0  # 0 is float 32 (only supported type currently)
            # width and height, with depth it could be 3 but not used by me yet
            som_dimensionality = 2
            neuron_layout = 0  # 0 is cartesian, 1 is hexagonal
            if number_of_channels == 1:
                neuron_dimensionality = 2  # width and height, with depth it could be 3
            else:
                raise NotImplementedError

            output.write(
                struct.pack(
                    "i" * 11,
                    version,
                    file_type,
                    data_type,
                    som_layout,
                    som_dimensionality,
                    som_width,
                    som_height,
                    neuron_layout,
                    neuron_dimensionality,
                    neuron_width,
                    neuron_height,
                )
            )

            # Write SOM neurons
            for _ in range(nb_neurons):
                output.write(input.read(size * 4))

    else:
        raise NotImplementedError


def write_som_to_data_binary(
    som, bin_output_path, verbose=False, overwrite=False, version=1
):
    if overwrite or not os.path.exists(bin_output_path):
        if version == 1:

            if verbose:
                print("Write cutouts to binary format at location:", bin_output_path)
            # Open binary file
            with open(bin_output_path, "wb") as output:
                # number of objectes
                output.write(struct.pack("i", som.som_width * som.som_height))
                # number of channels
                output.write(struct.pack("i", som.number_of_channels))
                output.write(struct.pack("i", som.rotated_size))  # input width
                # input height
                output.write(struct.pack("i", som.rotated_size))

                for image in som.data_som.reshape(
                    som.som_width * som.som_height, np.shape(som.data_som)[-1]
                ):
                    image.astype("f").tofile(output)
        elif version == 2:
            layout = som.layout
            version = 2  # binary file format version 2
            file_type = (
                0  # is data file, 1 is trained SOM, 2 is mapping, 3 is rotations
            )
            data_type = 0  # 0 is float 32 (only supported type currently)
            if som.number_of_channels > 1:
                dimensionality = 3
            else:
                dimensionality = 2

            if layout == "quadratic":
                layout = 0
            elif layout == "hexagonal":
                layout = 1
            else:
                raise Exception("Data layout not recognized")

            with open(bin_output_path, "wb") as file:
                data = som.data_som

                if verbose:
                    print("shape of data", np.shape(data))
                    print(
                        "numberOfImages (or prototypes)", som.som_width * som.som_height
                    )
                    # For 2D image
                    if len(data.shape) == 3:
                        print("width", np.shape(data)[1])
                        print("height", np.shape(data)[2])
                    # For more than 2D image
                    elif len(data.shape) == 4:
                        print("numberOfChannels", som.number_of_channels)
                        print("width", np.shape(data)[0])
                        print("height", np.shape(data)[1])

                # neuron_size = int(np.ceil(som.fullsize/np.sqrt(2)*2))
                neuron_size = int(np.sqrt(np.shape(data)[-1] / som.number_of_channels))
                print("Neuron_size:", neuron_size)
                # <file format version> 0 <data-type> <number of entries> <data layout> <data>
                file.write(
                    struct.pack(
                        "i" * 6,
                        version,
                        file_type,
                        data_type,
                        som.som_width * som.som_height,
                        layout,
                        dimensionality,
                    )
                )
                if som.number_of_channels > 1:
                    file.write(struct.pack("i", som.number_of_channels))
                file.write(struct.pack("i", neuron_size))
                file.write(struct.pack("i", neuron_size))
                # file.write(struct.pack('f' * data.size, *data.flatten()))

                for image in data:
                    image.astype("f").tofile(file)

    else:
        if verbose:
            print(
                "Binary file with the name '{0}' already exists".format(bin_output_path)
            )
    # Check binary filesize
    binsize = os.path.getsize(bin_output_path)
    if verbose:
        if version == 1:
            print(
                "Binary file size: {} MB. Expected file size {} MB".format(
                    round(binsize / 1024.0 / 1024.0, 2),
                    round(
                        (
                            4
                            + 4.0
                            * som.rotated_size
                            * som.rotated_size
                            * (som.som_width * som.som_height)
                        )
                        / 1024.0
                        / 1024.0,
                        2,
                    ),
                )
            )
        if version == 2:
            print(
                "Binary file size: {} MB. Expected file size {} MB".format(
                    round(binsize / 1024.0 / 1024.0, 2),
                    round(
                        (
                            6
                            + 4.0
                            * neuron_size
                            * neuron_size
                            * (som.som_width * som.som_height)
                        )
                        / 1024.0
                        / 1024.0,
                        2,
                    ),
                )
            )


def write_numpy_to_binary(
    som, data, bin_output_path, verbose=False, overwrite=False, version=1
):
    if overwrite or not os.path.exists(bin_output_path):
        if version == 1:

            if verbose:
                print("Write cutouts to binary format at location:", bin_output_path)
            # Open binary file
            with open(bin_output_path, "wb") as output:
                output.write(struct.pack("i", len(data)))  # number of objectes
                # number of channels
                output.write(struct.pack("i", som.number_of_channels))
                output.write(struct.pack("i", som.fullsize))  # input width
                output.write(struct.pack("i", som.fullsize))  # input height
                print("num objects:", len(data))
                print("num channels:", som.number_of_channels)
                print("cutout full dim:", som.fullsize)

                for image in data:
                    image.astype("f").tofile(output)
                    # output.write(struct.pack('f' * data.size, *data.flatten()))

        elif version == 2:
            version = 2  # binary file format version 2
            file_type = (
                0  # is data file, 1 is trained SOM, 2 is mapping, 3 is rotations
            )
            data_type = 0  # 0 is float 32 (only supported type currently)

            if som.layout == "quadratic":
                layout = 0
            elif som.layout == "hexagonal":
                layout = 1
            else:
                raise Exception("Data layout not recognized")

            with open(bin_output_path, "wb") as file:

                print("shape", np.shape(data))
                print("numberOfImages", np.shape(data)[0])
                # For 2D image
                if len(data.shape) == 3:
                    print("width", np.shape(data)[1])
                    print("height", np.shape(data)[2])
                    width = np.shape(data)[1]
                    dimensionality = len(np.shape(data)) - 1
                # For more than 2D image
                elif len(data.shape) == 4:
                    print("numberOfChannels", np.shape(data)[1])
                    print("width", np.shape(data)[2])
                    print("height", np.shape(data)[3])
                    width = np.shape(data)[2]
                    dimensionality = len(np.shape(data)) - 1

                if len(data.shape) == 2:
                    # In this case each cutout is a long 1D vector which we suppose should be 2d
                    width = int(np.sqrt(np.shape(data)[1]))
                    print("width", width)
                    dimensionality = 2
                file.write(
                    struct.pack(
                        "i" * 6,
                        version,
                        file_type,
                        data_type,
                        np.shape(data)[0],
                        layout,
                        dimensionality,
                    )
                )
                file.write(struct.pack("i", width))
                file.write(struct.pack("i", width))
                if len(data.shape) == 4:
                    file.write(struct.pack("i", np.shape(data)[1]))
                # file.write(struct.pack('f' * data.size, *data.flatten()))

                for image in data:
                    image.astype("f").tofile(file)

    else:
        if verbose:
            print(
                "Binary file with the name '{0}' already exists".format(bin_output_path)
            )
    # Check binary filesize
    binsize = os.path.getsize(bin_output_path)
    if verbose:
        print(
            "Binary file size: {} MB. Expected file size {} MB".format(
                round(binsize / 1024.0 / 1024.0, 2),
                round(
                    (12 + 4.0 * som.fullsize * som.fullsize * len(data))
                    / 1024.0
                    / 1024.0,
                    2,
                ),
            )
        )


def write_numpy_to_SOM_file(
    som,
    image_list,
    bin_output_path,
    overwrite=False,
    version=1,
    explicit_dimensions=None,
):
    if overwrite or not os.path.exists(bin_output_path):
        assert version == 1 or version == 2
        # Open binary file
        with open(bin_output_path, "wb") as output:
            if version == 1:
                # print(f'{som.number_of_channels=} {len(image_list)=} {som.som_depth=}')

                # number of channels
                output.write(struct.pack("i", som.number_of_channels))
                output.write(struct.pack("i", len(image_list)))  # SOM width
                output.write(struct.pack("i", 1))  # SOM height
                output.write(struct.pack("i", som.som_depth))  # SOM depth
                # Neuron width
                output.write(struct.pack("i", som.rotated_size))
                # Neuron height
                output.write(struct.pack("i", som.rotated_size))
                for image in image_list:
                    image.astype("f").tofile(output)
            if version == 2:
                print("Only works for single channel neurons.")
                # <file format version> 1 <data-type> <som layout> <neuron layout> <data>
                if explicit_dimensions is None:

                    output.write(
                        struct.pack(
                            "i" * 11,
                            2,  # <file format version>
                            1,  # <file type>
                            0,  # <data type>
                            0,  # <som layout>
                            2,
                            som.som_width,
                            som.som_height,  # <som dimensionality, width and height>
                            0,  # <neuron layout>
                            2,
                            np.shape(image_list)[1],
                            np.shape(image_list)[2],
                        )
                    )  # <neuron dimensionality and width>
                else:
                    output.write(
                        struct.pack(
                            "i" * 11,
                            2,  # <file format version>
                            1,  # <file type>
                            0,  # <data type>
                            0,  # <som layout>
                            2,
                            som.som_width,
                            som.som_height,  # <som dimensionality, width and height>
                            0,  # <neuron layout>
                            *explicit_dimensions,
                        )
                    )  # <neuron dimensionality and width>

                output.write(struct.pack("f" * image_list.size, *image_list.flatten()))

    else:
        print("Binary file with the name '{0}' already exists".format(bin_output_path))
    # Check binary filesize
    binsize = os.path.getsize(bin_output_path)
    print(
        "Binary file size: {} MB. Expected file size {} MB".format(
            round(binsize / 1024.0 / 1024.0, 2),
            round(
                (12 + 4.0 * som.rotated_size * som.rotated_size * len(image_list))
                / 1024.0
                / 1024.0,
                2,
            ),
        )
    )


def fits_to_hips(fits_path, hips_path, hipsgen_path, overwrite=False):
    """Convert fits file to hips"""
    if not os.path.exists(fits_path):
        print("Fits file not found at: {}".format(fits_path))
        return
    if not os.path.exists(hipsgen_path):
        print("Hipsgen.jar file not found at: {}".format(hipsgen_path))
        print(
            "Change the path to point at the Hipsgen.jar file or download the package at http://aladin.unistra.fr/java/Hipsgen.jar"
        )
        return

    if overwrite or not os.path.exists(hips_path):
        print("Creating Hips file. This might take a few minutes.")
        warnings.warn(
            "Errors during this process will not appear in this notebook, but will be written to the shell"
        )
        # Command is of the form: java -jar Hipsgen.jar in=<infile> out=<outfile>
        subprocess.call(
            "java -jar {} in={} out={}".format(hipsgen_path, fits_path, hips_path),
            shell=True,
        )
    else:
        print("The following hips-file already exists: {}".format(hips_path))


def preprocess_image(
    image,
    lower_clip_bound=None,
    upper_clip_bound=None,
    apply_clipping=False,
    discard_nan_cutouts=True,
):
    """Open image, remove NaNs, normalize using minmax, possibly clip background, return image."""

    # Replace NaN values with min values of the cutout
    # Rationale: we will always be missing edges of the image. Edges most likely not to
    # contain any critical morphological info. An alternative is to fully discard cutouts with
    # nans.
    empty = np.isnan(image)
    if empty.any():
        image[empty] = np.min(image[~empty])
    # replace missing values with noise
    # image[empty] = np.random.normal(loc=np.mean(image[~empty]), scale=np.std(image[~empty]),
    #                                   size=image.shape)[empty]

    # Clip background
    if apply_clipping:
        image = np.clip(image, lower_clip_bound, upper_clip_bound)

    # Scale from [0,1] using minmax
    minmax_interval = vis.MinMaxInterval()
    return minmax_interval(image)


def fits_catalogue_to_pandas(catalogue_path):
    """When given the path to a fits catalogue, returns a pandas dataframe
    that holds this catalogue."""
    # Check if path exists
    if not os.path.exists(catalogue_path):
        print("Catalogue path does not exist:", catalogue_path)
        asdasd
    df = Table.read(catalogue_path).to_pandas()
    str_df = df.select_dtypes([np.object])
    str_df = str_df.stack().str.decode("utf-8").unstack()
    for col in str_df:
        df[col] = str_df[col]
    return df


def to_composing_source_names(
    merged_source_names, comp_cat, save_path, overwrite=False
):
    """From a list of merged source names finds the component names"""
    if not os.path.exists(save_path) or overwrite:
        names = [
            comp_cat[
                comp_cat.Source_Name == merged_source_name
            ].Component_Name.values  # .astype(str)
            for merged_source_name in merged_source_names
        ]
        np.save(save_path, names)
        return names
    else:
        return np.load(save_path, allow_pickle=True)


def composing_source_names_to_cat_idx(
    list_of_composing_source_names, cat, save_path, overwrite=False
):
    """From a list of composing source names returns the corresponding cat indices"""
    if not os.path.exists(save_path) or overwrite:
        cat = copy.deepcopy(cat)
        cat = cat.reset_index()
        idxs = [
            cat[cat.Source_Name.isin(composing_source_names)].index.values
            for composing_source_names in list_of_composing_source_names
        ]
        np.save(save_path, idxs)
        return idxs
    else:
        return np.load(save_path, allow_pickle=True)


def source_names_to_cat_idx(source_names, cat, save_path, overwrite=False):
    """From a list of source names returns the corresponding cat indices"""
    if not os.path.exists(save_path) or overwrite:
        cat = copy.deepcopy(cat)
        cat = cat.reset_index()
        idxs = [
            cat[cat.Source_Name == source_name].index.values
            for source_name in source_names
        ]
        np.save(save_path, idxs)
        return idxs
    else:
        return np.load(save_path, allow_pickle=True)


def make_numpy_cutout_from_fits(
    width_in_arcsec,
    height_in_arcsec,
    ra_in_degree,
    dec_in_degree,
    fits_path,
    dimensions_normal=True,
    arcsec_per_pixel=1.5,
    just_data=True,
):
    """
    Use this function to go from a single fits file with path fits_path,
    and make a cutout around ra_in_degree, dec_in_degree with width_in_arcsec, height_in_arcsec.
    """

    # Get skycoords of the source
    skycoord = SkyCoord(ra_in_degree, dec_in_degree, unit="degree")

    # Convert dimensions in arcsec to pixelsizes
    width_in_pixels = int(width_in_arcsec / arcsec_per_pixel)
    height_in_pixels = int(height_in_arcsec / arcsec_per_pixel)

    # Load fits file
    hdulist = fits.open(fits_path)
    hdr = hdulist[0].header
    if dimensions_normal:
        hdu = hdulist[0].data
    else:
        hdu = hdulist[0].data[0, 0]
    hdulist.close()

    # Make cutout
    hdu_crop = Cutout2D(
        hdu,
        skycoord,
        (width_in_pixels, height_in_pixels),
        wcs=WCS(hdr, naxis=2),
        copy=True,
    )

    # Check dimensions
    a, b = np.shape(hdu_crop.data)
    if (a == width_in_pixels) and (b == height_in_pixels):
        if just_data:
            return hdu_crop.data
        else:
            return hdu_crop
    else:
        print(
            "Failed, dimensions are: ({},{}) and should be ({},{})".format(
                a, b, width_in_pixels, height_in_pixels
            )
        )
        return None


def create_circular_mask(size):
    """Create a circular mask that goes up to the edges of a square box.
    This square box is a numpy array with dimensions size x size."""
    r = int(size / 2)
    a, b = int(size / 2), int(size / 2)

    y, x = np.ogrid[-a : int(size) - a, -b : int(size) - b]
    mask = x * x + y * y > r * r
    return mask


def get_x_y_for_angle(length, angle_degree):

    y = np.sin(np.radians(angle_degree)) * length
    x = np.cos(np.radians(angle_degree)) * length
    return np.array([-x / 2, x / 2]), np.array([-y / 2, y / 2])


def FWHM_to_sigma_for_gaussian(fwhm):
    """Given a FWHM returns the sigma of the normal distribution."""
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def extract_gaussian_parameters_from_component_catalogue(
    pandas_cat,
    wcs,
    arcsec_per_pixel=1.5,
    PA_offset_degree=90,
    maj_min_in_arcsec=True,
    peak_flux_is_in_mJy=True,
    use_source_size=False,
):
    # Create skycoords for the center locations of all gaussians
    c = SkyCoord(pandas_cat.RA, pandas_cat.DEC, unit="deg")

    # transform ra, decs to pixel coordinates
    if maj_min_in_arcsec:
        deg2arcsec = 1
    else:
        deg2arcsec = 3600
    if peak_flux_is_in_mJy:
        mJy2Jy = 1000
    else:
        mJy2Jy = 1
    pixel_locs = skycoord_to_pixel(c, wcs, origin=0, mode="all")
    if use_source_size:
        gaussians = [
            models.Gaussian2D(
                row.Peak_flux / mJy2Jy,
                x,
                y,
                FWHM_to_sigma_for_gaussian(
                    row.source_size * deg2arcsec / arcsec_per_pixel
                ),
                FWHM_to_sigma_for_gaussian(
                    row.source_width * deg2arcsec / arcsec_per_pixel
                ),
                theta=np.deg2rad(row.source_PA + PA_offset_degree),
            )
            for ((irow, row), x, y) in zip(
                pandas_cat.iterrows(), pixel_locs[0], pixel_locs[1]
            )
        ]
    else:
        gaussians = [
            models.Gaussian2D(
                row.Peak_flux / mJy2Jy,
                x,
                y,
                FWHM_to_sigma_for_gaussian(row.Maj * deg2arcsec / arcsec_per_pixel),
                FWHM_to_sigma_for_gaussian(row.Min * deg2arcsec / arcsec_per_pixel),
                theta=np.deg2rad(row.PA + PA_offset_degree),
            )
            for ((irow, row), x, y) in zip(
                pandas_cat.iterrows(), pixel_locs[0], pixel_locs[1]
            )
        ]

    return gaussians


def mask_gaussians_from_data(gaussians, astropy_cutout):
    # Create indices
    yi, xi = np.indices(astropy_cutout.shape)

    model = np.zeros(astropy_cutout.shape, dtype=bool)
    for g in gaussians:
        gau = g(xi, yi)
        max_gau = np.max(gau)
        model[gau > 0.01 * max_gau] = True
    # astropy_cutout = np.where(gau < 0.1*max_gau, astropy_cutout, np.nan)
    astropy_cutout[model] = np.nan

    # residual = astropy_cutout - model
    return model, astropy_cutout


def subtract_gaussians_from_data(gaussians, astropy_cutout):
    # Create indices
    yi, xi = np.indices(astropy_cutout.shape)

    model = np.zeros(astropy_cutout.shape)
    for g in gaussians:
        model += g(xi, yi)
    residual = astropy_cutout - model
    return model, residual


def visually_inspect_cutouts(
    cutouts,
    pandas_catalogue=None,
    n=3,
    seed=42,
    source_name_key="Source_Name",
    plot_LGZ_Size=False,
    resize=None,
    arcsec_per_pixel=1.5,
    convolve_with_gaussian=False,
    half_signal_to_noise=False,
    convolve_sigma=1,
):
    """Given a set of cutouts (numpy arrays) with optionally corresponding pandas_catalogue,
    plot a random subset. resize should be a tuple or array with two components (output_width and
    output_length)"""
    np.random.seed(seed)
    a = np.random.randint(0, len(cutouts), n)
    if not type(cutouts) is np.ndarray:
        cutouts = np.array(cutouts)
    for im, a_index in zip(cutouts[a], a):
        image = np.copy(im)
        size = int(np.sqrt(np.size(image)))
        if not pandas_catalogue is None:
            row = pandas_catalogue.iloc[a_index]
            if "cutout_rms" in row.keys():
                print(
                    f"[RA,DEC] = {row['RA']}, {row['DEC']}, Source name {row[source_name_key]},"
                    f"{row.cutout_rms:.4f}, summed cutout value {np.sum(image):.4f}"
                )
            else:
                print(
                    f"[RA,DEC] = {row['RA']}, {row['DEC']}, Source name {row[source_name_key]},"
                    f"summed cutout value {np.sum(image):.4f}"
                )
            print(f"cutout dimensions are {size}x{size}")
        LGZ_size = row.LGZ_Size / arcsec_per_pixel
        plt.imshow(image, cmap="viridis")
        if plot_LGZ_Size:
            xs, ys = get_x_y_for_angle(LGZ_size, row.LGZ_PA)
            plt.plot(xs + size / 2, ys + size / 2, "--w")
        plt.colorbar()
        plt.show()
        if half_signal_to_noise:
            assert not pandas_catalogue is None
            noise = 0.5 * np.random.rand(size, size)
            plt.imshow(noise, cmap="viridis")
            plt.colorbar()
            plt.show()

        if convolve_with_gaussian:
            convolved_image = gaussian_filter(
                gaussian_filter(gaussian_filter(image, convolve_sigma), convolve_sigma),
                convolve_sigma,
            )
            plt.imshow(convolved_image, cmap="viridis")
            plt.colorbar()
            plt.show()

        if convolve_with_gaussian and half_signal_to_noise:
            assert not pandas_catalogue is None
            noise = 0.5 * np.random.rand(size, size)
            convolved_image = gaussian_filter(
                gaussian_filter(
                    gaussian_filter(image + noise, convolve_sigma), convolve_sigma
                ),
                convolve_sigma,
            )
            minmax_interval = vis.MinMaxInterval()
            plt.imshow(minmax_interval(convolved_image), cmap="viridis")
            plt.colorbar()
            plt.show()

        if not resize is None:
            print("Resizing")
            crop = image[
                int((size - LGZ_size) / 2) : int((size + LGZ_size) / 2),
                int((size - LGZ_size) / 2) : int((size + LGZ_size) / 2),
            ]
            img = np.array(
                Image.fromarray(crop).resize(
                    (resize[0], resize[1]), resample=Image.NEAREST
                )
            )
            plt.imshow(img, cmap="viridis")
            plt.show()
            img = np.array(
                Image.fromarray(crop).resize(
                    (resize[0], resize[1]), resample=Image.BILINEAR
                )
            )
            plt.imshow(img, cmap="viridis")
            plt.show()


class FTPWalk:
    """
    This class is contain corresponding functions for traversing the FTP
    servers using BFS algorithm.
    """

    def __init__(self, connection):
        self.connection = connection

    def listdir(self, _path):
        """
        return files and directory names within a path (directory)
        """

        file_list, dirs, nondirs = [], [], []
        try:
            self.connection.cwd(_path)
        except Exception as exp:
            print("the current path is : ", self.connection.pwd(), exp.__str__(), _path)
            return [], []
        else:
            self.connection.retrlines("LIST", lambda x: file_list.append(x.split()))
            for info in file_list:
                ls_type, name = info[0], info[-1]
                if ls_type.startswith("d"):
                    dirs.append(name)
                else:
                    nondirs.append(name)
            return dirs, nondirs

    def walk(self, path="/"):
        """
        Walk through FTP server's directory tree, based on a BFS algorithm.
        """
        dirs, nondirs = self.listdir(path)
        yield path, dirs, nondirs
        for name in dirs:
            path = os.path.join(path, name)
            yield from self.walk(path)
            # In python2 use:
            # for path, dirs, nondirs in self.walk(path):
            #     yield path, dirs, nondirs
            self.connection.cwd("..")
            path = os.path.dirname(path)


def get_FIRST_image(
    ra,
    dec,
    local_dir="/home/rafael/data/mostertrij/data/FIRST",
    FIRST_index_filename="index.npy",
    verbose=False,
):
    """query FIRST image ftp.
    ra and dec are expected to be in degree"""
    import wget

    print("Get FIRST image")

    assert isinstance(ra, float)
    assert isinstance(dec, float)
    # Transform ra, dec into hms as used in FIRST
    c = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="fk5")
    RA_str = f"{int(c.ra.hms[0]):02d}{int(c.ra.hms[1]):02d}{round(c.ra.hms[2]/60):.0f}"
    RA_part_object = int(RA_str)
    DEC_str = f"{c.dec.degree*1000:.0f}"
    DEC_part_object = int(DEC_str)

    # Read list of FIRST files
    FIRST_index_path = os.path.join(local_dir, FIRST_index_filename)
    if not os.path.exists(FIRST_index_path):
        connection = ftplib.FTP("archive.stsci.edu")  # /pub/vla_first/data")
        connection.login()
        ftpwalk = FTPWalk(connection)
        all_files = []
        top_dirs, _ = ftpwalk.listdir("/pub/vla_first/data/")
        start = time.time()
        for i, d in enumerate(top_dirs):
            try:
                _, files = ftpwalk.listdir(os.path.join("/pub/vla_first/data/", d))
                all_files += files
            except:
                time.sleep(5)
            time.sleep(1)
            print(
                f"{i}/{len(top_dirs)} dirs searched. Time taken: {time.time()-start:.0f} sec."
            )
        np.save(FIRST_index_path, all_files)
    else:
        all_files = np.load(FIRST_index_path)

    # Get FIRST directory
    RA_parts_str = np.array([v[:5] for v in all_files])
    RA_parts = np.array([int(v[:5]) for v in all_files])
    RA_parts_index = np.argmin(abs(RA_parts - RA_part_object))
    RA_dif = np.min(abs(RA_parts - RA_part_object))
    # print("RA:", RA_str, RA_parts_str, RA_parts)
    dir_name = RA_parts_str[RA_parts_index]
    local_dir_path = os.path.join(local_dir, dir_name)
    if not os.path.exists(local_dir_path):
        os.mkdir(local_dir_path)

    # DEC_subset_with_dups = [(i,v) for i, v in enumerate(all_files) if v.startswith(dir_name)]
    DEC_subset_with_dups = [v for v in all_files if v.startswith(dir_name)]
    check_val = set()  # Check Flag
    DEC_subset = []
    for v in DEC_subset_with_dups:
        if v[5:11] not in check_val:
            DEC_subset.append(v)
            check_val.add(v[5:11])

    DEC_parts_str = np.array([v[5:11] for v in DEC_subset])
    DEC_parts = np.array([int(v[5:11]) for v in DEC_subset])
    # print(c.dec.degree, DEC_str,"DEC:", DEC_parts)
    # DEC_parts_index = np.argmin(abs(DEC_parts-DEC_part_object))
    # DEC_dif = np.min(abs(DEC_parts-DEC_part_object))
    DEC_parts_index = np.argsort(abs(DEC_parts - DEC_part_object))[:4]
    DEC_dif = np.sort(abs(DEC_parts - DEC_part_object))[:4]
    DEC_parts_index = [
        i for i, v in zip(DEC_parts_index, DEC_dif) if v - DEC_dif[0] < 600
    ]
    if verbose:
        # print('DEC_parts',[i for i in DEC_parts_index])
        print("DEC_parts", [DEC_subset[i] for i in DEC_parts_index])
        print("min RA distance:", RA_dif, "min DEC distance:", DEC_dif)

    first_filenames = [DEC_subset[i] for i in DEC_parts_index]
    # first_url = os.path.join('https://archive.stsci.edu/pub/vla_first/data/',dir_name, first_filename)
    first_urls = [
        os.path.join(
            "https://archive.stsci.edu/pub/vla_first/data/", dir_name, first_filename
        )
        for first_filename in first_filenames
    ]
    if DEC_dif[0] > 46 * 60:
        return None, None, None, None

    if verbose:
        print(ra, dec, first_urls[0])
        print(c.ra.hms, RA_str, RA_part_object)
    # Download file
    local_fits_paths = [
        os.path.join(local_dir_path, first_filename)
        for first_filename in first_filenames
    ]
    [
        wget.download(url=first_url, out=local_dir_path)
        for first_url, local_fits_path in zip(first_urls, local_fits_paths)
        if not os.path.exists(local_fits_path)
    ]

    # If multiple fits files have been suggested, test distance to the central coordinate
    if len(first_filenames) > 1:
        best_i = 0
        shortest_dist = 1e9
        for i, local_fits_path in enumerate(local_fits_paths):
            # Open hdr and get central coords
            hdu, hdr = load_fits(local_fits_path, dimensions_normal=True)
            if verbose:
                print(hdr["CRVAL1"], hdr["CRVAL2"])
            central_ra, central_dec = hdr["CRVAL1"], hdr["CRVAL2"]
            central_c = SkyCoord(
                ra=central_ra, dec=central_dec, unit=(u.deg, u.deg), frame="fk5"
            )
            dist = central_c.separation(c).deg
            if dist < shortest_dist:
                best_i = i
                shortest_dist = dist
        return (
            dir_name,
            first_filenames[best_i],
            first_urls[best_i],
            local_fits_paths[best_i],
        )
    else:
        return dir_name, first_filenames[0], first_urls[0], local_fits_paths[0]


def return_FIRST_cutout(
    ra,
    dec,
    size_in_arcsec,
    save_dir="/home/rafael/data/mostertrij/data",
    arcsec_per_pixel=1.8,
    overwrite=False,
    mode="partial",
    verbose=False,
):
    """Create FIRST cutout
    ra and dec are expected to be in degree"""
    assert isinstance(ra, float)
    assert isinstance(dec, float)
    save_dir = os.path.join(save_dir, "FIRST")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"FIRST_ra{ra}dec{dec}size{size_in_arcsec}.npy")
    if not overwrite and os.path.exists(save_path):
        fim = np.load(save_path, allow_pickle=True)
    else:

        # Get local FIRST field (download if not locally present)
        dir_name, first_filename, first_url, local_fits_path = get_FIRST_image(
            ra, dec, local_dir=save_dir, verbose=verbose
        )
        if dir_name is None:
            print(f"FIRST cutout not found for RA;DEC {ra:.3f} {dec:.3f}")
            return None

        # Create skycoord
        skycoord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="fk5")

        # Load FITS file
        hdu, hdr = load_fits(local_fits_path, dimensions_normal=False)

        # Extract cutout
        pixelsize = size_in_arcsec / arcsec_per_pixel

        try:
            hdu_crop = Cutout2D(
                hdu,
                skycoord,
                (pixelsize, pixelsize),
                wcs=WCS(hdr, naxis=2),
                copy=True,
                mode=mode,
                fill_value=0,
            )
            fim = hdu_crop.data
        except:
            fim = None
        np.save(save_path, fim)

    return fim


def convert1d_22d(fits_directory, store_file="cutout_list", apply_clipping=True):

    store_file = store_file + f"_clipped_{apply_clipping}"
    image_list_new = []
    if os.path.exists(os.path.join(fits_directory, store_file + ".npy")):
        image_list = np.load(os.path.join(fits_directory, store_file + ".npy"))
        if len(np.shape(image_list[0])) < 2:
            width = int(np.sqrt(len(image_list[0])))
            print(width, np.shape(image_list[0])[0])
            image_list_new = np.array(
                [image.reshape(width, width) for image in image_list]
            )
            np.save(os.path.join(fits_directory, store_file), np.array(image_list_new))
        else:
            print("No cleanup needed")
    else:
        print("Its not my job to clean up")


def get_coordinates_on_grid_in_observation(
    hdu, hdr, wcs, grid_interval_arcsec=500, verbose=True
):
    """Given a HDU and header (hdr) of an observation,
    sample a dense grid of coordinates that cover the entire observation.
    Return these coordinates in the form of Astropy skycoords"""

    # Calculate arcsec per pixel for RA
    w, h = hdu.data.shape[0], hdu.data.shape[1]
    # Create skycoord for each corner and the middle of each side of the image
    coords = pixel_to_skycoord(
        [0, w, 0, w, int(0.5 * w), int(0.5 * w), 0, w],
        [0, 0, h, h, 0, h, int(0.5 * h), int(0.5 * h)],
        wcs,
    )

    # Find largest range of coordinates in our image
    minra = np.min([c.ra.deg for c in coords])
    mindec = np.min([c.dec.deg for c in coords])
    maxra = np.max([c.ra.deg for c in coords])
    maxdec = np.max([c.dec.deg for c in coords])

    # Find image width
    c = pixel_to_skycoord([0, w, 0, w], [0, 0, h, h], wcs)
    width_degree = abs((c[1].ra.deg - c[0].ra.deg + c[3].ra.deg - c[2].ra.deg) * 0.5)
    width_arcsec = width_degree * 3600
    arcsec_per_pixel_RA = width_arcsec / w
    # Calculate arcsec per pixel for DEC
    c = pixel_to_skycoord([0, 0, w, w], [0, h, 0, h], wcs)
    height_degree = abs(
        (c[1].dec.deg - c[0].dec.deg + c[3].dec.deg - c[2].dec.deg) * 0.5
    )
    height_arcsec = height_degree * 3600
    arcsec_per_pixel_DEC = height_arcsec / h
    if verbose:
        print(
            f"Image widthxheight is {width_degree:.3f}x{height_degree:.3f} degree,\n"
            f"Image widthxheight is {width_arcsec:.1f}x{height_arcsec:.1f} arcsec, and the RA, DEC resolution is:\n"
            f"{arcsec_per_pixel_RA:.3f},{arcsec_per_pixel_DEC:.3f} arcsec per pixel"
        )

    # Create meshgrid for these coordinates
    nx, ny = (
        int(np.ceil(width_arcsec / grid_interval_arcsec)),
        int(np.ceil(height_arcsec / grid_interval_arcsec)),
    )
    x = np.linspace(minra, maxra, nx)
    y = np.linspace(mindec, maxdec, ny)
    xv, yv = np.meshgrid(x, y)
    # Turn ra,decs into skycoords
    grid_skycoords = SkyCoord(xv.ravel(), yv.ravel(), unit="degree")

    return grid_skycoords, minra, mindec, maxra, maxdec


def single_fits_to_png_cutouts(
    destination_size_arcsec,
    giant_RAs,
    giant_DECs,
    fits_directory,
    store_directory,
    low_mosaic_name=None,
    high_mosaic_name=None,
    verbose=True,
    store_name_prefix="",
    n_fields="",
    mode="strict",
    dimensions_normal=True,
    probability_of_saving_non_giant=0.3,
    asinh=True,
    sqrt_stretch=True,
    rescale=False,
):
    """Use this function to go from a single fits file, located in fits_directory,
    to a gridded bunch of cutouts saved as pngs."""

    start = time.time()
    minmax_interval = vis.MinMaxInterval()
    assert (low_mosaic_name is None) or (
        high_mosaic_name is None
    ), "Only supports one at the time for now"
    if low_mosaic_name is None:
        mosaic_name = high_mosaic_name
    else:
        mosaic_name = low_mosaic_name

    assert mosaic_name.endswith(".fits"), mosaic_name
    # Create dirs for cutouts
    giant_dir = os.path.join(store_directory, "giants")
    not_giant_dir = os.path.join(store_directory, "not_giants")
    [os.makedirs(d, exist_ok=True) for d in [store_directory, giant_dir, not_giant_dir]]

    # Load fits file
    if verbose:
        print("Tring to open fitsfile:", os.path.join(fits_directory, mosaic_name))
    hdu, hdr = load_fits(
        os.path.join(fits_directory, mosaic_name), dimensions_normal=dimensions_normal
    )
    wcs = WCS(hdr, naxis=2)
    # Create grid points along fits file
    skycoords, minra, mindec, maxra, maxdec = get_coordinates_on_grid_in_observation(
        hdu, hdr, wcs, grid_interval_arcsec=int(destination_size_arcsec), verbose=True
    )

    # Discard giant coordinates outside the range of this fits file
    indices = [
        i
        for i, (ra, dec) in enumerate(zip(giant_RAs, giant_DECs))
        if minra < ra < maxra and mindec < dec < maxdec
    ]
    if len(indices) < 1:
        print("No giants inside this fits file. Skipping pointing: ", mosaic_name)
        return
    if verbose:
        print(
            f"Limits of this fits file are (minra,maxra,mindec,maxdec):",
            minra,
            maxra,
            mindec,
            maxdec,
        )
        print(
            f"At most {len(indices)} out of the {len(giant_RAs)} giants lie within this fits file."
        )
    giant_RAs = giant_RAs[indices]
    giant_DECs = giant_DECs[indices]
    giant_skycoords = SkyCoord(giant_RAs, giant_DECs, unit="degree")

    # Get rotated size
    fullsize = int(destination_size_arcsec * np.sqrt(2))
    if verbose:
        print(f"Attempt to create {len(skycoords)} cutouts.")
        ctrl_ra, ctrl_dec = hdr_to_central_RA_DEC(
            os.path.join(fits_directory, mosaic_name)
        )
        print(
            f"In the fits file centred on RA,DEC [degree] {ctrl_ra:.2f} {ctrl_dec:.2f}"
        )
    # Create cutout and calculate rms for each skycoord
    failcount = 0
    successcount = 0
    for i, co in enumerate(skycoords):

        # Extract cutout
        try:
            hdu_crop = Cutout2D(
                hdu, co, (fullsize, fullsize), wcs=wcs, copy=True, mode=mode
            )
        except (NoOverlapError, PartialOverlapError) as e:
            failcount += 1
            continue

        # if verbose:
        #    print('Cutout created. shape is:', np.shape(hdu_crop.data))
        a, b = np.shape(hdu_crop.data)
        if a * b == int(round(fullsize)) ** 2 and not np.isnan(hdu_crop.data).any():
            image = hdu_crop.data
            # Scale from [0,1] using minmax
            if rescale:
                image = minmax_interval(image)
                if sqrt_stretch:
                    stretch = vis.SqrtStretch()
                    image = stretch(image)
            if asinh:
                assert rescale == False
                transform = vis.AsinhStretch() + vis.PercentileInterval(99.0)  # 5)
                image = transform(image)

            # Check if giant is present in cutout
            hdu_crop_small = Cutout2D(
                hdu,
                co,
                (destination_size_arcsec, destination_size_arcsec),
                wcs=wcs,
                copy=True,
                mode=mode,
            )
            contains_giant = hdu_crop_small.wcs.footprint_contains(giant_skycoords)
            # print("Contains giant:", contains_giant)

            # Save image as png
            save_name = store_name_prefix + f"{co.ra.deg:.4f}_{co.dec.deg:.4f}.png"
            if any(contains_giant):
                save_path = os.path.join(giant_dir, save_name)
                # Plot giant location
                cutout_skycoords = SkyCoord(
                    giant_RAs[contains_giant], giant_DECs[contains_giant], unit="degree"
                )
                xp, yp = skycoord_to_pixel(cutout_skycoords, hdu_crop.wcs)
                xp = list(map(int, xp))
                yp = list(map(int, yp))
                # FOr debugging to see where the giants are at.
                """
                image[yp,xp] = 0.9
                for tt in range(-5,5):
                    image[np.array(yp)+tt,xp] = 0.9
                    image[yp,np.array(xp)+tt] = 0.9
                """

            else:
                save_path = os.path.join(not_giant_dir, save_name)
                coin = np.random.uniform(0, 1)
                # Here we limit the number
                # of abundant class members
                # print("Flipping a coin:", coin, coin<probability_of_saving_non_giant)
                if coin > probability_of_saving_non_giant:
                    continue
            plt.imsave(save_path, image, cmap="viridis")  # ,cmap='RdBu')
            successcount += 1

        else:
            # If you end up here that probably means some of skycoords
            # were located too close to the border of the fits-file for cutout2D to create a
            #  (fullsize * fullsize)-sized box around them.
            failcount += 1
            # if verbose:
            #    print('Cutout extraction failed, probably close to the pointing border.')

    if verbose:
        print(f"successcount: {successcount}, failcount: {failcount}")
        print("Time elapsed:", int(time.time() - start), "seconds.")


# @profile
def single_fits_to_numpy_cutouts_using_astropy_better(
    fullsize,
    pandas_catalogue,
    ra_key,
    dec_key,
    fits_directory,
    mosaic_name,
    apply_clipping=True,
    apply_mask=True,
    verbose=True,
    mode="strict",
    store_file="cutout_list",
    dimensions_normal=True,
    variable_size=False,
    asinh=False,
    hdf=True,
    rescale=True,
    sqrt_stretch=False,
    store_directory=None,
    destination_size=None,
    lower_sigma_limit=1.5,
    upper_sigma_limit=1e9,
    arcsec_per_pixel=1.5,
    overwrite=False,
):
    """Use this function to go from a single fits file, located in a pointing directory given by
    fits_directory, and a corresponding pandas catalogue with the RA and DEC coordinates to its objects,
    to get cutouts  of these objects."""

    start = time.time()
    if verbose:
        print(f"Retrieving cutouts of pointing")
    store_file = store_file + f"_clipped_{apply_clipping}"
    store_cat_path = os.path.join(
        store_directory, store_file + "_mosaic.cat.extracted.h5"
    )
    if store_directory is None:
        store_directory = fits_directory
    if overwrite or not os.path.exists(
        os.path.join(store_directory, store_file + ".npy")
    ):
        # Initialize rms_list and fail counters and mosaic path
        image_list, extraction_succeeded_list = [], []
        minmax_interval = vis.MinMaxInterval()

        if apply_mask:
            mask = create_circular_mask(fullsize)

        # Get skycoords of all sources in the pointing
        if verbose:
            print("pandas to skycoord")
        skycoords = pandas_catalogue_to_astropy_coords(
            pandas_catalogue, ra_key, dec_key
        )

        # Load fits file
        if verbose:
            print("Try to open fitsfile:", os.path.join(fits_directory, mosaic_name))
        if not mosaic_name.endswith(".fits"):
            mosaic_name += ".fits"
        hdu, hdr = load_fits(
            os.path.join(fits_directory, mosaic_name),
            dimensions_normal=dimensions_normal,
        )

        if verbose:
            print("Attempt to create cutout...")
        # Create cutout and calculate rms for each skycoord
        for i, ((i_p, source), co) in enumerate(
            zip(pandas_catalogue.iterrows(), skycoords)
        ):
            # if verbose:
            #    print(source)
            #    print(fullsize)
            #    print(co)
            if variable_size:
                try:
                    # Where 1.5 is the amount of arcsec per pixel
                    fullsize = int(
                        np.ceil(source.source_size * np.sqrt(2) / arcsec_per_pixel)
                    )
                except:
                    # Where 1.5 is the amount of arcsec per pixel
                    fullsize = int(
                        np.ceil(source.LGZ_Size * np.sqrt(2) / arcsec_per_pixel)
                    )
            # Extract cutout
            hdu_crop = Cutout2D(
                hdu,
                co,
                (fullsize, fullsize),
                wcs=WCS(hdr, naxis=2),
                copy=True,
                mode=mode,
            )
            if verbose:
                print("Cutout created. shape is:", np.shape(hdu_crop.data))
                print("cutout_rms", source.cutout_rms)
                # print(source)
            a, b = np.shape(hdu_crop.data)
            if (
                a * b == int(round(fullsize)) ** 2
                and not np.isnan(source.cutout_rms)
                and not np.isnan(hdu_crop.data).any()
            ):
                image = hdu_crop.data
                # Rescale to destination size for variable cutout
                if variable_size:
                    image = np.array(
                        Image.fromarray(image).resize(
                            (destination_size, destination_size),
                            resample=Image.BILINEAR,
                        )
                    )
                    fullsize = destination_size
                if apply_clipping:
                    if verbose:
                        print("attempt to clip")
                    image = np.clip(
                        image,
                        lower_sigma_limit * source.cutout_rms,
                        upper_sigma_limit * source.cutout_rms,
                    )
                if apply_mask:
                    # image.shape = (fullsize, fullsize)
                    image[mask] = np.min(image)
                    # image.shape = (fullsize*fullsize)
                # Scale from [0,1] using minmax
                if rescale:
                    if verbose:
                        print("attempt to rescale")
                    image = minmax_interval(image)
                    if sqrt_stretch:
                        stretch = vis.SqrtStretch()
                        image = stretch(image)
                if asinh:
                    assert rescale == False
                    transform = vis.AsinhStretch() + vis.PercentileInterval(99.5)
                    image = transform(image)

                image_list.append(image)
                extraction_succeeded_list.append(True)
            else:
                # If you end up here that probably means some of skycoords
                # were located too close to the border of the fits-file for cutout2D to create a
                #  (fullsize * fullsize)-sized box around them.
                extraction_succeeded_list.append(False)
                if verbose:
                    # , source, 'dimensions: ({},{})'.format(a,b))
                    print("failed:")

        if verbose:
            print("Attempt to save image list...")
        # Save rms_list and add it to the pandas catalogue
        np.save(os.path.join(store_directory, store_file), np.array(image_list))
        if verbose:
            print("Attempt to save pandas cat...")
        # Save catalogue
        pandas_catalogue = pandas_catalogue.loc[extraction_succeeded_list]
        if not os.path.exists(store_cat_path) or overwrite:
            if hdf:
                pandas_catalogue.to_hdf(store_cat_path, key="df")
            else:
                pandas_catalogue.to_csv(
                    os.path.join(
                        store_directory, store_file + "_mosaic.cat.success.csv"
                    )
                )
    else:
        if verbose:
            print(
                "Numpy store_file containing cutouts already exists. Loading it in..."
            )
        image_list = np.load(os.path.join(store_directory, store_file + ".npy"))
        if hdf:
            pandas_catalogue = pd.read_hdf(store_cat_path, "df")
        else:
            pandas_catalogue = pd.read_csv(
                os.path.join(store_directory, store_file + "_mosaic.cat.success.csv")
            )
    if verbose:
        print("Time elapsed:", int(time.time() - start), "seconds.")
    return np.array(image_list), pandas_catalogue


# @profile
def single_fits_to_rms_using_astropy(
    fullsize,
    pandas_catalogue,
    ra_key,
    dec_key,
    rms_fits_directory,
    rms_fits_path,
    verbose=True,
    arcsec_per_pixel=1.5,
    mode="partial",
    store_file="cutouts_rms",
    store_directory=None,
    dimensions_normal=True,
    overwrite=False,
):
    """Use this function to go from a single fits file, located in a pointing directory given by
    rms_fits_directory, and a corresponding pandas catalogue with the RA and DEC coordinates to its objects,
    to get cutouts  of these objects and save per cutout the rms median."""

    start = time.time()
    if verbose:
        print(f"Retrieving median rms of pointing")
    if store_directory is None:
        store_directory = rms_fits_directory
    if overwrite or not os.path.exists(
        os.path.join(store_directory, store_file + ".npy")
    ):
        # Initialize rms_list and fail counters and mosaic path
        rms_list, count = [], 0

        # Get skycoords of all sources in the pointing
        skycoords = pandas_catalogue_to_astropy_coords(
            pandas_catalogue, ra_key, dec_key
        )

        # Load fits file
        # If error pops up mentioning "large and small arrays do not match up"
        # dimensions are probably not normal
        hdu, hdr = load_fits(rms_fits_path, dimensions_normal=dimensions_normal)

        # Create cutout and calculate rms for each skycoord
        for i, ((i_p, source), co) in enumerate(
            zip(pandas_catalogue.iterrows(), skycoords)
        ):
            hdu_crop = Cutout2D(
                hdu,
                co,
                (fullsize, fullsize),
                wcs=WCS(hdr, naxis=2),
                copy=True,
                mode=mode,
            )
            a, b = np.shape(hdu_crop.data)
            if a * b == int(round(fullsize)) ** 2:
                image = hdu_crop.data
                rms_list.append(np.nanmedian(image))
            else:
                # If you end up here that probably means some of skycoords
                # were located too close to the border of the fits-file for cutout2D to create a
                #  (fullsize * fullsize)-sized box around them.
                rms_list.append(None)
                if verbose:
                    print("failed:", source, "dimensions: ({},{})".format(a, b))

        # Save rms_list and add it to the pandas catalogue
        np.save(os.path.join(store_directory, store_file), np.array(rms_list))
    else:
        print("Numpy store_file containing cutouts already exists. Loading it in...")
        rms_list = np.load(os.path.join(store_directory, store_file + ".npy"))
    # Note that None objects will be converted into NaNs bij pandas
    pandas_catalogue["cutout_rms"] = rms_list
    if verbose:
        print("Time elapsed:", int(time.time() - start), "seconds.")
    return pandas_catalogue, rms_list


def single_fits_to_numpy_cutouts_using_astropy(
    fullsize,
    pandas_catalogue,
    ra_key,
    dec_key,
    data_directory,
    numpy_filename,
    fits_filename,
    mode="partial",
    dimensions_normal=True,
    overwrite=False,
    preprocess=True,
    apply_clipping=False,
    lower_clip_bound=None,
    upper_clip_bound=None,
):
    """Use this function to go from a single fits file, located in the data_directory, and a
    corresponding pandas catalogue with the RA and DEC coordinates to its objects, to get cutouts
    of these objects delivered in a numpy list.
    If you have multiple directories containing fits files, check the function
    fits_to_numpy_cutouts_using_astropy. Subsequently use the function write_binary or
    write_train_test_binary to go from numpy to binary format."""

    warnings.warn(
        "deprecated, use the 'single_fits_to_numpy_cutouts_using_astropy_better'"
        "function instead.",
        DeprecationWarning,
    )

    start = time.time()
    if overwrite or not os.path.exists(
        os.path.join(data_directory, numpy_filename + ".npy")
    ):
        # Make python list
        image_list = []
        corrected_catalogue_index = []
        count = 0
        failures = 0

        # Get skycoords of all sources
        skycoords = pandas_catalogue_to_astropy_coords(
            pandas_catalogue, ra_key, dec_key
        )
        print("Skycoords retrieved")

        ra_index = list(pandas_catalogue.iloc[0].index).index(ra_key)
        dec_index = list(pandas_catalogue.iloc[0].index).index(dec_key)
        fits_file_path = os.path.join(data_directory, fits_filename + ".fits")
        # Load fits file
        hdulist = fits.open(fits_file_path)
        hdr = hdulist[0].header
        if dimensions_normal:
            hdu = hdulist[0].data
            # Convert skycoords to pixel values (useful for individual pybdsf manipulations)
            pixel_positions = np.array(
                [cor.to_pixel(WCS(hdr, naxis=1)) for cor in skycoords]
            )
        else:
            hdu = hdulist[0].data[0, 0]
            # Convert skycoords to pixel values (useful for individual pybdsf manipulations)
            pixel_positions = np.array(
                [cor.to_pixel(WCS(hdr, naxis=2)) for cor in skycoords]
            )
        # Header
        hdulist.close()

        for i, ((i_p, source), co) in enumerate(
            zip(pandas_catalogue.iterrows(), skycoords)
        ):
            # Progress notification
            if i % 1000 == 0 and i > 0:
                print(
                    i,
                    "/",
                    len(pandas_catalogue),
                    "Time elapsed:",
                    int(time.time() - start),
                    "seconds. Estimated time left:",
                    round(
                        len(pandas_catalogue) / (i / (time.time() - start)) / 3600, 1
                    ),
                    " hours.",
                )

            hdu_crop = Cutout2D(
                hdu,
                co,
                (fullsize, fullsize),
                wcs=WCS(hdr, naxis=2),
                copy=True,
                mode=mode,
            )
            a, b = np.shape(hdu_crop.data)
            # print(type(hdu_crop.data), np.shape(hdu_crop.data))
            if a * b == int(round(fullsize)) ** 2:
                if preprocess:
                    image = preprocess_image(
                        hdu_crop.data,
                        apply_clipping=apply_clipping,
                        lower_clip_bound=lower_clip_bound,
                        upper_clip_bound=upper_clip_bound,
                    )
                else:
                    image = hdu_crop.data
                image_list.append(image.reshape(fullsize * fullsize))
                count += 1
                corrected_catalogue_index.append(i_p)
            else:
                print("failed:", source, "dimensions: ({},{})".format(a, b))
                failures += 1

        np.save(os.path.join(data_directory, numpy_filename), np.array(image_list))
        np.save(os.path.join(data_directory, "pixel_positions"), pixel_positions)
        np.save(
            os.path.join(data_directory, "corrected_catalogue_" + numpy_filename),
            np.array(corrected_catalogue_index),
        )
        print("Failed to extract:", failures)
    else:
        print("Numpy file containing cutouts already exists. Loading it in...")
        image_list = np.load(os.path.join(data_directory, numpy_filename + ".npy"))
        pixel_positions = np.load(os.path.join(data_directory, "pixel_positions.npy"))
        corrected_catalogue_index = np.load(
            os.path.join(
                data_directory, "corrected_catalogue_" + numpy_filename + ".npy"
            )
        )
    print(
        "Time elapsed:",
        int(time.time() - start),
        "seconds." " \nNumber of images in numpy file:",
        len(image_list),
        "\\nnNumber of images in catalog:",
        len(pandas_catalogue),
    )
    if len(image_list) != len(pandas_catalogue):
        print(
            """If not all catalogue entries were extracted, that probably means some of them
                    were located too close to the border of the fits-file for cutout2D to create a
                    (fullsize * fullsize)-sized box around them."""
        )
    return image_list, corrected_catalogue_index, pixel_positions


def check_std_of_fits_files(data_path, fits_filename, save_dir, save_name):
    """Given a set of fits files, returns for each fits file its
    standard deviation and the mean of these deviations.
    """
    if not os.path.exists(os.path.join(save_dir, save_name + ".npz")):
        # Create list of directories that contain fits files in the data directory
        directory_names = get_immediate_subdirectories(data_path)
        standard_deviations = []
        for i, directory in enumerate(directory_names):
            fits_file_path = os.path.join(data_path, directory, fits_filename + ".fits")
            print("Progress: {}/{}, ".format(i, len(directory_names)), fits_file_path)

            # Load new fits file
            hdulist = fits.open(fits_file_path)
            hdu = hdulist[0].data
            hdu = hdu[~np.isnan(hdu)]
            standard_deviations.append(np.std(hdu))
            print(np.std(hdu))
            hdulist.close()
            mstd = np.mean(standard_deviations)
        # save
        np.savez(
            os.path.join(save_dir, save_name), mean_std=mstd, stds=standard_deviations
        )
    else:
        # load
        loaded = np.load(os.path.join(save_dir, save_name + ".npz"))
        mstd = loaded["mean_std"]
        standard_deviations = loaded["stds"]

    return mstd, standard_deviations


def export_fits_catalogue_to_hdf(
    fits_catalogue_path, out_path, verbose=True, overwrite=False, binary_to_str=False
):
    """Loads a fits catalogue with Astropy Table and exports it to a HDF5 file."""
    # Check if fits file has extension
    assert fits_catalogue_path.endswith(".fits"), fits_catalogue_path
    assert out_path.endswith(".h5"), out_path

    # Load catalog with astropy Table
    if overwrite or not os.path.exists(out_path):
        print(f"transforming {fits_catalogue_path} to {out_path}")
        df = Table.read(fits_catalogue_path).to_pandas()

        if binary_to_str:
            str_df = df.select_dtypes([np.object])
            str_df = str_df.stack().str.decode("utf-8").unstack()
            for col in str_df:
                df[col] = str_df[col]
        df.to_hdf(out_path, key="df")
    else:
        if verbose:
            print(f"HDF5 file already exists at: {out_path}")


def export_fits_catalogue_to_csv(
    stilts_path, data_dir, cat_name, out_dir, out_name, verbose=True, overwrite=False
):
    """Using STILTS (commandline topcat tool) export a fits catalogue to a csv file."""
    # Check if fits file has extension
    if not cat_name.endswith(".fits"):
        cat_name += ".fits"
    if not out_name.endswith(".csv"):
        out_name += ".csv"
    # Export csv file using stilts
    if overwrite or not os.path.exists(os.path.join(out_dir, out_name)):

        stilts_string = """{0} tpipe in={1}/{2}  ofmt=csv out={3}/{4}""".format(
            stilts_path, data_dir, cat_name, out_dir, out_name
        )
        if verbose:
            print("\n", stilts_string)
        output = subprocess.call(stilts_string, shell=True)
        # Check if new file is created
        assert os.path.exists(os.path.join(data_dir, out_name))
        if verbose:
            print("Done.")
    else:
        if verbose:
            print(
                "Csv file with the name '{0}' already exists in directory '{1}'.".format(
                    cat_name, data_dir
                )
            )


def get_pointing_directories(
    data_directory,
    cat_directory,
    mosaic_name,
    rms_name,
    cat_name,
    RA_0h_13h_split=False,
):
    """Data directory is the path to directory where the mosaic directories are located,
    mosaic_name, rms_name and cat_names are fits files that should reside in each mosaic directory"""
    if RA_0h_13h_split:
        pointing_names1 = [
            o
            for o in os.listdir(os.path.join(data_directory, "RA0h_field"))
            if os.path.isdir(os.path.join(data_directory, "RA0h_field", o))
            and o.startswith("P")
        ]
        pointing_names2 = [
            o
            for o in os.listdir(os.path.join(data_directory, "RA13h_field"))
            if os.path.isdir(os.path.join(data_directory, "RA13h_field", o))
            and o.startswith("P")
        ]
        pointing_names = pointing_names1 + pointing_names2
        pointing_directories1 = [
            os.path.join(data_directory, "RA0h_field", o) for o in pointing_names
        ]
        pointing_directories2 = [
            os.path.join(data_directory, "RA13h_field", o) for o in pointing_names
        ]
        pointing_directories = pointing_directories1 + pointing_directories2
        cat_directories = [os.path.join(cat_directory, o) for o in pointing_names]

    else:

        pointing_names = [
            o
            for o in os.listdir(data_directory)
            if os.path.isdir(os.path.join(data_directory, o)) and o.startswith("P")
        ]
        pointing_directories = [os.path.join(data_directory, o) for o in pointing_names]
        cat_directories = [os.path.join(cat_directory, o) for o in pointing_names]

    # Make sure the file-endings are fits files
    if not mosaic_name.endswith(".fits"):
        mosaic_name += ".fits"
    if not rms_name.endswith(".fits"):
        rms_name += ".fits"
    if not cat_name.endswith(".fits"):
        cat_name += ".fits"
    p_names, p_dirs, p_cats, p_rms = [], [], [], []
    for pd, pn, cat_dir in zip(pointing_directories, pointing_names, cat_directories):
        mp = os.path.join(pd, mosaic_name)
        rp = os.path.join(cat_dir, rms_name)
        cp = os.path.join(cat_dir, cat_name)
        if os.path.exists(mp) and os.path.exists(rp) and os.path.exists(cp):
            p_names.append(pn)
            p_dirs.append(pd)
            p_cats.append(cp)
            p_rms.append(rp)
    print(
        f"{data_directory} contains {len(p_names)} fields with corresponding rms_map and catalogue."
    )
    return p_names, p_dirs, p_cats, p_rms


def get_remote_pointing_directories_beyond_DR2(
    remote_data_directory, cat_directory, mosaic_name, cat_name
):
    """Data directory is the path to directory where the mosaic directories are located,
    mosaic_name,  and cat_names are fits files that should reside in each mosaic directory"""

    pointing_names = np.array(
        [
            o
            for o in os.listdir(remote_data_directory)
            if os.path.isdir(os.path.join(remote_data_directory, o))
            and o.startswith("P")
        ]
    )
    pointing_directories = np.array(
        [os.path.join(remote_data_directory, o) for o in pointing_names]
    )
    cat_directories = np.array([os.path.join(cat_directory, o) for o in pointing_names])
    ind = np.argsort(pointing_names)
    pointing_names = pointing_names[ind]
    pointing_directories = pointing_directories[ind]
    cat_directories = cat_directories[ind]

    # Make sure the file-endings are fits files
    if not mosaic_name.endswith(".fits"):
        mosaic_name += ".fits"
    if not cat_name.endswith(".fits"):
        cat_name += ".fits"
    p_names, p_dirs, p_cats = [], [], []
    for pd, pn, cat_dir in zip(pointing_directories, pointing_names, cat_directories):
        mp = os.path.join(pd, mosaic_name)
        cp = os.path.join(cat_dir, cat_name)
        if os.path.exists(mp) and os.path.exists(cp):
            p_names.append(pn)
            p_dirs.append(pd)
            p_cats.append(cp)
    print(
        f"{remote_data_directory} contains {len(p_names)} fields with corresponding rms_map and catalogue."
    )

    return p_names, p_dirs, p_cats


def get_remote_pointing_directories(
    remote_data_directory, cat_directory, mosaic_name, rms_name, cat_name
):
    """Data directory is the path to directory where the mosaic directories are located,
    mosaic_name, rms_name and cat_names are fits files that should reside in each mosaic directory"""

    pointing_names1 = [
        o
        for o in os.listdir(os.path.join(remote_data_directory, "RA0h_field"))
        if os.path.isdir(os.path.join(remote_data_directory, "RA0h_field", o))
        and o.startswith("P")
    ]
    pointing_names2 = [
        o
        for o in os.listdir(os.path.join(remote_data_directory, "RA13h_field"))
        if os.path.isdir(os.path.join(remote_data_directory, "RA13h_field", o))
        and o.startswith("P")
    ]
    pointing_names = np.array(pointing_names1 + pointing_names2)
    pointing_directories1 = [
        os.path.join(remote_data_directory, "RA0h_field", o) for o in pointing_names1
    ]
    pointing_directories2 = [
        os.path.join(remote_data_directory, "RA13h_field", o) for o in pointing_names2
    ]
    pointing_directories = np.array(pointing_directories1 + pointing_directories2)
    ind = np.argsort(pointing_names)
    pointing_names = pointing_names[ind]
    pointing_directories = pointing_directories[ind]
    cat_directories = pointing_directories

    # Make sure the file-endings are fits files
    if not mosaic_name.endswith(".fits"):
        mosaic_name += ".fits"
    if not rms_name.endswith(".fits"):
        rms_name += ".fits"
    if not cat_name.endswith(".fits"):
        cat_name += ".fits"
    p_names, p_dirs, p_cats, p_rms = [], [], [], []
    for pd, pn, cat_dir in zip(pointing_directories, pointing_names, cat_directories):
        mp = os.path.join(pd, mosaic_name)
        rp = os.path.join(pd, rms_name)
        cp = os.path.join(cat_dir, cat_name)
        if os.path.exists(mp) and os.path.exists(rp) and os.path.exists(cp):
            p_names.append(pn)
            p_dirs.append(pd)
            p_cats.append(cp)
            p_rms.append(rp)
    print(
        f"{remote_data_directory} contains {len(p_names)} fields with corresponding rms_map and catalogue."
    )

    return p_names, p_dirs, p_cats, p_rms


def create_local_output_folders(
    local_data_directory,
    remote_data_directory,
    remote_field_directories,
    remote_catalogue_paths,
):
    """LoTSS-DR2 fits files are located remotely, this function serves to
    write our own output to local datadirectories with identical directory names"""
    local_0h = os.path.join(local_data_directory, "RA0h_field")
    local_13h = os.path.join(local_data_directory, "RA13h_field")
    os.makedirs(local_0h, exist_ok=True)
    os.makedirs(local_13h, exist_ok=True)

    local_dirs = [
        r.replace(remote_data_directory, local_data_directory)
        for r in remote_field_directories
    ]
    [os.makedirs(l, exist_ok=True) for l in local_dirs]
    local_cat_paths = [
        r.replace(remote_data_directory, local_data_directory).replace(".fits", ".h5")
        for r in remote_catalogue_paths
    ]

    return local_dirs, local_cat_paths


def filter_out_sources_that_are_closer_to_other_pointings(
    dataframe, data_dir, overwrite=False
):
    """Using a pandas dataframe dat contains paths to the mosaic, its rms map, its catalogue,
    and its closest neighbours. Will filter and save catalogue"""

    catalogue_source_count = {}
    # For each pointing
    for i, df in dataframe.iterrows():
        if (not os.path.exists(df.cat_path)) or overwrite:
            # See if distance to its central coordinate is >1 degree then check if any neighbouring
            # pointings are closer than the current pointing. if so, break/drop from table.
            p_cat_all = pd.read_hdf(df.old_cat_path, "df")
            p_cat_all["idx_all"] = p_cat_all.index
            # p_cat_all['skycoord'] = SkyCoord(p_cat_all.RA, p_cat_all.DEC, unit="degree")
            skycoord = SkyCoord(p_cat_all.RA, p_cat_all.DEC, unit="degree")
            p_cat_all["d2center"] = df["central_skycoord"].separation(skycoord)
            p_cat_sub = p_cat_all[p_cat_all["d2center"] > 1.3 * u.degree]
            # print('Len cat of objects farther than 1.3 degree from the center:',len(p_cat_sub))
            if len(df.neighbouring_central_skycoords.tolist()) > 0:
                mask = list(
                    map(
                        lambda x, d2center: any(
                            x.separation(
                                SkyCoord(df.neighbouring_central_skycoords.tolist())
                            ).degree
                            < d2center
                        ),
                        skycoord,
                        p_cat_sub.d2center,
                    )
                )
                index_to_drop = p_cat_sub[mask]["idx_all"]

                p_cat = p_cat_all.drop(index_to_drop)
                catalogue_source_count[df.pointing_name] = (len(p_cat_all), len(p_cat))
                print(
                    f"{i+1}/{len(dataframe)}, {df.pointing_name}: Len cat at start: {len(p_cat_all)},"
                    f"filtered out {len(p_cat_all)-len(p_cat)}"
                )
            else:
                print(
                    f"{i+1}/{len(dataframe)}, {df.pointing_name}: Len cat at start: {len(p_cat_all)},"
                    f"filtered out 0. No neighbouring pointings available."
                )
                p_cat = p_cat_all
            # Write new catalog to hdf
            p_cat.to_hdf(df.cat_path, "df")

    # Save the number of sources contained in the old catalogues and in the new filtered catalogues
    # The old catalogue should contain more sources in total as a number of sources overlap
    # (are present in multiple pointings)
    dataframe_cat_source_count = pd.DataFrame(catalogue_source_count)
    if not os.path.exists(os.path.join(data_dir, "source_count.pkl")) or overwrite:
        dataframe_cat_source_count.to_pickle(os.path.join(data_dir, "source_count.pkl"))
    return dataframe_cat_source_count


def load_fits(fits_filepath, dimensions_normal=True):
    """Load a fits file and return its header and content"""
    # Load first fits file
    # hdulist = fits.open(fits_filepath)
    with fits.open(fits_filepath) as hdulist:
        # Header
        hdr = hdulist[0].header
        if dimensions_normal:
            hdu = hdulist[0].data
        else:
            hdu = hdulist[0].data[0, 0]
    # hdulist.close()
    return hdu, hdr


def get_neighbouring_central_skycoords(
    dataframe,
    separation_in_degree,
    save_directory="",
    save_name="neighbouring_central_skycoords.npy",
    overwrite=False,
):
    """Given a list of skycoords, will return for each skycoord a list of other skycoords lie within,
    the separation_in_degree threshold."""
    if overwrite or not os.path.exists(os.path.join(save_directory, save_name)):

        neighbours = []
        for i, skycoord in enumerate(dataframe.central_skycoord):
            # Remove the considered pointing from the rest of the list
            search_list = SkyCoord(
                dataframe.central_skycoord[dataframe.index != i].tolist()
            )
            neighbouring_central_skycoords = dataframe.central_skycoord[
                dataframe.index != i
            ]
            # search for neighbours
            d2d = skycoord.separation(search_list)
            mask = d2d < separation_in_degree * u.deg
            neighbours.append(neighbouring_central_skycoords[mask])
        np.save(os.path.join(save_directory, save_name), neighbours)
    else:
        neighbours = np.load(os.path.join(save_directory, save_name))

    return neighbours


def hdr_to_central_RA_DEC(fits_filepath):
    """Takes a FITS file as input, retrieves the central coordinate
    through its hdr and returns its central RA and DEC."""
    # Get Fits header: HDR
    _, hdr = load_fits(fits_filepath)
    # Get central ra and dec
    ra, dec = hdr["CRVAl1"], hdr["CRVAL2"]
    return ra, dec


def hdr_to_central_skycoord(fits_filepath):
    """Takes a FITS file as input, retrieves the central coordinate
    through its hdr and returns a skycoord object."""
    ra, dec = hdr_to_central_RA_DEC(fits_filepath)

    return SkyCoord([ra], [dec], unit="deg")[0]


def hdrs_to_central_skycoords(fits_filepaths):
    """Take a FITS files as input, retrieves the central coordinate
    through its hdr and returns a skycoord object."""
    ras, decs = [], []
    for fits_filepath in fits_filepaths:
        ra, dec = hdr_to_central_RA_DEC(fits_filepath)
        # Get central ra and dec
        ras.append(ra)
        decs.append(dec)

    return SkyCoord(ras, decs, unit="deg")


def create_truncated_directories_dictionary(data_directory):
    # Create dictionary to translate truncated pointing names in the catalogue to existing paths
    directory_names = get_immediate_subdirectories(data_directory)
    pointing_dict = {directory[:8]: directory for directory in directory_names}
    for directory in directory_names:
        if not (directory in pointing_dict):
            pointing_dict[directory] = directory
    return {**pointing_dict, **{directory: directory for directory in directory_names}}


def ellipse(x0, y0, a, b, pa, n=200):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    st = np.sin(theta)
    ct = np.cos(theta)
    pa = np.deg2rad(pa + 90)
    sa = np.sin(pa)
    ca = np.cos(pa)
    p = np.empty((n, 2))
    p[:, 0] = x0 + a * ca * ct - b * sa * st
    p[:, 1] = y0 + a * sa * ct + b * ca * st
    return Polygon(p)


class Make_Shape(object):
    """
    Before being altered slightly the code for this class was taken from the process_lgz.py code written by Martin Hardcastle.

    Basic idea taken from remove_lgz_sources.py -- maybe should be merged with this one day
    but the FITS keywords are different.
    """

    def __init__(self, clist):
        """
        clist: a list of components that form part of the source, with RA, DEC, DC_Maj...
        """

        from shapely.geometry import Polygon
        from shapely.ops import cascaded_union

        ra = np.mean(clist["RA"])
        dec = np.mean(clist["DEC"])

        ellist = []
        for i in range(len(clist)):
            # Import the RA and DEC of the components
            n_ra, n_dec = clist.iloc[i]["RA"], clist.iloc[i]["DEC"]
            # Calculate the central coordinates of the ellipses relative to the mean
            x = 3600 * np.cos(dec * np.pi / 180.0) * (ra - n_ra)
            # x=3600*np.cos(dec*np.pi/180.0)*(n_ra-ra)
            y = 3600 * (n_dec - dec)
            # Form the ellipses with relative coordinates
            newp = ellipse(
                x,
                y,
                clist.iloc[i]["DC_Maj"] + 0.1,
                clist.iloc[i]["DC_Min"] + 0.1,
                clist.iloc[i]["PA"],
            )
            ellist.append(newp)

        self.cp = cascaded_union(ellist)
        self.ra = ra
        self.dec = dec
        self.h = self.cp.convex_hull
        a = np.asarray(self.h.exterior.coords)

        # for i,e in enumerate(ellist):
        #    if i==0:
        #        a=np.asarray(e.exterior.coords)
        #    else:
        #        a=np.append(a,e.exterior.coords,axis=0)
        mdist2 = 0
        bestcoords = None

        # Here the coordinates of the two points that are furthest appart from each other are selected
        for r in a:
            dist2 = (a[:, 0] - r[0]) ** 2.0 + (a[:, 1] - r[1]) ** 2.0
            idist = np.argmax(dist2)
            mdist = dist2[idist]
            if mdist > mdist2:
                mdist2 = mdist
                bestcoords = (r, a[idist])
        self.mdist2 = mdist2
        self.bestcoords = bestcoords
        self.a = a
        self.best_a = 0

    def length(self):
        # Returns distance between the two 'best coordinates'
        return np.sqrt(self.mdist2)

    def pa(self):
        # Returns the north to east angle of the convex_hull main axis
        p1, p2 = self.bestcoords
        dp = p2 - p1
        angle = (180 * np.arctan2(dp[1], dp[0]) / np.pi) - 90
        if angle < -180:
            angle += 360
        if angle < 0:
            angle += 180
        return angle

    def width(self):
        # Returns the maximum distance from the length vector to the outer edge of the hull
        p1, p2 = self.bestcoords
        print(p1, p2)
        d = np.cross(p2 - p1, self.a - p1) / self.length()
        print(self.a.shape, d.shape)
        self.best_a = self.a[np.argmax(d)]
        return 2 * np.max(d)

    def geo_center(self):
        # Returns the geometric center of the convex hull
        w, pa = self.width(), self.pa()
        dx, dy = w / 2.0 * np.cos((np.pi / 180) * (pa)), w / 2.0 * np.sin(
            (np.pi / 180) * (pa)
        )
        print("dx,dy:", dx, dy)
        ax, ay = self.best_a
        print("best_a", self.best_a)
        # print self.a
        return self.ra + (
            (ax + dx) / (3600 * np.cos(self.dec * np.pi / 180.0))
        ), self.dec + ((ay + dy) / 3600)

    def hull_box(self):
        # Returns the x,y coordinates of the box enclosing the convexl hull
        # print self.a
        # return self.ra+((self.a[:,0])/(3600*np.cos(self.dec*np.pi/180.0))),self.dec+((self.a[:,1])/3600)
        return self.a


def fits_to_rms_using_astropy(
    fits_filename,
    store_filename,
    fullsize,
    pandas_catalogue,
    mosaic_id_key,
    ra_key,
    dec_key,
    data_directory,
    output_directory=None,
    verbose=True,
    single_field=False,
    single_field_path=None,
    data_directories_truncated_in_catalog=True,
    variable_size=False,
    arcsec_per_pixel=1.5,
    mode="partial",
    sort=True,
    dimensions_normal=True,
    overwrite=False,
    Okke=False,
):
    """Use this function to go from a single catalogue with sources coming from multiple fits files,
    located in a pointing directories with the name of the pointing directly under the
    datadirectory,
    , and a corresponding pandas catalogue with the RA and DEC coordinates to its objects,
    to get cutouts  of these objects and save per cutout the rms median."""

    start = time.time()
    print(f"Retrieving mean rms of sources")
    if Okke:
        data_directory_2 = output_directory
    else:
        data_directory_2 = data_directory
    if overwrite or not os.path.exists(
        os.path.join(data_directory_2, store_filename + ".npy")
    ):

        # Create dictionary to translate truncated pointing names in the catalogue to existing paths
        pointing_dict = create_truncated_directories_dictionary(data_directory)
        # Manual hack to correct for the namechange of the field P41 from DR1 to DR2
        pointing_dict["P41Hetde"] = "P5Hetdex41"
        pointing_dict["P41Hetdex"] = "P5Hetdex41"

        # Initialize rms_list and fail counters and mosaic path
        rms_list, count = [], 0
        current_pointing_name = "blanco"
        if not fits_filename.endswith(".fits"):
            fits_filename += ".fits"

        # Sort catalogue on origin mosaic (in order to load each fits file only once)
        if sort:
            pandas_catalogue = pandas_catalogue.sort_values(by=[mosaic_id_key])
            print(
                "Catalogue sorted on origin mosaic, NOTE: this means the order changed!"
            )

        # Get skycoords of all sources in the pointing
        skycoords = pandas_catalogue_to_astropy_coords(
            pandas_catalogue, ra_key, dec_key
        )

        # Create cutout and calculate rms for each skycoord
        for i, ((i_p, source), co) in enumerate(
            zip(pandas_catalogue.iterrows(), skycoords)
        ):

            # Load fits file if source is in a different fits file than the one currently opened
            if single_field:
                hdu, hdr = load_fits(
                    single_field_path, dimensions_normal=dimensions_normal
                )
            else:
                if current_pointing_name != source[mosaic_id_key]:
                    new_pointing_path = os.path.join(
                        data_directory,
                        pointing_dict[source[mosaic_id_key]],
                        fits_filename,
                    )
                    hdu, hdr = load_fits(
                        new_pointing_path, dimensions_normal=dimensions_normal
                    )
                    current_pointing_name = source[mosaic_id_key]

            if variable_size:
                try:
                    # Where 1.5 is the amount of arcsec per pixel
                    fullsize = int(np.ceil(2 * source.source_size / arcsec_per_pixel))
                except:
                    # Where 1.5 is the amount of arcsec per pixel
                    fullsize = int(np.ceil(2 * source.LGZ_Size / arcsec_per_pixel))
            hdu_crop = Cutout2D(
                hdu,
                co,
                (fullsize, fullsize),
                wcs=WCS(hdr, naxis=2),
                copy=True,
                mode=mode,
            )
            a, b = np.shape(hdu_crop.data)
            if a * b == fullsize * fullsize:
                image = hdu_crop.data
                rms_list.append(np.nanmedian(image))
            else:
                # If you end up here that probably means some of skycoords
                # were located too close to the border of the fits-file for cutout2D to create a
                #  (fullsize * fullsize)-sized box around them.
                rms_list.append(None)
                count += 1
                if verbose:
                    print(
                        f"{count}th failure, {source.Source_Name} shape is {a}x{b} and should be {fullsize}x{fullsize}"
                    )

            # Progress notification
            if i % 1000 == 0 and i > 0:
                time_elapsed = time.time() - start
                print(
                    i,
                    "/",
                    len(pandas_catalogue),
                    "Time elapsed:",
                    int(time_elapsed),
                    "seconds. Estimated time left:",
                    round(
                        (time_elapsed * len(pandas_catalogue) / i - time_elapsed)
                        / 60.0,
                        2,
                    ),
                    " minutes.",
                )

        # Save rms_list and add it to the pandas catalogue
        np.save(os.path.join(data_directory_2, store_filename), np.array(rms_list))
    else:
        print(
            "Numpy store_filename containing cutouts already exists. Loading it in..."
        )
        rms_list = np.load(os.path.join(data_directory_2, store_filename + ".npy"))
    # Note that None objects will be converted into NaNs bij pandas
    pandas_catalogue["cutout_rms"] = rms_list
    print("Time elapsed:", int(time.time() - start), "seconds.")
    return pandas_catalogue, rms_list


def plot_data_model_residual(
    data,
    model,
    residual,
    stretch=vis.AsinhStretch,
    model_angle=0,
    dra_deg=None,
    ddec_deg=None,
    cal_dec=None,
    cal_ra=None,
    convex_hull=None,
    arcsec_per_pixel_RA=1.5,
    arcsec_per_pixel_DEC=1.5,
    pixel_locs=None,
):
    # Plot the data, the model and the residual image
    fig = plt.figure(figsize=(15, 5))

    grid = ImageGrid(
        fig,
        111,  # as in plt.subplot(111)
        nrows_ncols=(1, 3),
        axes_pad=0.15,
        # share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="7%",
        cbar_pad=0.15,
    )

    norm = vis.ImageNormalize(data, stretch=vis.AsinhStretch())
    grid[0].imshow(
        data,
        norm=norm,
        origin="lower",
        interpolation="nearest",
        vmin=np.min(data),
        vmax=np.max(data),
    )
    grid[0].set_title("Data")
    for g in grid:
        g.grid(False)
        g.plot(data.shape[0] / 2, data.shape[1] / 2, "w+")
    if not dra_deg is None:
        dra = dra_deg * 3600 / arcsec_per_pixel_RA
        ddec = ddec_deg * 3600 / arcsec_per_pixel_DEC

        grid[0].add_patch(
            Rectangle(
                (model.shape[0] / 2 - dra, model.shape[0] / 2 - ddec),
                dra * 2,
                ddec * 2,
                linewidth=1,
                edgecolor="w",
                facecolor="none",
            )
        )

    norm = vis.ImageNormalize(model, stretch=vis.AsinhStretch())
    grid[1].imshow(
        model,
        norm=norm,
        interpolation="nearest",
        origin="lower",
        vmin=np.min(data),
        vmax=np.max(data),
    )
    if not convex_hull is None:
        grid[0].plot(
            np.array(convex_hull[0]) + model.shape[0] / 2,
            np.array(convex_hull[1]) + model.shape[1] / 2,
            "lightblue",
        )
        grid[1].plot(
            np.array(convex_hull[0]) + model.shape[0] / 2,
            np.array(convex_hull[1]) + model.shape[1] / 2,
            "lightblue",
        )
    grid[1].set_xlim([0, data.shape[0]])
    grid[1].set_ylim([0, data.shape[1]])

    if not dra_deg is None:
        dra = dra_deg * 3600 / arcsec_per_pixel_RA
        ddec = ddec_deg * 3600 / arcsec_per_pixel_DEC

        grid[1].add_patch(
            Rectangle(
                (model.shape[0] / 2 - dra, model.shape[0] / 2 - ddec),
                dra * 2,
                ddec * 2,
                linewidth=1,
                edgecolor="w",
                facecolor="none",
            )
        )
        x = np.linspace(0, 50, 5 / 1.5)
        grid[1].plot(pixel_locs[0], pixel_locs[1], "go")

    grid[1].set_title("Model")
    # plt.colorbar()
    norm = vis.ImageNormalize(residual, stretch=vis.AsinhStretch())
    im = grid[2].imshow(
        residual,
        norm=norm,
        origin="lower",
        interpolation="nearest",
        vmin=np.min(data),
        vmax=np.max(data),
    )
    grid[2].set_title("Residual")
    # plt.colorbar()
    # Colorbar
    grid[2].cax.colorbar(im)  # , label='mJy')
    # plt.colorbar(cax=cbaxes, orientation='horizontal', label='mJy')
    # grid[2].cax.toggle_label(True)
    grid[2].cax.set_label("mJy")
    plt.show()


def return_cutout_pixel_resolution(image, wcs, verbose=False):

    # Calculate arcsec per pixel for RA
    w, h = image.shape[0], image.shape[1]
    c = pixel_to_skycoord([0, w, 0, w], [0, 0, h, h], wcs)
    width_degree = (c[1].ra.deg - c[0].ra.deg + c[3].ra.deg - c[2].ra.deg) * 0.5
    width_arcsec = width_degree * 3600
    # print(np.shape(image))
    # print(image)
    # print(width_arcsec)
    arcsec_per_pixel_RA = width_arcsec / w
    # Calculate arcsec per pixel for DEC
    c = pixel_to_skycoord([0, 0, w, w], [0, h, 0, h], wcs)
    height_degree = (c[1].dec.deg - c[0].dec.deg + c[3].dec.deg - c[2].dec.deg) * 0.5
    height_arcsec = height_degree * 3600
    arcsec_per_pixel_DEC = height_arcsec / h
    if verbose:
        print(
            f"Image widthxheight is {width_arcsec}x{height_arcsec} arcsec, and the RA, DEC resolution is:\n"
            f"{arcsec_per_pixel_RA},{arcsec_per_pixel_DEC} arcsec per pixel"
        )
    return abs(arcsec_per_pixel_RA), abs(arcsec_per_pixel_DEC)


def return_local_rms(
    image, cutout_wcs, source, source_cat, use_source_size=True, debug=True
):
    image = copy.deepcopy(image)

    # retrieve pixel resolution as in practice it often varies between cutouts,
    # especially the RA can have less pixels representing a fixed set of arcseconds
    arcsec_per_pixel_RA, arcsec_per_pixel_DEC = return_cutout_pixel_resolution(
        image, cutout_wcs, verbose=False
    )

    # create subset of gaussian list
    ra, dec = source.RA, source.DEC

    sr = image.shape[0] * arcsec_per_pixel_RA / 3600
    sd = image.shape[1] * arcsec_per_pixel_DEC / 3600

    local_source_cat = source_cat[
        (source_cat.DEC > dec - sd)
        & (source_cat.DEC < dec + sd)
        & (source_cat.RA > ra - sr)
        & (source_cat.RA < ra + sr)
    ]

    # Create gaussians
    gaussians = extract_gaussian_parameters_from_component_catalogue(
        local_source_cat,
        cutout_wcs,
        use_source_size=use_source_size,
        arcsec_per_pixel=(arcsec_per_pixel_RA + arcsec_per_pixel_DEC) / 2,
    )

    # Subtract them from the data
    model, residual = mask_gaussians_from_data(gaussians, image)
    # Calculate root mean square noise
    noise_sum = np.count_nonzero(~np.isnan(residual))
    if noise_sum != 0:
        rms = np.sqrt(np.nansum(np.square(residual)) / noise_sum)
    else:
        rms = np.nan

    # DEBUG
    if debug:
        print(
            f"Local rms: {rms:.2g}. RA and DEC resolution: {arcsec_per_pixel_RA:.2f} "
            f"{arcsec_per_pixel_DEC:.2f}"
        )  # PyBDSF local rms: {source.cutout_rms:.2g}")
        plt.imshow(residual)
        plt.title("masked image")
        plt.colorbar()
        plt.show()
    return rms


def rough_remove_neighbouring_sources_from_cutout(
    image, cutout_wcs, source, cat, use_source_size=True, debug=False
):
    """If no component and gaussian catalogue exist, just
    remove the entire neighbouring sources.
    A first order approach that might still do a decent job at removing
    unresolved sources.

    params:
        cat = a source catalogue in the pandas dataframe format"""

    # retrieve pixel resolution as in practice it often varies between cutouts,
    # especially the RA can have less pixels representing a fixed set of arcseconds
    arcsec_per_pixel_RA, arcsec_per_pixel_DEC = return_cutout_pixel_resolution(
        image, cutout_wcs, verbose=False
    )
    if debug:
        print("angular res, ra, dec:", arcsec_per_pixel_RA, arcsec_per_pixel_DEC)

    # create subset of gaussian list
    ra, dec = source.RA, source.DEC

    sr = image.shape[0] * arcsec_per_pixel_RA / 3600
    sd = image.shape[1] * arcsec_per_pixel_DEC / 3600

    local_cat = cat[
        (cat.DEC > dec - sd)
        & (cat.DEC < dec + sd)
        & (cat.RA > ra - sr)
        & (cat.RA < ra + sr)
    ]
    local_cat = local_cat[source.Source_Name != local_cat.Source_Name]
    if debug:
        print("Rough neighbour removal found this many neighbours:", len(local_cat))

    # Create gaussians
    gaussians = extract_gaussian_parameters_from_component_catalogue(
        local_cat, cutout_wcs, use_source_size=use_source_size
    )
    # Subtract them from the data
    model, residual = subtract_gaussians_from_data(gaussians, image)
    # DEBUG
    debug = False
    if debug:
        dra, ddec = return_tight_box_around_source(
            source.LGZ_Size / 3600, source.LGZ_Width / 3600, source.LGZ_PA
        )
        dra *= arcsec_per_pixel_RA / arcsec_per_pixel_DEC
        c = SkyCoord(local_cat.RA, local_cat.DEC, unit="deg")
        pixel_locs = skycoord_to_pixel(c, cutout_wcs, origin=0, mode="all")

        # gaus_list = gaus_cat[gaus_cat.Source_Name.isin(component_names)]
        convex_hull = Make_Shape(local_cat)
        convex_hull_xs, convex_hull_ys = zip(*convex_hull.hull_box())

        plot_data_model_residual(
            image,
            model,
            residual,
            dra_deg=dra,
            ddec_deg=ddec,
            pixel_locs=pixel_locs,
            convex_hull=[convex_hull_xs, convex_hull_ys],
            arcsec_per_pixel_RA=arcsec_per_pixel_RA,
            arcsec_per_pixel_DEC=arcsec_per_pixel_DEC,
        )
    return residual


def remove_neighbouring_sources_from_cutout(
    image, cutout_wcs, source, component_cat, gaus_cat, debug=False, deepfield=False
):

    # retrieve pixel resolution as in practice it often varies between cutouts,
    # especially the RA can have less pixels representing a fixed set of arcseconds
    arcsec_per_pixel_RA, arcsec_per_pixel_DEC = return_cutout_pixel_resolution(
        image, cutout_wcs, verbose=False
    )

    # Get list of gaussians belonging to our source
    if deepfield:
        component_names = component_cat[
            component_cat.Parent_Source == source.Source_Name
        ].Component_Name.values
    else:
        component_names = component_cat[
            component_cat.Source_Name == source.Source_Name
        ].Component_Name.values

    # create subset of gaussian list
    ra, dec = source.RA, source.DEC

    sr = image.shape[0] * arcsec_per_pixel_RA / 3600
    sd = image.shape[1] * arcsec_per_pixel_DEC / 3600

    if deepfield:
        local_gaus_cat = gaus_cat[
            (gaus_cat.DEC > dec - sd)
            & (gaus_cat.DEC < dec + sd)
            & (gaus_cat.RA > ra - sr)
            & (gaus_cat.RA < ra + sr)
            & ~gaus_cat.Parent_source.isin(component_names)
        ]
    else:
        local_gaus_cat = gaus_cat[
            (gaus_cat.DEC > dec - sd)
            & (gaus_cat.DEC < dec + sd)
            & (gaus_cat.RA > ra - sr)
            & (gaus_cat.RA < ra + sr)
            & ~gaus_cat.Source_Name.isin(component_names)
        ]
        # ((gaus_cat.DEC > dec + ddec) | (gaus_cat.DEC < dec - ddec) |
        # (gaus_cat.RA > ra + dra) | (gaus_cat.RA < ra - dra))]

    # Create gaussians
    gaussians = extract_gaussian_parameters_from_component_catalogue(
        local_gaus_cat, cutout_wcs
    )
    # Subtract them from the data
    model, residual = subtract_gaussians_from_data(gaussians, image)

    # DEBUG
    if debug:
        dra, ddec = return_tight_box_around_source(
            source.LGZ_Size / 3600, source.LGZ_Width / 3600, source.LGZ_PA
        )
        dra *= arcsec_per_pixel_RA / arcsec_per_pixel_DEC
        c = SkyCoord(local_gaus_cat.RA, local_gaus_cat.DEC, unit="deg")
        pixel_locs = skycoord_to_pixel(c, cutout_wcs, origin=0, mode="all")

        gaus_list = gaus_cat[gaus_cat.Source_Name.isin(component_names)]
        convex_hull = Make_Shape(gaus_list)
        convex_hull_xs, convex_hull_ys = zip(*convex_hull.hull_box())

        plot_data_model_residual(
            image,
            model,
            residual,
            dra_deg=dra,
            ddec_deg=ddec,
            pixel_locs=pixel_locs,
            convex_hull=[convex_hull_xs, convex_hull_ys],
            arcsec_per_pixel_RA=arcsec_per_pixel_RA,
            arcsec_per_pixel_DEC=arcsec_per_pixel_DEC,
        )
    return residual


def pixelvalues_within_radius(image, radius, debug=False):
    """Given an image, determine the summed pixelvalue within a circle
    of radius r."""
    assert len(np.shape(image)) == 2
    width, height = np.shape(image)
    assert radius <= width and radius <= height
    image_copy = copy.deepcopy(image)
    initial_sum = np.sum(image_copy)
    a, b = int(height / 2), int(width / 2)

    y, x = np.ogrid[-a : int(height) - a, -b : int(width) - b]
    mask = x * x + y * y > radius * radius

    if debug:
        # debug plot
        image_copy[mask] = 2
        plt.figure()
        plt.imshow(image_copy, origin="lower", cmap="viridis")
        plt.grid(False)
        plt.show()

    image_copy[mask] = 0
    return np.sum(image_copy) / initial_sum


def fits_to_cutouts_using_astropy_given_paths(
    fits_paths,
    path_indices,
    store_filename,
    fullsize,
    skycoords,
    pandas_catalogue,
    data_directory,
    lower_sigma_limit=1.5,
    upper_sigma_limit=1e9,
    output_directory=None,
    verbose=True,
    normed=False,
    rescale=True,
    apply_clipping=True,
    apply_mask=True,
    variable_size=False,
    destination_size=None,
    asinh=False,
    gaus_cat=None,
    remove_neighbours=False,
    rough_remove_neighbours=False,
    full_cat=None,
    single_field=False,
    deepfield=False,
    single_field_path=None,
    rely_on_catalogue_size=True,
    get_local_rms=False,
    component_cat=None,
    only_calculate_median_and_max=False,
    maj_multiplier=1.8,
    LGZ_multiplier=1.5,
    skip_empty_cutouts=True,
    debug=False,
    dimensions_normal=True,
    arcsec_per_pixel=1.5,
    overwrite=False,
    using_remote_data=False,
):
    """
    Use this function to go from a list of fits_paths and corresponding skycoords
    datadirectory,
    , and a corresponding pandas catalogue with the RA and DEC coordinates to its objects,
    to get cutouts  of these objects and save per cutout the rms mean."""
    pd.options.mode.chained_assignment = None
    if using_remote_data:
        data_directory_2 = output_directory
    else:
        data_directory_2 = data_directory
    if get_local_rms or remove_neighbours or rough_remove_neighbours:
        from shapely.geometry import Polygon
    start = time.time()
    if (
        overwrite
        or not os.path.exists(os.path.join(data_directory_2, store_filename + ".npy"))
        or not os.path.exists(
            os.path.join(data_directory_2, "catalogue_" + store_filename + ".h5")
        )
    ):

        # input sanity check
        if remove_neighbours:
            assert not component_cat is None
            assert not gaus_cat is None
        if rough_remove_neighbours:
            assert (
                not full_cat is None
            ), "If you applied no cut to your original cat you can insert that one"
        if get_local_rms:
            assert (
                not full_cat is None
            ), "If you applied no cut to your original cat you can insert that one"
            local_rms = []

        # Initialize rms_list and fail counters and mosaic path
        image_list, extraction_succeeded_list = [], []
        minmax_interval = vis.MinMaxInterval()
        rms_list, max_median_list, count = [], [], 0
        current_pointing_name = "blanco"
        if apply_mask:
            if variable_size and (not destination_size is None):
                mask = create_circular_mask(destination_size)
            else:
                mask = create_circular_mask(fullsize)

        # Create cutout and calculate rms for each skycoord
        print(f"Catalogue initially contains {len(pandas_catalogue)} entries.")
        for i, ((i_p, source), p, co) in enumerate(
            zip(pandas_catalogue.iterrows(), path_indices, skycoords)
        ):

            # Load fits file if source is in a different fits file than the one currently opened
            if p > -1:
                hdu, hdr = load_fits(fits_paths[p], dimensions_normal=dimensions_normal)
                # ctrl_ra = hdr['CRVAL1']
                # ctrl_dec = hdr['CRVAL2']
                # ctrl_skycoord = SkyCoord(ra=ctrl_ra, dec=ctrl_dec,unit='deg',frame='fk5')
            else:
                extraction_succeeded_list.append(False)
                count += 1
                if get_local_rms:
                    local_rms.append(np.nan)
                continue

            if variable_size:
                try:
                    # Where 1.5 is the amount of arcsec per pixel
                    fullsize = int(
                        np.ceil(source.source_size * np.sqrt(2) / arcsec_per_pixel)
                    )
                    catsize = source.source_size / arcsec_per_pixel
                except:
                    # Where 1.5 is the amount of arcsec per pixel
                    fullsize = int(
                        np.ceil(source.LGZ_Size * np.sqrt(2) / arcsec_per_pixel)
                    )
                    catsize = source.LGZ_Size / arcsec_per_pixel
                # if not rely_on_catalogue_size:
                if np.isnan(source.LGZ_Size):
                    # PyBDSF generated Major axis size is consistently smaller than LGZ convex hull
                    # size
                    fullsize = int(fullsize * maj_multiplier)
                else:
                    fullsize = int(fullsize * LGZ_multiplier)

            # Extract cutout
            try:
                hdu_crop = Cutout2D(
                    hdu,
                    co,
                    (fullsize, fullsize),
                    wcs=WCS(hdr, naxis=2),
                    copy=True,
                    mode="strict",
                )
            except (NoOverlapError, PartialOverlapError) as e:
                extraction_succeeded_list.append(False)
                count += 1
                if get_local_rms:
                    local_rms.append(np.nan)
                continue

            if not np.isnan(hdu_crop.data).any():
                image = hdu_crop.data

                # Get local image rms by masking all sources in view and returning the median signal
                if get_local_rms:
                    rms = np.nan
                    f = fullsize
                    while np.isnan(rms):
                        f *= 1.5
                        larger_hdu_crop = Cutout2D(
                            hdu,
                            co,
                            (f, f),
                            wcs=WCS(hdr, naxis=2),
                            copy=True,
                            mode="partial",
                            fill_value=np.nan,
                        )
                        rms = return_local_rms(
                            copy.deepcopy(larger_hdu_crop.data),
                            larger_hdu_crop.wcs,
                            source,
                            full_cat,
                            debug=False,
                        )
                    local_rms.append(rms)

                # Remove neighbouring objects
                if remove_neighbours:
                    image = remove_neighbouring_sources_from_cutout(
                        image,
                        hdu_crop.wcs,
                        source,
                        component_cat,
                        gaus_cat,
                        debug=False,
                        deepfield=deepfield,
                    )

                # If no component and gaussian catalogue exists, try removing entire sources
                # (inferior option to the regular remove_neighbour function)
                if rough_remove_neighbours:
                    image = rough_remove_neighbouring_sources_from_cutout(
                        image, hdu_crop.wcs, source, full_cat
                    )

                # Calculate median and max values of the image
                if only_calculate_median_and_max:
                    if variable_size:
                        mask = create_circular_mask(fullsize)
                    masked_image = np.ma.array(
                        image,
                        mask=(image < lower_sigma_limit * source.cutout_rms)
                        | (image > upper_sigma_limit * source.cutout_rms),
                    )
                    masked_image[mask] = np.ma.masked

                    if masked_image.mask.all():
                        if verbose:
                            print(
                                f"{source.Source_Name} has no intensity > {lower_sigma_limit} the local noise"
                            )
                        max_median_list.append((i_p, None, None))
                        continue

                    max_median_list.append(
                        (i_p, np.ma.max(masked_image), np.ma.median(masked_image))
                    )

                    # Progress notification
                    if i % 1000 == 0 and i > 0:
                        time_elapsed = time.time() - start
                        print(
                            i,
                            "/",
                            len(pandas_catalogue),
                            "Time elapsed:",
                            int(time_elapsed),
                            "seconds. Estimated time left:",
                            round(
                                (
                                    time_elapsed * len(pandas_catalogue) / i
                                    - time_elapsed
                                )
                                / 60.0,
                                2,
                            ),
                            " minutes.",
                        )
                    continue

                # Resize image based on flux contained in radius
                if not rely_on_catalogue_size:
                    # """
                    radii = np.linspace(3, fullsize / 2, 20)
                    new_radius = 0
                    # Determine 95% flux containing radius
                    f0 = pixelvalues_within_radius(image, fullsize, debug=False)
                    for radius in radii:
                        f = pixelvalues_within_radius(image, radius, debug=False)
                        if f > 0.99:
                            new_radius = radius
                            break
                    rotated_size = fullsize / np.sqrt(2)
                    if new_radius < catsize / 3:
                        new_radius = catsize
                    # Crop image
                    w = max(int((fullsize - new_radius * 1.5) / 2), 0)
                    image = image[w : fullsize - w, w : fullsize - w]

                # Rescale to destination size for variable cutout
                if variable_size:
                    image = np.array(
                        Image.fromarray(image).resize(
                            (destination_size, destination_size),
                            resample=Image.BILINEAR,
                        )
                    )
                    fullsize = destination_size

                if apply_clipping:
                    if get_local_rms:
                        if not np.isnan(rms):
                            # print('RMS', rms)
                            image = np.clip(
                                image, lower_sigma_limit * rms, upper_sigma_limit * rms
                            )
                    else:
                        image = np.clip(
                            image,
                            lower_sigma_limit * source.cutout_rms,
                            upper_sigma_limit * source.cutout_rms,
                        )
                if apply_mask:
                    if rescale:
                        image[mask] = np.min(image)
                    else:
                        image[mask] = 0

                if normed:
                    image_sum = np.sum(image)
                    if not image_sum == 0:
                        image = image / image_sum
                else:
                    # Scale from [0,1] using minmax
                    if rescale:
                        image = minmax_interval(image)
                if asinh:
                    assert rescale == False
                    assert normed == False
                    transform = vis.AsinhStretch() + vis.PercentileInterval(99.5)
                    image = transform(image)
                if apply_clipping and rescale:
                    if np.min(image) == np.max(image) or np.sum(image) == 0:
                        if skip_empty_cutouts:
                            print(
                                f"Cutout at RA DEC {source.RA:.4f} {source.DEC:.4f} empty!"
                            )
                            count += 1
                            if verbose:
                                print(
                                    count,
                                    "failure. Apply clipping and rescaling is True. But min==max in the iamge or the sum of the image is zero",
                                )
                                print(
                                    f"{source.Source_Name} has no intensity > {lower_sigma_limit} the local noise"
                                )
                            # Debug stuff below
                            # print(i, "min equals max:", np.min(image) == np.max(image), "min and max are:", \
                            #        np.min(image), np.max(image), "sum is:", np.sum(image))
                            extraction_succeeded_list.append(False)
                            print(
                                "Cutout skipped because all signal was below the sigmaclip lower limit"
                            )
                            continue

                image_list.append(image)
                extraction_succeeded_list.append(True)

            else:
                # If you end up here that probably means some of skycoords
                # were located too close to the border of the fits-file for cutout2D to create a
                #  (fullsize * fullsize)-sized box around them.
                extraction_succeeded_list.append(False)
                count += 1
                if get_local_rms:
                    local_rms.append(np.nan)
                if verbose:
                    print(count, "failure")
                    if np.isnan(source.cutout_rms):
                        print(
                            f"{count}th failure, {source.Source_Name} does not have a corresponding rms value"
                        )
                    if np.isnan(hdu_crop.data).any():
                        print(
                            f"{count}th failure, {source.Source_Name} contains nan values"
                        )

            # Progress notification
            if i % 100 == 0 and i > 0:
                time_elapsed = time.time() - start
                print(
                    i,
                    "/",
                    len(pandas_catalogue),
                    "Time elapsed:",
                    int(time_elapsed),
                    "seconds. Estimated time left:",
                    round(
                        (time_elapsed * len(pandas_catalogue) / i - time_elapsed)
                        / 60.0,
                        2,
                    ),
                    " minutes.",
                )

        # If only_calculate_median_and_max return just the max and median values
        if only_calculate_median_and_max:
            return max_median_list
        if get_local_rms:
            pandas_catalogue["local_rms"] = local_rms

        # Save image_list and add it to the pandas catalogue
        image_list = np.array(image_list)
        np.save(os.path.join(data_directory_2, store_filename), image_list)
        np.save(f"extraction_succeeded_{store_filename}.npy", extraction_succeeded_list)
        # Save catalogue
        pandas_catalogue = pandas_catalogue[extraction_succeeded_list]
        pandas_catalogue.to_hdf(
            os.path.join(data_directory_2, "catalogue_" + store_filename + ".h5"), "df"
        )
    else:
        print(
            "Numpy store_filename containing cutouts already exists. Loading it in..."
        )
        image_list = np.load(os.path.join(data_directory_2, store_filename + ".npy"))
        pandas_catalogue = pd.read_hdf(
            os.path.join(data_directory_2, "catalogue_" + store_filename + ".h5"), "df"
        )

    print("Time elapsed:", int(time.time() - start), "seconds.")
    print("catalogue_" + store_filename + ".h5")
    pd.options.mode.chained_assignment = "warn"
    return image_list, pandas_catalogue


def fits_to_cutouts_using_astropy(
    fits_filename,
    store_filename,
    fullsize,
    pandas_catalogue,
    mosaic_id_key,
    ra_key,
    dec_key,
    data_directory,
    lower_sigma_limit=1.5,
    upper_sigma_limit=1e9,
    output_directory=None,
    verbose=True,
    normed=False,
    rescale=True,
    apply_clipping=True,
    apply_mask=True,
    variable_size=False,
    destination_size=None,
    asinh=False,
    gaus_cat=None,
    remove_neighbours=False,
    rough_remove_neighbours=False,
    full_cat=None,
    single_field=False,
    deepfield=False,
    single_field_path=None,
    rely_on_catalogue_size=True,
    get_local_rms=False,
    component_cat=None,
    only_calculate_median_and_max=False,
    maj_multiplier=1.8,
    LGZ_multiplier=1.5,
    skip_empty_cutouts=True,
    mode="partial",
    debug=False,
    sort=True,
    dimensions_normal=True,
    arcsec_per_pixel=1.5,
    overwrite=False,
    using_remote_data=False,
):
    """
    This is the version that supersedes the fits_to_numpy_cutouts_using_astropy function.
    Use this function to go from a single catalogue with sources coming from multiple fits files,
    located in a pointing directories with the name of the pointing directly under the
    datadirectory,
    , and a corresponding pandas catalogue with the RA and DEC coordinates to its objects,
    to get cutouts  of these objects and save per cutout the rms mean."""
    pd.options.mode.chained_assignment = None
    if using_remote_data:
        data_directory_2 = output_directory
    else:
        data_directory_2 = data_directory
    if get_local_rms or remove_neighbours or rough_remove_neighbours:
        from shapely.geometry import Polygon
    start = time.time()
    if (
        overwrite
        or not os.path.exists(os.path.join(data_directory_2, store_filename + ".npy"))
        or not os.path.exists(
            os.path.join(data_directory_2, "catalogue_" + store_filename + ".h5")
        )
    ):

        # input sanity check
        if remove_neighbours:
            assert not component_cat is None
            assert not gaus_cat is None
        if rough_remove_neighbours:
            assert (
                not full_cat is None
            ), "If you applied no cut to your original cat you can insert that one"
        if get_local_rms:
            assert (
                not full_cat is None
            ), "If you applied no cut to your original cat you can insert that one"
            local_rms = []

        # Create dictionary to translate truncated pointing names in the catalogue to existing paths
        pointing_dict1 = create_truncated_directories_dictionary(data_directory)
        pointing_dict2 = create_truncated_directories_dictionary(
            data_directory.replace("0h", "13h")
        )
        pointing_dict = {}
        pointing_dict.update(pointing_dict1)
        pointing_dict.update(pointing_dict2)
        print("data_directory", data_directory)
        # Manual hack to correct for the namechange of the field P41 from DR1 to DR2
        pointing_dict["P41Hetde"] = "P5Hetdex41"
        pointing_dict["P41Hetdex"] = "P5Hetdex41"

        # Initialize rms_list and fail counters and mosaic path
        image_list, extraction_succeeded_list = [], []
        minmax_interval = vis.MinMaxInterval()
        rms_list, max_median_list, count = [], [], 0
        current_pointing_name = "blanco"
        if not fits_filename.endswith(".fits"):
            fits_filename += ".fits"
        if apply_mask:
            if variable_size and (not destination_size is None):
                mask = create_circular_mask(destination_size)
            else:
                mask = create_circular_mask(fullsize)

        # Sort catalogue on origin mosaic (in order to load each fits file only once)
        if sort:
            pandas_catalogue = pandas_catalogue.sort_values(by=[mosaic_id_key])
            print(
                "Catalogue sorted on origin mosaic, NOTE: this means the order changed!"
            )

        # Get skycoords of all sources in the pointing
        skycoords = pandas_catalogue_to_astropy_coords(
            pandas_catalogue, ra_key, dec_key
        )

        # Create cutout and calculate rms for each skycoord
        print(f"Catalogue initially contains {len(pandas_catalogue)} entries.")
        for i, ((i_p, source), co) in enumerate(
            zip(pandas_catalogue.iterrows(), skycoords)
        ):

            # if not i in [2606,  716,  527,  611, 2654]:
            #    continue
            # Load fits file if source is in a different fits file than the one currently opened
            if single_field:
                hdu, hdr = load_fits(
                    single_field_path, dimensions_normal=dimensions_normal
                )
            else:
                if current_pointing_name != source[mosaic_id_key]:
                    # print("sourfce:", source[mosaic_id_key], source)
                    new_pointing_path = os.path.join(
                        data_directory,
                        pointing_dict[source[mosaic_id_key]],
                        fits_filename,
                    )
                    if not os.path.exists(new_pointing_path):
                        new_pointing_path = new_pointing_path.replace("13h", "0h")
                    hdu, hdr = load_fits(
                        new_pointing_path, dimensions_normal=dimensions_normal
                    )
                    current_pointing_name = source[mosaic_id_key]

            use_source_size = True
            if variable_size:
                try:
                    # Where 1.5 is the amount of arcsec per pixel
                    fullsize = int(
                        np.ceil(source.source_size * np.sqrt(2) / arcsec_per_pixel)
                    )
                    catsize = source.source_size / arcsec_per_pixel
                except:
                    use_source_size = False
                    try:
                        # Where 1.5 is the amount of arcsec per pixel
                        fullsize = int(
                            np.ceil(source.LGZ_Size * np.sqrt(2) / arcsec_per_pixel)
                        )
                        catsize = source.LGZ_Size / arcsec_per_pixel
                    except:
                        # Where 1.5 is the amount of arcsec per pixel
                        fullsize = int(
                            np.ceil(source.Maj * np.sqrt(2) / arcsec_per_pixel)
                        )
                        catsize = source.Maj / arcsec_per_pixel

                # if not rely_on_catalogue_size:
                if np.isnan(source.Maj):
                    # PyBDSF generated Major axis size is consistently smaller than LGZ convex hull
                    # size
                    fullsize = int(fullsize * LGZ_multiplier)
                else:
                    fullsize = int(fullsize * maj_multiplier)

            # Extract cutout
            try:
                hdu_crop = Cutout2D(
                    hdu,
                    co,
                    (fullsize, fullsize),
                    wcs=WCS(hdr, naxis=2),
                    copy=True,
                    mode=mode,
                )
            except (NoOverlapError, PartialOverlapError) as e:
                extraction_succeeded_list.append(False)
                count += 1
                if get_local_rms:
                    local_rms.append(np.nan)
                continue
            a, b = np.shape(hdu_crop.data)
            if a * b == fullsize * fullsize and not np.isnan(hdu_crop.data).any():
                image = hdu_crop.data

                # Get local image rms by masking all sources in view and returning the median signal
                if get_local_rms:
                    rms = np.nan
                    # return_local_rms(copy.deepcopy(image), hdu_crop.wcs, source,
                    #        full_cat, debug=False)
                    f = fullsize
                    while np.isnan(rms):
                        f *= 1.5
                        larger_hdu_crop = Cutout2D(
                            hdu,
                            co,
                            (f, f),
                            wcs=WCS(hdr, naxis=2),
                            copy=True,
                            mode="partial",
                            fill_value=np.nan,
                        )
                        rms = return_local_rms(
                            copy.deepcopy(larger_hdu_crop.data),
                            larger_hdu_crop.wcs,
                            source,
                            full_cat,
                            debug=False,
                            use_source_size=use_source_size,
                        )
                    local_rms.append(rms)

                # Remove neighbouring objects
                if remove_neighbours:
                    image = remove_neighbouring_sources_from_cutout(
                        image,
                        hdu_crop.wcs,
                        source,
                        component_cat,
                        gaus_cat,
                        debug=False,
                        deepfield=deepfield,
                    )

                # If no component and gaussian catalogue exists, try removing entire sources
                # (inferior option to the regular remove_neighbour function)
                if rough_remove_neighbours:
                    image = rough_remove_neighbouring_sources_from_cutout(
                        image, hdu_crop.wcs, source, full_cat
                    )

                # Calculate median and max values of the image
                if only_calculate_median_and_max:
                    if variable_size:
                        mask = create_circular_mask(fullsize)
                    masked_image = np.ma.array(
                        image,
                        mask=(image < lower_sigma_limit * source.cutout_rms)
                        | (image > upper_sigma_limit * source.cutout_rms),
                    )
                    masked_image[mask] = np.ma.masked

                    if masked_image.mask.all():
                        if verbose:
                            print(
                                f"{source.Source_Name} has no intensity > {lower_sigma_limit} the local noise"
                            )
                        max_median_list.append((i_p, None, None))
                        continue

                    max_median_list.append(
                        (i_p, np.ma.max(masked_image), np.ma.median(masked_image))
                    )

                    # Progress notification
                    if i % 1000 == 0 and i > 0:
                        time_elapsed = time.time() - start
                        print(
                            i,
                            "/",
                            len(pandas_catalogue),
                            "Time elapsed:",
                            int(time_elapsed),
                            "seconds. Estimated time left:",
                            round(
                                (
                                    time_elapsed * len(pandas_catalogue) / i
                                    - time_elapsed
                                )
                                / 60.0,
                                2,
                            ),
                            " minutes.",
                        )
                    continue

                # Resize image based on flux contained in radius
                if not rely_on_catalogue_size:
                    # """
                    radii = np.linspace(3, fullsize / 2, 20)
                    new_radius = 0
                    # Determine 95% flux containing radius
                    f0 = pixelvalues_within_radius(image, fullsize, debug=False)
                    for radius in radii:
                        f = pixelvalues_within_radius(image, radius, debug=False)
                        if f > 0.99:
                            new_radius = radius
                            break
                    rotated_size = fullsize / np.sqrt(2)
                    if new_radius < catsize / 3:
                        new_radius = catsize
                    # """
                    # rotated_size = fullsize/np.sqrt(2)
                    # new_radius = rotated_size
                    # Crop image
                    w = max(int((fullsize - new_radius * 1.5) / 2), 0)
                    image = image[w : fullsize - w, w : fullsize - w]

                # Rescale to destination size for variable cutout
                if variable_size:
                    image = np.array(
                        Image.fromarray(image).resize(
                            (destination_size, destination_size),
                            resample=Image.BILINEAR,
                        )
                    )
                    fullsize = destination_size

                if apply_clipping:
                    if get_local_rms:
                        if not np.isnan(rms):
                            # print('RMS', rms)
                            image = np.clip(
                                image, lower_sigma_limit * rms, upper_sigma_limit * rms
                            )
                    else:
                        image = np.clip(
                            image,
                            lower_sigma_limit * source.cutout_rms,
                            upper_sigma_limit * source.cutout_rms,
                        )
                if apply_mask:
                    if rescale:
                        image[mask] = np.min(image)
                    else:
                        image[mask] = 0

                if normed:
                    image_sum = np.sum(image)
                    if not image_sum == 0:
                        image = image / image_sum
                else:
                    # Scale from [0,1] using minmax
                    if rescale:
                        image = minmax_interval(image)
                if asinh:
                    assert rescale == False
                    assert normed == False
                    transform = vis.AsinhStretch() + vis.PercentileInterval(99.5)
                    image = transform(image)
                if apply_clipping and rescale:
                    if np.min(image) == np.max(image) or np.sum(image) == 0:
                        if skip_empty_cutouts:
                            count += 1
                            if verbose:
                                print(
                                    count,
                                    "failure. Apply clipping and rescaling is True. But min==max in the iamge or the sum of the image is zero",
                                )
                                print(
                                    f"{source.Source_Name} has no intensity > {lower_sigma_limit} the local noise"
                                )
                            # Debug stuff below
                            # print(i, "min equals max:", np.min(image) == np.max(image), "min and max are:", \
                            #        np.min(image), np.max(image), "sum is:", np.sum(image))
                            extraction_succeeded_list.append(False)
                            continue
                """
                else:

                    if np.max(image) <= lower_sigma_limit*source.cutout_rms: # or np.min(image) == np.max(image) or np.sum(image) == 0:
                        count += 1
                        im5 = debug_image(image,5)
                        debug_images(im0,im1,im5)
                        #print(i, "min equals max:", np.min(image) == np.max(image), "min and max are:", \
                        #        np.min(image), np.max(image), "sum is:", np.sum(image))
                        extraction_succeeded_list.append(False)
                        if verbose:
                            print(count, "failure. Apply clipping and rescaling is False. But max <= lower sigmalimit * rms")
                            print(f"max is {np.max(image):.2g} <= {lower_sigma_limit:.2g} * {source.cutout_rms:.2g} = {lower_sigma_limit*source.cutout_rms:.2g}")
                        if count>10:
                            sdfsdf
                        continue
                """

                image_list.append(image)
                extraction_succeeded_list.append(True)

            else:
                # If you end up here that probably means some of skycoords
                # were located too close to the border of the fits-file for cutout2D to create a
                #  (fullsize * fullsize)-sized box around them.
                extraction_succeeded_list.append(False)
                count += 1
                if get_local_rms:
                    local_rms.append(np.nan)
                if verbose:
                    print(count, "failure")
                    if not a * b == fullsize * fullsize:
                        print(
                            f"{count}th failure, {source.Source_Name} shape is {a}x{b} and should be {fullsize}x{fullsize}"
                        )
                    if np.isnan(source.cutout_rms):
                        print(
                            f"{count}th failure, {source.Source_Name} does not have a corresponding rms value"
                        )
                    if np.isnan(hdu_crop.data).any():
                        print(
                            f"{count}th failure, {source.Source_Name} contains nan values"
                        )

            # Progress notification
            if i % 100 == 0 and i > 0:
                time_elapsed = time.time() - start
                print(
                    i,
                    "/",
                    len(pandas_catalogue),
                    "Time elapsed:",
                    int(time_elapsed),
                    "seconds. Estimated time left:",
                    round(
                        (time_elapsed * len(pandas_catalogue) / i - time_elapsed)
                        / 60.0,
                        2,
                    ),
                    " minutes.",
                )

        # If only_calculate_median_and_max return just the max and median values
        if only_calculate_median_and_max:
            return max_median_list
        if get_local_rms:
            pandas_catalogue["local_rms"] = local_rms

        # Save image_list and add it to the pandas catalogue
        image_list = np.array(image_list)
        np.save(os.path.join(data_directory_2, store_filename), image_list)
        np.save(f"extraction_succeeded_{store_filename}.npy", extraction_succeeded_list)
        # Save catalogue
        pandas_catalogue = pandas_catalogue[extraction_succeeded_list]
        pandas_catalogue.to_hdf(
            os.path.join(data_directory_2, "catalogue_" + store_filename + ".h5"), "df"
        )
        # pandas_catalogue.to_csv(os.path.join(data_directory, 'catalogue_'+ store_filename+'.csv'))
    else:
        print(
            "Numpy store_filename containing cutouts already exists. Loading it in..."
        )
        image_list = np.load(os.path.join(data_directory_2, store_filename + ".npy"))
        pandas_catalogue = pd.read_hdf(
            os.path.join(data_directory_2, "catalogue_" + store_filename + ".h5"), "df"
        )

    print("Time elapsed:", int(time.time() - start), "seconds.")
    print("catalogue_" + store_filename + ".h5")
    pd.options.mode.chained_assignment = "warn"
    return image_list, pandas_catalogue


def debug_image(image, number):
    # plt.figure()
    # plt.imshow(image)
    # plt.title(number)
    # plt.colorbar()
    # plt.show()
    # plt.close()
    return copy.deepcopy(image)


def debug_images(*images):
    print(np.shape(images))
    for i_image, image in enumerate(images):
        if not image is None:
            plt.figure()
            plt.imshow(image)
            plt.title(i_image)
            plt.colorbar()
            plt.show()


def fits_to_numpy_cutouts_using_astropy(
    fullsize,
    pandas_catalogue,
    mosaic_id_key,
    ra_key,
    dec_key,
    data_directory,
    output_directory,
    cutouts_filename,
    fits_filename,
    overwrite=False,
    sort=True,
    short_mosaic_id=True,
    mode="partial",
    rms_fits_dimensions=False,
):

    start = time.time()
    if (
        overwrite
        or not os.path.exists(os.path.join(output_directory, cutouts_filename + ".npy"))
        or not os.path.exists(
            os.path.join(output_directory, "catalogue_" + cutouts_filename + ".pkl")
        )
    ):
        # Create list of directories that contain fits files in the data directory
        directory_names = get_immediate_subdirectories(data_directory)
        if short_mosaic_id:
            truncated_directory_names = [
                directory[:8] if len(directory) > 8 else directory
                for directory in directory_names
            ]
            directory_names = np.array(directory_names)[
                np.argsort(truncated_directory_names)
            ]
            truncated_directory_names = np.sort(truncated_directory_names)
            failures = {directory: 0 for directory in truncated_directory_names}
            failures = {directory: 0 for directory in directory_names}
        else:
            failures = {directory: 0 for directory in directory_names}

        print(
            "Found the following subdirectories from which fits files named {} will be attempted to be extracted:".format(
                fits_filename
            )
        )
        print(directory_names)

        # Make python list
        image_list = []
        corrected_catalogue_index = []
        count = 0

        # Sort catalogue on origin mosaic (in order to load each fits file only once)
        if sort:
            pandas_catalogue_sorted = pandas_catalogue.sort_values(by=[mosaic_id_key])
            print(
                "Catalogue sorted on origin mosaic, NOTE: this means the order changed!"
            )
        else:
            pandas_catalogue_sorted = pandas_catalogue
            print("Catalogue NOT sorted on origin mosaic")

        # Get skycoords of all sources
        skycoords = pandas_catalogue_to_astropy_coords(
            pandas_catalogue_sorted, ra_key, dec_key
        )

        ra_index = list(pandas_catalogue.iloc[0].index).index(ra_key)
        dec_index = list(pandas_catalogue.iloc[0].index).index(dec_key)
        mosaic_id_index = list(pandas_catalogue.iloc[0].index).index(mosaic_id_key)
        inspected_directory = pandas_catalogue_sorted.iloc[0][mosaic_id_key]
        print(inspected_directory)
        if short_mosaic_id:
            directory = directory_names[
                list(truncated_directory_names).index(inspected_directory)
            ]
        else:
            directory = directory_names[
                list(directory_names).index(inspected_directory)
            ]
        fits_file_path = os.path.join(
            data_directory, directory, fits_filename + ".fits"
        )
        # Load first fits file
        hdulist = fits.open(fits_file_path)
        if rms_fits_dimensions:
            hdu = hdulist[0].data[0, 0]
        else:
            hdu = hdulist[0].data
        # Header
        hdr = hdulist[0].header
        hdulist.close()
        if rms_fits_dimensions:
            useless_keys = [i for i in list(hdr.keys()) if i[-1] in ["3", "4"]]
            hdr["NAXIS"] = 2
            try:
                for useless in useless_keys:
                    hdr.remove(useless)
            except:
                print("keys nonexistent (possibly already removed)")
        start = time.time()
        for i, ((i_p, source), co) in enumerate(
            zip(pandas_catalogue_sorted.iterrows(), skycoords)
        ):
            # print(source, co)
            # Adjust hips-file to look into to find the source
            if source[mosaic_id_index] != inspected_directory:
                if short_mosaic_id:
                    for trunc, directory in zip(
                        truncated_directory_names, directory_names
                    ):
                        if (
                            source[mosaic_id_index] == trunc
                            or source[mosaic_id_index] == directory
                        ):
                            fits_file_path = os.path.join(
                                data_directory, directory, fits_filename + ".fits"
                            )
                            print(fits_file_path)
                            inspected_directory = source[mosaic_id_index]
                            # Load new fits file
                            hdulist = fits.open(fits_file_path)
                            if rms_fits_dimensions:
                                hdu = hdulist[0].data[0, 0]
                            else:
                                hdu = hdulist[0].data
                            # Header
                            hdr = hdulist[0].header
                            hdulist.close()
                            if rms_fits_dimensions:
                                useless_keys = [
                                    i for i in list(hdr.keys()) if i[-1] in ["3", "4"]
                                ]
                                hdr["NAXIS"] = 2
                                try:
                                    for useless in useless_keys:
                                        hdr.remove(useless)
                                except:
                                    print("keys nonexistent (possibly already removed)")
                else:
                    for directory in directory_names:
                        if source[mosaic_id_index] == directory:
                            fits_file_path = os.path.join(
                                data_directory, directory, fits_filename + ".fits"
                            )
                            print(fits_file_path)
                            inspected_directory = source[mosaic_id_index]
                            # Load new fits file
                            hdulist = fits.open(fits_file_path)
                            if rms_fits_dimensions:
                                hdu = hdulist[0].data[0, 0]
                            else:
                                hdu = hdulist[0].data
                            # Header
                            hdr = hdulist[0].header
                            hdulist.close()
                            if rms_fits_dimensions:
                                useless_keys = [
                                    i for i in list(hdr.keys()) if i[-1] in ["3", "4"]
                                ]
                                hdr["NAXIS"] = 2
                                try:
                                    for useless in useless_keys:
                                        hdr.remove(useless)
                                except:
                                    print("keys nonexistent (possibly already removed)")

            # Progress notification
            if i % 1000 == 0 and i > 0:
                time_elapsed = time.time() - start
                print(
                    i,
                    "/",
                    len(pandas_catalogue),
                    "Time elapsed:",
                    int(time_elapsed),
                    "seconds. Estimated time left:",
                    round(
                        (time_elapsed * len(pandas_catalogue) / i - time_elapsed)
                        / 60.0,
                        2,
                    ),
                    " minutes.",
                )

            try:
                hdu_crop = Cutout2D(hdu, co, (fullsize, fullsize), wcs=WCS(hdr))
                a, b = np.shape(hdu_crop.data)
                # print(type(hdu_crop.data), np.shape(hdu_crop.data))
                if a * b == fullsize * fullsize:
                    image = hdu_crop.data
                    image_list.append(image.reshape(fullsize * fullsize))
                    count += 1
                    corrected_catalogue_index.append(i_p)
                else:
                    print(
                        "failed:",
                        source[mosaic_id_index],
                        "dimensions: ({},{})".format(a, b),
                    )
                    # failures[inspected_directory] += 1
            except:
                print("failed:", source[mosaic_id_index])

        np.save(os.path.join(output_directory, cutouts_filename), np.array(image_list))
        # Create corrected catalogue
        catalogue_extracted = pandas_catalogue.loc[corrected_catalogue_index]
        catalogue_extracted.to_pickle(
            os.path.join(output_directory, "catalogue_" + cutouts_filename + ".pkl")
        )
        # print('Failed to extract:')
        # for i in failures.items():
        #    if i[1] > 0:
        #        print(i[0],':', i[1])
    else:
        print("Numpy file containing cutouts already exists. Loading it in...")
        image_list = np.load(os.path.join(output_directory, cutouts_filename + ".npy"))
        catalogue_extracted = pd.read_pickle(
            os.path.join(output_directory, "catalogue_" + cutouts_filename + ".pkl")
        )
    print(
        "Time elapsed:",
        int(time.time() - start),
        "seconds." "Number of images in numpy file:",
        len(image_list),
        " In corresponding catalogue:",
        len(catalogue_extracted),
        "\nNumber of images in original catalog:",
        len(pandas_catalogue),
    )
    assert len(catalogue_extracted) == len(image_list)
    if len(image_list) != len(pandas_catalogue):
        print(
            """If not all catalogue entries were extracted, that probably means some of them
                were located too close to the border of the fits-file for cutout2D to create a
                (fullsize * fullsize)-sized box around them."""
        )
    return image_list, catalogue_extracted


def discard_selected_source(image, i_p, source, discard_list):
    """Only returns image if it is not on the discard_list."""
    if i_p in discard_list.index:
        return False, None
    else:
        return True, image


def discard_image_with_nans(image, i_p, source):
    """Takes in an image, returns it only if it contains no NaNs."""
    if np.isnan(image).any():
        return False, None
    else:
        return True, image


def divide_by_sum(image, i_p, source):
    """Takes in an image, divide each pixel by the sum of the pixels."""
    return True, image / np.sum(image)


def sigmaclip_normalize_image(
    image_in,
    _,
    source,
    apply_clipping=False,
    lower_clip_bound=0,
    upper_clip_bound=1e9,
    apply_normalization=False,
    apply_mask=False,
    mask=[],
    image_side=0,
    flux_space=True,
):
    """Takes in an image, sigmaclips it and normalizes it."""

    # Remove area that won't be sampled (because of rotation)
    image = np.copy(image_in)

    # Clip background
    if apply_clipping:
        # factor 1000 to go from Jy to mJy
        if flux_space and (source["Peak_flux"] < lower_clip_bound * 1000):
            return False, None
        image = np.clip(image, lower_clip_bound, upper_clip_bound)

    if apply_mask:
        image.shape = (image_side, image_side)
        image[mask] = lower_clip_bound
        image.shape = image_side * image_side

    # Scale from [0,1] using minmax
    minmax_interval = vis.MinMaxInterval()
    image = minmax_interval(image)

    integrated_flux = np.sum(image)
    if integrated_flux < 1e-4:
        # Cutout vanishes in clipping process (no peak flux above lower_clip_bound)
        return False, None
    else:
        if apply_normalization:
            # Normalize integrated flux in the image
            image /= integrated_flux
        return True, image


def sigmaclip_image(
    image_in,
    _,
    source,
    apply_clipping=False,
    lower_clip_bound=0,
    upper_clip_bound=1e9,
    apply_mask=False,
    mask=[],
    image_side=0,
):
    """Takes in an image, sigmaclips it and minmax stretches it."""

    # Remove area that won't be sampled (because of rotation)
    image = np.copy(image_in)

    # Clip background
    if apply_clipping:
        image = np.clip(
            image, lower_clip_bound * source["cutout_rms"], upper_clip_bound
        )

    if apply_mask:
        image.shape = (image_side, image_side)
        image[mask] = np.min(image)
        image.shape = image_side * image_side
    # Scale from [0,1] using minmax
    minmax_interval = vis.MinMaxInterval()
    image = minmax_interval(image)
    return True, image


def map_function_on_cutouts(
    image_list,
    pandas_catalogue,
    save_path_images,
    save_path_catalogue,
    function,
    *args,
    overwrite=False,
    **kwargs,
):
    """
    Takes in a catalogue and imagelist and processes them using a given function.
    Returns corresponding new imagelist and catalogue.
    """

    if (
        overwrite
        or not os.path.exists(save_path_images)
        or not os.path.exists(save_path_catalogue)
    ):
        image_list_new = []
        catalogue_new_index = []
        for image, (i_p, source) in zip(image_list, pandas_catalogue.iterrows()):
            success, temp_image = function(image, i_p, source, *args, **kwargs)
            if success:
                image_list_new.append(temp_image)
                catalogue_new_index.append(i_p)
        # Save imagelist
        # image_list_new = np.array(image_list_new)
        np.save(save_path_images, np.array(image_list_new))
        # Save catalogue
        catalogue_new = pandas_catalogue.loc[catalogue_new_index]
        catalogue_new.to_pickle(save_path_catalogue)
    else:
        print("Files already exist, loading them from memory.")
        image_list_new = np.load(save_path_images)
        catalogue_new = pd.read_pickle(save_path_catalogue)
    print(
        "Length of new catalogue:",
        len(catalogue_new),
        "\nNumber of images in original catalog:",
        len(pandas_catalogue),
    )
    assert len(catalogue_new) == len(image_list_new)
    return image_list_new, catalogue_new


def fits_to_numpy_cutouts_using_astropy_coordinates(
    size_in_arcsec,
    ra_list,
    dec_list,
    pandas_catalogue,
    mosaic_id_key,
    ra_key,
    dec_key,
    data_directory,
    numpy_filename,
    fits_filename,
    overwrite=False,
    sort=True,
    short_mosaic_id=True,
    alternative_data_path=None,
    mode="partial",
    arcsec_per_pixel=1.5,
    apply_clipping=False,
    lower_clip_bound=None,
    upper_clip_bound=None,
):
    """
    Use this function to extract cutouts with size= size_in_arcsec*size_in_arcsec [in arcsec] using a list
    of RA ra_list and DEC dec_list [in degree].
    """

    start = time.time()
    if (
        overwrite
        or not os.path.exists(os.path.join(data_directory, numpy_filename + ".npy"))
        or not os.path.exists(
            os.path.join(
                data_directory, "corrected_catalogue_" + numpy_filename + ".npy"
            )
        )
    ):
        if alternative_data_path == None:
            alternative_data_path = data_directory
        # Create list of directories that contain fits files in the data directory
        directory_names = get_immediate_subdirectories(alternative_data_path)
        if short_mosaic_id:
            truncated_directory_names = [
                directory[:8] if len(directory) > 8 else directory
                for directory in directory_names
            ]
            directory_names = np.array(directory_names)[
                np.argsort(truncated_directory_names)
            ]
            truncated_directory_names = np.sort(truncated_directory_names)
            failures = {directory: 0 for directory in truncated_directory_names}
        else:
            failures = {directory: 0 for directory in directory_names}

        print(
            "Found the following subdirectories from which fits files named {} will be attempted to be extracted:".format(
                fits_filename
            )
        )
        print(directory_names)

        # Calculate pixelsize from arcsec_size
        fullsize = np.ceil(size_in_arcsec / arcsec_per_pixel)

        # Make python list
        image_list = []
        corrected_catalogue_index = []
        count = 0

        # if not os.path.exists(os.path.join(data_directory, numpy_filename + '.npy')):
        # Sort catalogue on origin mosaic (in order to load each fits file only once)
        if sort:
            pandas_catalogue_sorted = pandas_catalogue.sort_values(by=[mosaic_id_key])
            print(
                "Catalogue sorted on origin mosaic, NOTE: this means the order changed!"
            )
        else:
            pandas_catalogue_sorted = pandas_catalogue
            print("Catalogue NOT sorted on origin mosaic")

        # Find a close object in the catalogue (note: we use euclid not
        # spherical dist so we will find a close not the closest object)
        closest_entries = []
        for (ra, dec) in zip(ra_list, dec_list):
            closest_entries.append(
                pandas_catalogue.iloc[
                    (
                        (pandas_catalogue["RA"] - ra).abs()
                        + (pandas_catalogue["DEC"] - dec).abs()
                    ).argmin()
                ]
            )
        pandas_catalogue_sorted = pd.concat(closest_entries, axis=1).T
        print("Length of catalogue subset:", len(pandas_catalogue_sorted))

        # Get skycoords of all sources
        # skycoords = pandas_catalogue_to_astropy_coords(pandas_catalogue_sorted, ra_key, dec_key)
        skycoords = ra_dec_to_astropy_coords(ra_list, dec_list)

        # ra_index =
        # ra_index = list(pandas_catalogue_sorted.iloc[0].index).index(ra_key)
        # dec_index = list(pandas_catalogue_sorted.iloc[0].index).index(dec_key)
        mosaic_id_index = list(pandas_catalogue_sorted.iloc[0].index).index(
            mosaic_id_key
        )
        inspected_directory = pandas_catalogue_sorted.iloc[0][mosaic_id_key]
        print(inspected_directory)
        if short_mosaic_id:
            directory = directory_names[
                list(truncated_directory_names).index(inspected_directory)
            ]
        else:
            directory = directory_names[
                list(directory_names).index(inspected_directory)
            ]
        fits_file_path = os.path.join(
            alternative_data_path, directory, fits_filename + ".fits"
        )
        print(fits_file_path)
        # Load first fits file
        hdulist = fits.open(fits_file_path)
        hdu = hdulist[0].data
        # Header
        hdr = hdulist[0].header
        hdulist.close()
        for i, ((i_p, source), co) in enumerate(
            zip(pandas_catalogue_sorted.iterrows(), skycoords)
        ):
            # print(source, co)
            # Adjust hips-file to look into to find the source
            if source[mosaic_id_index] != inspected_directory:
                if short_mosaic_id:
                    for trunc, directory in zip(
                        truncated_directory_names, directory_names
                    ):
                        if (
                            source[mosaic_id_index] == trunc
                            or source[mosaic_id_index] == directory
                        ):
                            fits_file_path = os.path.join(
                                alternative_data_path,
                                directory,
                                fits_filename + ".fits",
                            )
                            print(fits_file_path)
                            inspected_directory = source[mosaic_id_index]
                            # Load new fits file
                            hdulist = fits.open(fits_file_path)
                            hdu = hdulist[0].data
                            # Header
                            hdr = hdulist[0].header
                            hdulist.close()
                else:
                    for directory in directory_names:
                        if source[mosaic_id_index] == directory:
                            fits_file_path = os.path.join(
                                alternative_data_path,
                                directory,
                                fits_filename + ".fits",
                            )
                            print(fits_file_path)
                            inspected_directory = source[mosaic_id_index]
                            # Load new fits file
                            hdulist = fits.open(fits_file_path)
                            hdu = hdulist[0].data
                            # Header
                            hdr = hdulist[0].header
                            hdulist.close()

            # Progress notification
            if i % 10 == 0 and i > 0:
                time_elapsed = time.time() - start
                print(
                    i,
                    "/",
                    len(pandas_catalogue),
                    "Time elapsed:",
                    int(time_elapsed),
                    "seconds. Estimated time left:",
                    round((time_elapsed * len(pandas_catalogue) / i) / 60.0, 2),
                    " minutes.",
                )

            try:
                hdu_crop = Cutout2D(
                    hdu, co, (fullsize, fullsize), wcs=WCS(hdr), copy=True, mode=mode
                )
                a, b = np.shape(hdu_crop.data)
                # print(type(hdu_crop.data), np.shape(hdu_crop.data))
                if a * b == fullsize * fullsize:
                    image_list.append(hdu_crop.data)
                    count += 1
                    corrected_catalogue_index.append(i)
                else:
                    print(
                        "failed:",
                        source[mosaic_id_index],
                        "dimensions: ({},{})".format(a, b),
                    )
                    failures[inspected_directory] += 1
            except:
                print("failed:", source[mosaic_id_index], inspected_directory)
                failures[inspected_directory] += 1

        np.save(os.path.join(data_directory, numpy_filename), np.array(image_list))
        np.save(
            os.path.join(data_directory, "corrected_catalogue_" + numpy_filename),
            np.array(corrected_catalogue_index),
        )
        print("Failed to extract:")
        for i in failures.items():
            if i[1] > 0:
                print(i[0], ":", i[1])
    else:
        print("Numpy file containing cutouts already exists. Loading it in...")
        image_list = np.load(os.path.join(data_directory, numpy_filename + ".npy"))
        corrected_catalogue_index = np.load(
            os.path.join(
                data_directory, "corrected_catalogue_" + numpy_filename + ".npy"
            )
        )
    print(
        "Time elapsed:",
        int(time.time() - start),
        "seconds." "Number of images in numpy file:",
        len(image_list),
        "num images in catalog:",
        len(pandas_catalogue),
    )
    if len(image_list) != len(pandas_catalogue):
        print(
            """If not all catalogue entries were extracted, that probably means some of them
                were located too close to the border of the fits-file for cutout2D to create a
                (fullsize * fullsize)-sized box around them."""
        )
    return image_list, corrected_catalogue_index


def get_immediate_subdirectories(a_dir):
    return [
        name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))
    ]


def pandas_catalogue_to_astropy_coords(pandas_catalogue, ra_key, dec_key):
    skycoords = SkyCoord(
        pandas_catalogue[ra_key], pandas_catalogue[dec_key], frame="icrs", unit="deg"
    )
    # print('Skycoords generated from pandas catalogue.')
    return skycoords


def ra_dec_to_astropy_coords(ra, dec):
    skycoords = SkyCoord(ra, dec, frame="icrs", unit="deg")
    return skycoords


def write_train_test_binary(
    fullsize,
    cutouts_list,
    test_fraction,
    random_seed,
    bin_train_path,
    bin_test_path,
    overwrite=False,
):
    """Write cutouts to train and test binary"""
    from sklearn.model_selection import train_test_split

    if overwrite or (
        not os.path.exists(bin_train_path) or not os.path.exists(bin_test_path)
    ):
        train, test = train_test_split(
            cutouts_list, test_size=test_fraction, random_state=random_seed
        )
        print("Size of train/test set:", len(train), len(test))
        # Open binary train file
        with open(bin_train_path, "wb") as output:
            output.write(struct.pack("i", len(train)))  # number of objectes
            output.write(struct.pack("i", 1))  # number of channels
            output.write(struct.pack("i", fullsize))  # width
            output.write(struct.pack("i", fullsize))  # height

            for image in train:
                image.astype("f").tofile(output)
        print("Done. Cutouts written to file:", bin_train_path)

        # Open binary test file
        with open(bin_test_path, "wb") as output:
            output.write(struct.pack("i", len(test)))  # number of objectes
            output.write(struct.pack("i", 1))  # number of channels
            output.write(struct.pack("i", fullsize))  # width
            output.write(struct.pack("i", fullsize))  # height

            for image in test:
                image.astype("f").tofile(output)
        print("Done. Cutouts written to file:", bin_test_path)
    else:
        print("These files already exist:", bin_train_path, bin_test_path)


def rotate_brightest_spot_to_top_right(image):
    best_angle = 0
    most_flux = 0
    half_size = int(np.floor(np.shape(image)[1] / 2))
    for test_rotation_angle_degree in range(1, 180, 3):
        img = rotate(image, test_rotation_angle_degree, reshape=False)
        diag = np.diagonal(img)
        if np.sum(diag[:half_size]) > most_flux:
            most_flux = np.sum(diag[:half_size])
            best_angle = test_rotation_angle_degree
        if np.sum(diag[half_size:]) > most_flux:
            most_flux = np.sum(diag[half_size:])
            best_angle = test_rotation_angle_degree + 180

    return rotate(image, best_angle, reshape=False), best_angle


def ignore_header_comments(inputStream):
    """Ignore header"""

    inputStream.seek(0)
    binary_start_position = 0
    for line in inputStream:
        if line == b"# END OF HEADER\n":
            binary_start_position = inputStream.tell()
            break

    inputStream.seek(binary_start_position)


def get_header_comments(inputStream):
    """Return header"""

    header = b""

    inputStream.seek(0)
    binary_start_position = 0
    for line in inputStream:
        if line == b"# END OF HEADER\n":
            binary_start_position = inputStream.tell()
            break

    inputStream.seek(binary_start_position)

    if binary_start_position != 0:
        for line in inputStream:
            header = header + inputStream.readline()
            if line == b"# END OF HEADER\n":
                break

    inputStream.seek(binary_start_position)

    return header


def plot_som_3D_contour(
    som,
    gap=4,
    save=False,
    save_dir=None,
    save_name="",
    normalize=False,
    cmap="viridis",
    colorbar=False,
    highlight=[],
    highlight_colors=[],
    legend=False,
    ax=None,
    legend_list=[],
    border_color="white",
    plot_norm_AQE=False,
    AQE_per_node=None,
    replace_nans=False,
    compress=False,
    AQE_std_per_node=None,
    align_prototypes=False,
    version=2,
    zoom_in=False,
    trained_path="",
    overwrite=False,
):
    """Simple way to show or save trained quadratic som"""

    if som.flip_axis0 or som.flip_axis1 or som.rot90:
        raise NotImplementedError
    if compress:
        assert (
            som.som_width == som.som_height
        ), "Compression only implemented for square SOMs"
        assert som.som_width > 2
    assert version == 2
    assert som.number_of_channels == 2, "Only implemented for two channels"
    print("Shape of SOM data", np.shape(som.data_som))
    assert som.layout == "quadratic" or som.layout == "cartesian"
    if (save_name == "") and (not legend) and (highlight == []):
        save_name = f"{som.som_width}_{som.som_height}"
        print(f"Changing save output name to '{save_name}'")

    # size of single SOM-node
    neuron_size = som.neuron_width
    r = som.rotated_size
    if zoom_in:
        img_xsize, img_ysize = r, r
    else:
        img_xsize, img_ysize = neuron_size, neuron_size

    # if False and align_prototypes and aligned_file_exists:
    #    assert version == 2
    #    print('Unpacking aligned SOM:', aligned_path)
    #    (data_som, som_width, som_height, som_depth,
    #            neuron_width, neuron_height, number_of_channels) = unpack_trained_som(aligned_path, som.layout,
    #                    version=version, verbose=False,replace_nans=replace_nans)
    # else:
    #    data_som = som.data_som
    data_som = som.data_som

    # Flip SOM to align multiple different SOMs
    data_som = copy.deepcopy(data_som)
    """
    if som.flip_axis0:
        print("Note: we are now flipping the SOM around axis 0 before plotting!")
        data_som = np.flip(data_som,axis=0)
    if som.flip_axis1:
        print("Note: we are now flipping the SOM around axis 1 before plotting!")
        data_som = np.flip(data_som,axis=1)
    if som.rot90:
        print("Note: we are now rotating the SOM 90deg before plotting!")
        data_som = np.rot90(data_som)
    """

    # Compress SOM to show about half its size
    if compress:
        i_compress, new_dimension = return_sparce_indices(som.som_width)
        print("icompress and new_dimension:", i_compress, new_dimension)
        # a 9x9,etc som should be reshaped to 81,etc for the indexes to work
        data_som = data_som.reshape(
            som.som_width * som.som_height, data_som.shape[-2], data_som.shape[-1]
        )
        print("New shape of SOM data", np.shape(data_som))
        data_som = data_som[i_compress].reshape(
            new_dimension, new_dimension, data_som.shape[-2], data_som.shape[-1]
        )
        print("Final shape of SOM data", np.shape(data_som))

        som_width = new_dimension
        som_height = new_dimension
    else:
        som_width = som.som_width
        som_height = som.som_height

    # x/y grid of wells per plate
    nx, ny, nz = int(som_width), int(som_height), int(som.number_of_channels)
    stitched_x, stitched_y = (gap + img_xsize) * nx + int(gap / 1), (
        gap + img_ysize
    ) * ny + int(gap / 1)
    mminterval = vis.MinMaxInterval()

    def add_img(nxi, nyi, img):
        assert img.shape == (img_ysize, img_xsize), f"shape: {img.shape}"
        xi = nxi * (img_xsize + gap) + gap
        yi = nyi * (img_ysize + gap) + gap
        img_stitched[yi : yi + img_ysize, xi : xi + img_xsize] = img

    assert som.som_depth == 1, "this func is ready for more channels not a 3D som :P"

    # For each channel:
    print(f"Plotting SOM for {som.number_of_channels} channels")
    best_angles = []
    angle_index = 0

    if ax is None:
        fig = plt.figure(figsize=(14, 14))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0], label="_nolegend_")
        ax.set_axis_off()
        fig.add_axes(ax)
    else:
        ax.set_axis_off()

    """
    fig = plt.figure(figsize=(20,20))
    ax = plt.Axes(fig, [0., 0., 1., 1.], label='_nolegend_')
    ax.set_axis_off()
    fig.add_axes(ax)
    """
    # Empty canvas
    img_stitched = np.zeros((stitched_y, stitched_x))

    for c in range(som.number_of_channels):

        # Check for aligned SOM
        aligned_path = trained_path.replace(".bin", "_aligned.bin")
        aligned_file_exists = False
        if os.path.exists(aligned_path):
            aligned_file_exists = True
        if 2 == 1 and align_prototypes and aligned_file_exists:
            assert version == 2
            print("Unpacking aligned SOM:", aligned_path)
            (
                data_som,
                som_width,
                som_height,
                som_depth,
                neuron_width,
                neuron_height,
                number_of_channels,
            ) = unpack_trained_som(
                aligned_path,
                som.layout,
                version=version,
                verbose=False,
                replace_nans=replace_nans,
            )
        else:
            if not compress:
                data_som = som.data_som

        save_rotated_prots = [[] for _ in range(som.number_of_channels)]
        offset_x = 1
        offset_y = 1
        for nxi in range(nx):
            if nxi > 0:
                ax.plot(
                    (
                        (img_xsize + gap) * nxi + offset_x,
                        (img_xsize + gap) * nxi + offset_x,
                    ),
                    (0, (img_ysize + gap) * ny),
                    linewidth=gap * 2,
                    color=border_color,
                )
            else:
                ax.plot(
                    (0, 0),
                    (0, (img_ysize + gap) * ny),
                    linewidth=gap * 2,
                    color=border_color,
                )
            for nyi in range(ny):

                img2 = data_som[nxi, nyi, 0]
                img = img2.reshape([som.number_of_channels, neuron_size, neuron_size])[
                    c
                ]
                if align_prototypes and (overwrite or not aligned_file_exists):
                    if c == 0:
                        img, best_angle = rotate_brightest_spot_to_top_right(img)
                        best_angles.append(best_angle)
                    else:
                        img = rotate(img, best_angles[angle_index], reshape=False)
                        angle_index += 1
                    save_rotated_prots[c].append(img)
                if zoom_in:
                    img = img[
                        int((neuron_size - r) / 2) : int((neuron_size + r) / 2),
                        int((neuron_size - r) / 2) : int((neuron_size + r) / 2),
                    ]
                if normalize:
                    img = mminterval(img)
                add_img(nxi, nyi, img)

                # Redo the border with chosen color
                ax.plot(
                    (0, (img_xsize + gap) * nx),
                    ((img_ysize + gap) * nyi, (img_ysize + gap) * nyi),
                    linewidth=gap * 2,
                    color=border_color,
                )

                # Plot values of AQE
                if plot_norm_AQE:
                    ax.text(
                        10 + nxi * img_xsize + gap,
                        10 + nyi * img_ysize + gap,
                        f"{AQE_per_node[nxi+nyi]:.5f}pm{AQE_std_per_node[nxi+nyi]:.5f}",
                        color="w",
                    )
        ax.plot(
            ((img_xsize + gap) * nx + offset_x, (img_xsize + gap) * nx + offset_x),
            (0, (img_ysize + gap) * ny + offset_y),
            linewidth=gap * 2,
            color=border_color,
        )
        ax.plot(
            (0, (img_xsize + gap) * nx + offset_x),
            ((img_ysize + gap) * ny + offset_y, (img_ysize + gap) * ny + offset_y),
            linewidth=gap * 2,
            color=border_color,
        )

        # if align_prototypes and not os.path.exists(os.path.join(som.output_directory,som.trained_subdirectory,'rotated.npy')):
        #    assert version == 2
        #    write_numpy_to_SOM_file(som, np.array(save_rotated_prot),
        #            trained_path.replace('.bin','_aligned.bin'), overwrite=True,
        #            version=2)
        #    np.save(os.path.join(som.output_directory, som.trained_subdirectory,'rotated'),1)
        if align_prototypes and (not aligned_file_exists or overwrite):
            print("align path does not exist", aligned_path)
            assert version == 2
            write_numpy_to_SOM_file(
                som,
                np.array(list(zip(*save_rotated_prots))),
                aligned_path,
                overwrite=overwrite,
                version=2,
                explicit_dimensions=[som.number_of_channels, neuron_size, neuron_size],
            )
        if c == 0:
            im = ax.imshow(
                img_stitched,
                aspect="equal",
                interpolation="nearest",
                origin="upper",
                cmap=cmap,
            )  # '_nolegend_')
        else:
            im = ax.contour(
                img_stitched,
                levels=np.array([0.1, 0.3, 0.5, 0.7]) * np.max(img_stitched),
            )

        # Highlight is a list of sublists, every sublist can contain a number of prototypes
        # all prototypes within a sublist are grouped. This way you can manually
        # color-code your SOM
        if not highlight == []:
            print("Entering highlight mode")
            appearance_list = []
            if not legend:
                legend_list = np.ones(len(highlight))
            else:
                assert not legend_list == []
            for group_index, (group, col, legend_label) in enumerate(
                zip(highlight, highlight_colors, legend_list)
            ):
                legend_flag = True
                for h in group:
                    ss = "solid"
                    if legend_flag:
                        ax.add_patch(
                            Rectangle(
                                (
                                    h[0] * (som.rotated_size + gap) + gap / 2,
                                    h[1] * (som.rotated_size + gap) + gap / 2,
                                ),
                                som.rotated_size,
                                som.rotated_size,
                                alpha=1,
                                facecolor="None",
                                edgecolor=col,
                                linewidth=gap * 2,
                                label=legend_label,
                                zorder=100,
                            )
                        )  # group_index+1))
                        if h in appearance_list:
                            print(
                                "To enable dashes move",
                                h,
                                "to later part in the sequence (or else the \
                                legend will be ugly)",
                            )
                        legend_flag = False
                    else:
                        if appearance_list.count(h) == 1:
                            # voor 2
                            # print('2-categories appearance', h)
                            ss = (0, (6, 6))
                        elif appearance_list.count(h) == 2:
                            # voor 3
                            # print('3-categories appearance', h)
                            ss = (3, (3, 6))
                        ax.add_patch(
                            Rectangle(
                                (
                                    h[0] * (som.rotated_size + gap) + gap / 2,
                                    h[1] * (som.rotated_size + gap) + gap / 2,
                                ),
                                som.rotated_size,
                                som.rotated_size,
                                alpha=1,
                                facecolor="None",
                                edgecolor=col,
                                linewidth=gap * 2,
                                label="_nolegend_",
                                linestyle=ss,
                                zorder=100,
                            )
                        )
                    appearance_list.append(h)
            if legend:
                ax.legend(
                    bbox_to_anchor=(1.04, 0.5),
                    loc="center left",
                    ncol=1,
                    # ax.legend(bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=1,
                    prop={"size": 24},
                )
        if colorbar:
            fig.colorbar(im, orientation="vertical")

    if save:
        if legend:
            # Tight layout is needed to include the legend in the saved figure
            print(os.path.join(save_dir, save_name + ".png"))
            plt.savefig(
                os.path.join(save_dir, save_name + ".png"),
                dpi="figure",
                bbox_inches="tight",
            )
        else:
            print(os.path.join(save_dir, save_name + ".png"))
            plt.savefig(
                os.path.join(save_dir, save_name + ".png"), dpi="figure"
            )  # , bbox_inches='tight')
    if ax is None:
        plt.show()
        plt.close()


def plot_som_3D(
    som,
    gap=4,
    save=False,
    save_dir=None,
    save_name="",
    normalize=False,
    cmap="viridis",
    colorbar=False,
    highlight=[],
    highlight_colors=[],
    legend=False,
    legend_list=[],
    border_color="white",
    plot_norm_AQE=False,
    AQE_per_node=None,
    replace_nans=False,
    AQE_std_per_node=None,
    align_prototypes=False,
    version=2,
    zoom_in=False,
    trained_path="",
    overwrite=False,
):
    """Simple way to show or save trained quadratic som"""

    if som.flip_axis0 or som.flip_axis1 or som.rot90:
        raise NotImplementedError
    assert version == 2
    print("Shape of SOM data", np.shape(som.data_som))
    assert som.layout == "quadratic" or som.layout == "cartesian"
    if (save_name == "") and (not legend) and (highlight == []):
        save_name = f"{som.som_width}_{som.som_height}"
        print(f"Changing save output name to '{save_name}'")

    # size of single SOM-node
    neuron_size = som.neuron_width
    r = som.rotated_size
    if zoom_in:
        img_xsize, img_ysize = r, r
    else:
        img_xsize, img_ysize = neuron_size, neuron_size

    # x/y grid of wells per plate
    nx, ny, nz = int(som.som_width), int(som.som_height), int(som.number_of_channels)
    stitched_x, stitched_y = (gap + img_xsize) * nx + int(gap / 1), (
        gap + img_ysize
    ) * ny + int(gap / 1)
    mminterval = vis.MinMaxInterval()

    def add_img(nxi, nyi, img):
        assert img.shape == (img_ysize, img_xsize), f"shape: {img.shape}"
        xi = nxi * (img_xsize + gap) + gap
        yi = nyi * (img_ysize + gap) + gap
        img_stitched[yi : yi + img_ysize, xi : xi + img_xsize] = img

    assert som.som_depth == 1, "this func is ready for more channels not a 3D som :P"

    # For each channel:
    print(f"Plotting SOM for {som.number_of_channels} channels")
    best_angles = []
    angle_index = 0
    for c in range(som.number_of_channels):
        fig = plt.figure(figsize=(14, 14))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0], label="_nolegend_")
        ax.set_axis_off()
        fig.add_axes(ax)

        # Empty canvas
        img_stitched = np.zeros((stitched_y, stitched_x))

        # Check for aligned SOM
        aligned_path = trained_path.replace(".bin", "_aligned.bin")
        aligned_file_exists = False
        if os.path.exists(aligned_path):
            aligned_file_exists = True
        if 2 == 1 and align_prototypes and aligned_file_exists:
            assert version == 2
            print("Unpacking aligned SOM:", aligned_path)
            (
                data_som,
                som_width,
                som_height,
                som_depth,
                neuron_width,
                neuron_height,
                number_of_channels,
            ) = unpack_trained_som(
                aligned_path,
                som.layout,
                version=version,
                verbose=False,
                replace_nans=replace_nans,
            )
        else:
            data_som = som.data_som

        save_rotated_prots = [[] for _ in range(som.number_of_channels)]
        offset_x = 1
        offset_y = 1
        for nxi in range(nx):
            if nxi > 0:
                ax.plot(
                    (
                        (img_xsize + gap) * nxi + offset_x,
                        (img_xsize + gap) * nxi + offset_x,
                    ),
                    (0, (img_ysize + gap) * ny),
                    linewidth=gap * 2,
                    color=border_color,
                )
            else:
                ax.plot(
                    (0, 0),
                    (0, (img_ysize + gap) * ny),
                    linewidth=gap * 2,
                    color=border_color,
                )
            for nyi in range(ny):

                img2 = data_som[nxi, nyi, 0]
                img = img2.reshape([som.number_of_channels, neuron_size, neuron_size])[
                    c
                ]
                if align_prototypes and (overwrite or not aligned_file_exists):
                    if c == 0:
                        img, best_angle = rotate_brightest_spot_to_top_right(img)
                        best_angles.append(best_angle)
                    else:
                        img = rotate(img, best_angles[angle_index], reshape=False)
                        angle_index += 1
                    save_rotated_prots[c].append(img)
                if zoom_in:
                    img = img[
                        int((neuron_size - r) / 2) : int((neuron_size + r) / 2),
                        int((neuron_size - r) / 2) : int((neuron_size + r) / 2),
                    ]
                if normalize:
                    img = mminterval(img)
                add_img(nxi, nyi, img)

                # Redo the border with chosen color
                ax.plot(
                    (0, (img_xsize + gap) * nx),
                    ((img_ysize + gap) * nyi, (img_ysize + gap) * nyi),
                    linewidth=gap * 2,
                    color=border_color,
                )

                # Plot values of AQE
                if plot_norm_AQE:
                    ax.text(
                        10 + nxi * img_xsize + gap,
                        10 + nyi * img_ysize + gap,
                        f"{AQE_per_node[nxi+nyi]:.5f}pm{AQE_std_per_node[nxi+nyi]:.5f}",
                        color="w",
                    )
        ax.plot(
            ((img_xsize + gap) * nx + offset_x, (img_xsize + gap) * nx + offset_x),
            (0, (img_ysize + gap) * ny + offset_y),
            linewidth=gap * 2,
            color=border_color,
        )
        ax.plot(
            (0, (img_xsize + gap) * nx + offset_x),
            ((img_ysize + gap) * ny + offset_y, (img_ysize + gap) * ny + offset_y),
            linewidth=gap * 2,
            color=border_color,
        )

        # if align_prototypes and not os.path.exists(os.path.join(som.output_directory,som.trained_subdirectory,'rotated.npy')):
        #    assert version == 2
        #    write_numpy_to_SOM_file(som, np.array(save_rotated_prot),
        #            trained_path.replace('.bin','_aligned.bin'), overwrite=True,
        #            version=2)
        #    np.save(os.path.join(som.output_directory, som.trained_subdirectory,'rotated'),1)
        if align_prototypes and (not aligned_file_exists or overwrite):
            print("align path does not exist", aligned_path)
            assert version == 2
            write_numpy_to_SOM_file(
                som,
                np.array(list(zip(*save_rotated_prots))),
                aligned_path,
                overwrite=overwrite,
                version=2,
                explicit_dimensions=[som.number_of_channels, neuron_size, neuron_size],
            )

        im = ax.imshow(
            img_stitched,
            aspect="equal",
            interpolation="nearest",
            origin="upper",
            cmap=cmap,
        )  # '_nolegend_')

        # Highlight is a list of sublists, every sublist can contain a number of prototypes
        # all prototypes within a sublist are grouped. This way you can manually
        # color-code your SOM
        if not highlight == []:
            print("Entering highlight mode")
            appearance_list = []
            if not legend:
                legend_list = np.ones(len(highlight))
            else:
                assert not legend_list == []
            for group_index, (group, col, legend_label) in enumerate(
                zip(highlight, highlight_colors, legend_list)
            ):
                legend_flag = True
                for h in group:
                    ss = "solid"
                    if legend_flag:
                        ax.add_patch(
                            Rectangle(
                                (
                                    h[0] * (som.rotated_size + gap) + gap / 2,
                                    h[1] * (som.rotated_size + gap) + gap / 2,
                                ),
                                som.rotated_size,
                                som.rotated_size,
                                alpha=1,
                                facecolor="None",
                                edgecolor=col,
                                linewidth=gap * 2,
                                label=legend_label,
                                zorder=100,
                            )
                        )  # group_index+1))
                        if h in appearance_list:
                            print(
                                "To enable dashes move",
                                h,
                                "to later part in the sequence (or else the \
                                legend will be ugly)",
                            )
                        legend_flag = False
                    else:
                        if appearance_list.count(h) == 1:
                            # voor 2
                            # print('2-categories appearance', h)
                            ss = (0, (6, 6))
                        elif appearance_list.count(h) == 2:
                            # voor 3
                            # print('3-categories appearance', h)
                            ss = (3, (3, 6))
                        ax.add_patch(
                            Rectangle(
                                (
                                    h[0] * (som.rotated_size + gap) + gap / 2,
                                    h[1] * (som.rotated_size + gap) + gap / 2,
                                ),
                                som.rotated_size,
                                som.rotated_size,
                                alpha=1,
                                facecolor="None",
                                edgecolor=col,
                                linewidth=gap * 2,
                                label="_nolegend_",
                                linestyle=ss,
                                zorder=100,
                            )
                        )
                    appearance_list.append(h)
            if legend:
                ax.legend(
                    bbox_to_anchor=(1.04, 0.5),
                    loc="center left",
                    ncol=1,
                    # ax.legend(bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=1,
                    prop={"size": 24},
                )
        if colorbar:
            fig.colorbar(im, orientation="vertical")
        if save:
            if legend:
                # Tight layout is needed to include the legend in the saved figure
                print(os.path.join(save_dir, save_name + ".png"))
                plt.savefig(
                    os.path.join(save_dir, save_name + ".png"),
                    dpi="figure",
                    bbox_inches="tight",
                )
            else:
                print(os.path.join(save_dir, save_name + ".png"))
                # , bbox_inches='tight')
                plt.savefig(os.path.join(save_dir, save_name + ".png"), dpi="figure")
        plt.show()
        plt.close()


def plot_som(
    som,
    gap=2,
    save=False,
    save_dir=None,
    save_name="",
    normalize=False,
    cmap="viridis",
    ax=None,
    colorbar=False,
    highlight=[],
    highlight_colors=[],
    legend=False,
    legend_list=[],
    border_color="white",
    plot_norm_AQE=False,
    AQE_per_node=None,
    replace_nans=False,
    AQE_std_per_node=None,
    align_prototypes=False,
    version=1,
    zoom_in=False,
    trained_path="",
    compress=False,
    highlight_rotatedsize=True,
    overwrite=False,
):
    """Simple way to show or save trained quadratic som"""

    print("Shape of SOM data", np.shape(som.data_som))
    # the current function can only deal with 1 channel quadratic SOMs
    assert som.number_of_channels == 1
    assert som.layout == "quadratic"
    if compress:
        assert (
            som.som_width == som.som_height
        ), "Compression only implemented for square SOMs"
        assert som.som_width > 2
    if (save_name == "") and (not legend) and (highlight == []):
        save_name = f"{som.som_width}_{som.som_height}"
        print(f"Changing save output name to '{save_name}'")

    # size of single SOM-node
    if version == 1:
        neuron_size = int(np.sqrt(np.shape(som.data_som)[3]))
    else:
        # Quick fix?
        neuron_size = som.fullsize
        # neuron_size = som.neuron_width #int(np.sqrt(np.shape(som.data_som)[3]))
    r = som.rotated_size
    if zoom_in:
        img_xsize, img_ysize = r, r
    else:
        img_xsize, img_ysize = neuron_size, neuron_size

    # Check for aligned SOM
    if trained_path == "":
        aligned_path = som.trained_path.replace(".bin", "_aligned.bin")
    else:
        aligned_path = trained_path.replace(".bin", "_aligned.bin")
    aligned_file_exists = False
    if os.path.exists(aligned_path):
        aligned_file_exists = True
    if False and align_prototypes and aligned_file_exists:
        assert version == 2
        print("Unpacking aligned SOM:", aligned_path)
        (
            data_som,
            som_width,
            som_height,
            som_depth,
            neuron_width,
            neuron_height,
            number_of_channels,
        ) = unpack_trained_som(
            aligned_path,
            som.layout,
            version=version,
            verbose=False,
            replace_nans=replace_nans,
        )
    else:
        data_som = som.data_som

    # Flip SOM to align multiple different SOMs
    data_som = copy.deepcopy(data_som)
    if som.flip_axis0:
        print("Note: we are now flipping the SOM around axis 0 before plotting!")
        data_som = np.flip(data_som, axis=0)
    if som.flip_axis1:
        print("Note: we are now flipping the SOM around axis 1 before plotting!")
        data_som = np.flip(data_som, axis=1)
    if som.rot90:
        print("Note: we are now rotating the SOM 90deg before plotting!")
        data_som = np.rot90(data_som)

    # Compress SOM to show about half its size
    if compress:
        i_compress, new_dimension = return_sparce_indices(som.som_width)
        data_som = data_som.reshape(
            som.som_width * som.som_height, data_som.shape[-2], data_som.shape[-1]
        )
        data_som = data_som[i_compress].reshape(
            new_dimension, new_dimension, data_som.shape[-2], data_som.shape[-1]
        )

        som_width = new_dimension
        som_height = new_dimension
    else:
        som_width = som.som_width
        som_height = som.som_height

    # x/y grid of wells per plate
    nx, ny = int(som_width), int(som_height)
    stitched_x, stitched_y = (gap + img_xsize) * nx + int(gap / 1), (
        gap + img_ysize
    ) * ny + int(gap / 1)
    mminterval = vis.MinMaxInterval()

    img_stitched = np.zeros((stitched_y, stitched_x))

    def add_img(nxi, nyi, img):
        assert img.shape == (img_ysize, img_xsize), f"shape: {img.shape}"
        xi = nxi * (img_xsize + gap) + gap
        yi = nyi * (img_ysize + gap) + gap
        img_stitched[yi : yi + img_ysize, xi : xi + img_xsize] = img

    if ax is None:
        fig = plt.figure(figsize=(14, 14))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0], label="_nolegend_")
        ax.set_axis_off()
        fig.add_axes(ax)
    else:
        ax.set_axis_off()

    save_rotated_prot = []
    for nxi in range(nx):
        ax.plot(
            ((img_xsize + gap) * nxi, (img_xsize + gap) * nxi),
            (0, (img_ysize + gap) * ny),
            linewidth=gap * 2,
            color=border_color,
        )
        for nyi in range(ny):
            img2 = data_som[nxi, nyi, 0]
            img = img2.reshape([neuron_size, neuron_size])
            if align_prototypes and (overwrite or not aligned_file_exists):
                img, _ = rotate_brightest_spot_to_top_right(img)
                save_rotated_prot.append(img)
            if zoom_in:
                img = img[
                    int((neuron_size - r) / 2) : int((neuron_size + r) / 2),
                    int((neuron_size - r) / 2) : int((neuron_size + r) / 2),
                ]
            if normalize:
                img = mminterval(img)
            add_img(nxi, nyi, img)
            # Redo the border with chosen color
            ax.plot(
                (0, (img_xsize + gap) * nx),
                ((img_ysize + gap) * nyi, (img_ysize + gap) * nyi),
                linewidth=gap * 2,
                color=border_color,
            )
            # Plot values of AQE
            if plot_norm_AQE:
                ax.text(
                    10 + nxi * img_xsize + gap,
                    10 + nyi * img_ysize + gap,
                    f"{AQE_per_node[nxi+nyi]:.5f}pm{AQE_std_per_node[nxi+nyi]:.5f}",
                    color="w",
                )
    ax.plot(
        ((img_xsize + gap) * nx, (img_xsize + gap) * nx),
        (0, (img_ysize + gap) * ny),
        linewidth=gap * 2,
        color=border_color,
    )
    ax.plot(
        (0, (img_xsize + gap) * nx),
        ((img_ysize + gap) * ny, (img_ysize + gap) * ny),
        linewidth=gap * 2,
        color=border_color,
    )

    # if align_prototypes and not os.path.exists(os.path.join(som.output_directory,som.trained_subdirectory,'rotated.npy')):
    #    assert version == 2
    #    write_numpy_to_SOM_file(som, np.array(save_rotated_prot),
    #            trained_path.replace('.bin','_aligned.bin'), overwrite=True,
    #            version=2)
    #    np.save(os.path.join(som.output_directory, som.trained_subdirectory,'rotated'),1)
    if align_prototypes and (not aligned_file_exists or overwrite):
        if not overwrite:
            print("align path does not exist", aligned_path)
        assert version == 2
        write_numpy_to_SOM_file(
            som,
            np.array(save_rotated_prot),
            aligned_path,
            overwrite=overwrite,
            version=2,
        )

    im = ax.imshow(
        img_stitched, aspect="equal", interpolation="nearest", origin="upper", cmap=cmap
    )  # '_nolegend_')
    if not save_dir is None:
        np.save(
            os.path.join(save_dir, f"trained_SOM_image_ID{som.run_id}"), img_stitched
        )

    # Highlight is a list of sublists, every sublist can contain a number of prototypes
    # all prototypes within a sublist are grouped. This way you can manually
    # color-code your SOM
    if not highlight == []:
        print("Entering highlight mode")
        appearance_list = []
        if not legend:
            legend_list = np.ones(len(highlight))
        else:
            assert not legend_list == []

        if highlight_rotatedsize:
            sw = som.rotated_size
        else:
            sw = som.fullsize
        for group_index, (group, col, legend_label) in enumerate(
            zip(highlight, highlight_colors, legend_list)
        ):
            legend_flag = True
            for h in group:
                ss = "solid"
                if legend_flag:
                    ax.add_patch(
                        Rectangle(
                            (h[0] * (sw + gap) + gap / 2, h[1] * (sw + gap) + gap / 2),
                            sw,
                            sw,
                            alpha=1,
                            facecolor="None",
                            edgecolor=col,
                            linewidth=gap * 2,
                            label=legend_label,
                            zorder=100,
                        )
                    )  # group_index+1))
                    if h in appearance_list:
                        print(
                            "To enable dashes move",
                            h,
                            "to later part in the sequence (or else the \
                            legend will be ugly)",
                        )
                    legend_flag = False
                else:
                    if appearance_list.count(h) == 1:
                        # voor 2
                        # print('2-categories appearance', h)
                        ss = (0, (6, 6))
                    elif appearance_list.count(h) == 2:
                        # voor 3
                        # print('3-categories appearance', h)
                        ss = (3, (3, 6))
                    ax.add_patch(
                        Rectangle(
                            (h[0] * (sw + gap) + gap / 2, h[1] * (sw + gap) + gap / 2),
                            sw,
                            sw,
                            alpha=1,
                            facecolor="None",
                            edgecolor=col,
                            linewidth=gap * 2,
                            label="_nolegend_",
                            linestyle=ss,
                            zorder=100,
                        )
                    )
                appearance_list.append(h)
        if legend:
            ax.legend(
                bbox_to_anchor=(1.04, 0.5),
                loc="center left",
                ncol=1,
                # ax.legend(bbox_to_anchor=(0.5, -0.3), loc='lower center', ncol=1,
                prop={"size": 24},
            )
    if colorbar:
        fig.colorbar(im, orientation="vertical")
    if save:
        if legend:
            # Tight layout is needed to include the legend in the saved figure
            print(os.path.join(save_dir, save_name + ".png"))
            plt.savefig(
                os.path.join(save_dir, save_name + ".png"),
                dpi="figure",
                bbox_inches="tight",
            )
        else:
            print(os.path.join(save_dir, save_name + ".png"))
            plt.savefig(
                os.path.join(save_dir, save_name + ".png"), dpi="figure"
            )  # , bbox_inches='tight')
    if ax is None:
        plt.show()
        plt.close()


def peak_divided_by_total_flux_all_prototypes(
    som, mask=False, mask_threshold=300, peak_range=3
):
    """Plots heatmap of total_flux/peak_flux of all prototypes of the given som.
    Returns the indexes of the masked prototypes"""
    assert som.layout == "quadratic"
    peak_map = np.zeros([som.som_width, som.som_height])
    for x in range(som.som_width):
        for y in range(som.som_height):
            # image = som.data_som[x,y,0]
            peak, total = get_peak_and_total_flux(
                som.data_som[x, y, 0].reshape(som.rotated_size, som.rotated_size),
                peak_range,
            )
            peak_map[x, y] = (total / peak) ** 0.5
            # print(total, peak, (total/peak)**0.5)
            # print(total/peak)
    # print(peak_map)
    cut_off_value = np.sort(peak_map.reshape(som.som_width * som.som_height))[10]
    peak_map_mask = peak_map < cut_off_value
    # peak_map_mask = peak_map<mask_threshold
    if mask:
        sns.heatmap(
            peak_map.T,
            mask=peak_map_mask.T,
            annot=False,
            square=True,
            fmt="d",
            cbar=True,
            cmap="inferno",
        )
    else:
        sns.heatmap(
            peak_map.T, annot=False, square=True, fmt="d", cbar=True, cmap="inferno"
        )
    # plt.imshow(peaik_map)
    plt.show()
    if mask:
        a, b = np.where(peak_map < mask_threshold)
        return [[i, j] for i, j in zip(a, b)]


def get_peak_and_total_flux(imag, peak_range=3):
    """Given an image returns peak and total flux"""
    image = np.copy(imag)
    image /= np.sum(image)
    image_x, image_y = np.shape(image)
    count = 0
    peak_flux = 0
    for i in range(int(image_x / 2 - peak_range), int(image_x / 2 + peak_range)):
        for j in range(int(image_y / 2 - peak_range), int(image_y / 2 + peak_range)):
            peak_flux += image[i, j]
            count += 1
    # peak_flux  /= count
    total_flux = np.sum(image)
    return peak_flux, total_flux


def get_AQE_and_TE_lists(
    output_directory, trained_subdirectories, version=1, overwrite=False
):
    numpy_filename = "errors"
    parameter_filename = "param"
    AQEs_train_list, AQEs_test_list, TEs_train_list, TEs_test_list, radius_list = (
        [],
        [],
        [],
        [],
        [],
    )
    for counter, trained_subdirectory in enumerate(trained_subdirectories):
        if 1 == 1 or not os.path.exists(
            os.path.join(
                output_directory, trained_subdirectory, parameter_filename + ".npz"
            )
        ):
            print("Calculating errors for:", trained_subdirectory)
            # Load maps
            # load variables from bash script
            variable_dic = {}
            run_filename = [
                f
                for f in os.listdir(
                    os.path.join(output_directory, trained_subdirectory)
                )
                if ".sh" in f
            ][0]
            run_path = os.path.join(
                output_directory, trained_subdirectory, run_filename
            )
            with open(run_path, "r", encoding="latin-1") as f:
                for i in range(25):
                    line = f.readline()
                    if line.find("=") != -1:
                        var = line.split("=")[0]
                        val = line.split("=")[1][:-1]
                        # print(var, val)
                        if val.find("#") != -1:
                            val = val[: val.find("#") - 1]
                        variable_dic[var] = val
            print(variable_dic)

            run_id = variable_dic["RUN_ID"].replace('"', "")
            neighborhood_radius = float(variable_dic["GAUSS"])
            radius_decrease = float(variable_dic["ALPHA"])
            learning_constraint = float(variable_dic["SIGMA"])
            epochs_per_epoch = int(variable_dic["EPOCHS_PER_EPOCH"])
            layout = variable_dic["LAYOUT"].replace('"', "")
            if layout == "quadratic":
                periodic_boundary_condition = (
                    variable_dic["PBC"].replace('"', "") == "True"
                )
            else:
                periodic_boundary_condition = False
            bin_name = variable_dic["BIN"].replace('"', "")
            som_width = int(variable_dic["W"])
            som_height = int(variable_dic["H"])
            som_depth = 1
            number_of_channels = 1
            som_label = "- 10x10 cyclic"
            rotated_size = int(variable_dic["RotatedSize"])
            full_size = int(variable_dic["FullSize"])
            print(epochs_per_epoch)

            # Get all map-filenames
            f_train = [
                f
                for f in os.listdir(
                    os.path.join(output_directory, trained_subdirectory)
                )
                if ("mapping_" in f) and ("train" in f)
            ]
            f_test = [
                f
                for f in os.listdir(
                    os.path.join(output_directory, trained_subdirectory)
                )
                if ("mapping_" in f) and ("test" in f)
            ]
            assert len(f_train) == len(f_test)
            epochs = len(f_train)

            # Create SOM object
            som = SOM(
                None,
                number_of_channels,
                som_width,
                som_height,
                som_depth,
                layout,
                output_directory,
                trained_subdirectory,
                som_label,
                rotated_size,
            )

            # Get neighborhood radii
            neighborhood_radii = np.array(
                [float(map_file.split("_")[-3]) for map_file in f_train]
            )
            neighborhood_radii2 = np.array(
                [float(map_file.split("_")[-3]) for map_file in f_test]
            )
            # Get sortindex
            # If neighborhood_radii are variable sort on those, else sort on learning constraint
            map_train_names = np.array(f_train)[np.argsort(neighborhood_radii)][::-1]
            map_test_names = np.array(f_test)[np.argsort(neighborhood_radii2)][::-1]

            map_train_paths = [
                os.path.join(output_directory, trained_subdirectory, n)
                for n in map_train_names
            ][:epochs]
            map_test_paths = [
                os.path.join(output_directory, trained_subdirectory, n)
                for n in map_test_names
            ][:epochs]

            np.savez(
                os.path.join(
                    output_directory, trained_subdirectory, parameter_filename
                ),
                run_id=run_id,
                neighborhood_radius=neighborhood_radius,
                radius_decrease=radius_decrease,
                learning_constraint=learning_constraint,
                bin_name=bin_name,
                rotated_size=rotated_size,
                full_size=full_size,
                epochs=epochs,
                epochs_per_epoch=epochs_per_epoch,
                som_width=som_width,
                som_height=som_height,
                periodic_boundary_condition=periodic_boundary_condition,
            )
            try:
                learning_constraint_start = float(
                    variable_dic["LEARNING_CONSTRAINT_START"]
                )
                learning_constraint_decrease = float(
                    variable_dic["LEARNING_CONSTRAINT_DECREASE"]
                )
                np.savez(
                    os.path.join(
                        output_directory,
                        trained_subdirectory,
                        parameter_filename + "_learning_constraint",
                    ),
                    learning_constraint_start=learning_constraint_start,
                    learning_constraint_decrease=learning_constraint_decrease,
                )
            except:
                print(
                    "This training file did not contain decreasing learning constraint yet."
                )

        if overwrite or not os.path.exists(
            os.path.join(
                output_directory, trained_subdirectory, numpy_filename + ".npz"
            )
        ):
            # load maps
            train_maps, test_maps = [], []
            fail_step = 999
            for i, (path_train, path_test) in enumerate(
                zip(map_train_paths, map_test_paths)
            ):
                (
                    data_map,
                    numberOfImages,
                    som_width,
                    som_height,
                    som_depth,
                ) = load_som_mapping(path_train, som, verbose=False, version=version)
                train_maps.append(data_map)
                (
                    data_map,
                    numberOfImages,
                    som_width,
                    som_height,
                    som_depth,
                ) = load_som_mapping(path_test, som, version=version, verbose=False)
                test_maps.append(data_map)

            AQEs_train = calculate_AQEs(som.rotated_size, train_maps)[0]
            AQEs_test = calculate_AQEs(som.rotated_size, test_maps)[0]
            TEs_train = calculate_TEs(
                train_maps, som_width, som_height, periodic_boundary_condition, layout
            )
            TEs_test = calculate_TEs(
                test_maps, som_width, som_height, periodic_boundary_condition, layout
            )
            np.savez(
                os.path.join(output_directory, trained_subdirectory, numpy_filename),
                AQEs_train=AQEs_train,
                AQEs_test=AQEs_test,
                TEs_train=TEs_train,
                TEs_test=TEs_test,
                neighborhood_radii=neighborhood_radii[np.argsort(neighborhood_radii)][
                    ::-1
                ],
            )
            # print(np.shape(AQEs_train), np.shape(AQEs_test))
            # print(AQEs_train, AQEs_test)
        else:
            data = np.load(
                os.path.join(
                    output_directory, trained_subdirectory, numpy_filename + ".npz"
                )
            )
            AQEs_train = data["AQEs_train"]
            AQEs_test = data["AQEs_test"]
            TEs_train = data["TEs_train"]
            TEs_test = data["TEs_test"]
            try:
                learning_constraint_data = np.load(
                    os.path.join(
                        output_directory,
                        trained_subdirectory,
                        numpy_filename + "_learning_constraint.npz",
                    )
                )
                learning_constraint_start = learning_constraint_data[
                    "learning_constraint_start"
                ]
                learning_constraint_decrease = learning_constraint_data[
                    "learning_constraint_decrease"
                ]
                neighborhood_radii = data["neighborhood_radii"]
            except:
                pass

        if not os.path.exists(
            os.path.join(
                output_directory, trained_subdirectory, parameter_filename + ".npz"
            )
        ):
            minkey = np.argmin(AQEs_test)
            print(trained_subdirectory, map_train_names[minkey].replace("mapping_", ""))
        AQEs_train_list.append(AQEs_train)
        AQEs_test_list.append(AQEs_test)
        TEs_train_list.append(TEs_train)
        TEs_test_list.append(TEs_test)
        try:
            radius_list.append(neighborhood_radii)
        except:
            pass

        print(
            "Finished calculating {}/{}".format(
                counter + 1, len(trained_subdirectories)
            )
        )
    AQEs_train_list = np.array(AQEs_train_list)
    AQEs_test_list = np.array(AQEs_test_list)
    TEs_train_list = np.array(TEs_train_list)
    TEs_test_list = np.array(TEs_test_list)
    radius_list = np.array(radius_list)
    return AQEs_train_list, AQEs_test_list, TEs_train_list, TEs_test_list, radius_list


def compare_source_heatmap_to_all_other_heatmaps(
    source_index, pd_catalog, n_closest_proto=10
):
    """Given the index to the pd catalog of a source,
    returns a sorted list of index aranged from most similar to least similar.
    The similarity measure between sources x and y will be the summed Euclidean distance between x and y.
    n_closest_proto determines the length of the heatmap that will be compared (do you want the full
    heatmap to be used or not?) Its max. value is equal to the size of the SOM."""

    # Get heatmap from given source index
    heatmap = np.array(pd_catalog.ix[source_index, "Heatmap"])
    # Normalize the heatmap from given source
    heatmap /= np.sum(heatmap)
    smallest_index = np.argsort(heatmap)
    smallest_key = np.array(range(len(heatmap)))[smallest_index][:n_closest_proto]
    # Get normalized heatmaps for all sources
    heatmaps = [
        list(np.array(row)[smallest_key] / np.sum(np.array(row)[smallest_key]))
        for row in pd_catalog["Heatmap"].tolist()
    ]
    # Calculate euclidean distance between heatmaps
    summed_heatmaps = scipy.spatial.distance.cdist(
        [heatmap[smallest_key]], heatmaps, "euclidean"
    )[0]

    # Return indices of the sorted distances (ascending order)
    sort_key = np.argsort(summed_heatmaps)
    return np.array(range(len(heatmaps)))[sort_key]


def compare_heatmap_to_all_other_heatmaps(
    heatmap,
    pd_catalogue,
    n_closest_proto=10,
    plot_histogram=False,
    inverted=False,
    normalize=False,
    xmax=1000,
    distance_metric="euclidean",
):
    """Given a heatmap,
    returns a sorted list of index aranged from most similar to least similar.
    Note: The output returns a list of iloc indexes, not actual indexes.
    The similarity measure between sources x and y will be the summed Euclidean distance between x and y.
    n_closest_proto determines the length of the heatmap that will be compared (do you want the full
    heatmap to be used or not?) Its max. value is equal to the size of the SOM."""

    if inverted:
        heatmap = 1 / heatmap
        pd_catalogue = pd_catalogue.assign(Heatmap=lambda x: (1 / x["Heatmap"]))
    smallest_index = np.argsort(heatmap)[:n_closest_proto]
    smallest_key = np.array(range(len(heatmap)))[smallest_index]
    # Calculate euclidean distance between heatmaps
    heatmaps = [
        list(np.array(row)[smallest_key]) for row in pd_catalogue["Heatmap"].tolist()
    ]
    if normalize:
        heatmaps = [
            list(np.array(row)[smallest_key] / np.sum(np.array(row)[smallest_key]))
            for row in pd_catalogue["Heatmap"].tolist()
        ]
    summed_heatmaps = scipy.spatial.distance.cdist(
        [heatmap[smallest_key]], heatmaps, distance_metric
    )[0]

    # Return indices of the sorted distances (ascending order)
    sort_key = np.argsort(summed_heatmaps)

    # Plot histogram of compared differences
    if plot_histogram:
        # Get the maj-size that contains 90% of the sources
        sorted_extremes = np.sort(summed_heatmaps)
        fig = plt.figure()
        fig.set_size_inches(10, 5)
        bins = plt.hist(summed_heatmaps, bins=2000)
        height = max(bins[0])

        sections = [0.9, 0.99, 0.999]
        sections = [0.1, 0.5]
        cutoff = [sorted_extremes[int(len(summed_heatmaps) * x)] for x in sections]

        # Plot red line for outliers shown
        # red_x = sorted_extremes[-number_of_outliers_to_show]
        # red_y = height*0.7
        # plt.text(red_x, red_y, str(number_of_outliers_to_show)+" biggest outliers shown below", color='r')
        # plt.vlines(red_x, ymax=red_y-0.02*height, ymin=0, color='r')
        # plt.arrow(red_x, red_y/2, 20, 0, shape='full', length_includes_head=True, head_width=height*0.05, head_length = 10, fc='r', ec='r')

        # Visualize the size-distribution
        hh = [height * 0.3, height * 0.2, height * 0.1]
        # for c, s, h in zip(cutoff, sections, hh):
        #    plt.vlines(c, ymax=h*0.95, ymin=0)
        #    print('Cut-off that includes {0}% of the sources: {1} (= {2} x median)'.format(s,
        #        round(c,1), round(c/np.median(summed_heatmaps),1)))
        #    plt.text(c, h, str(s*100)+'% of sources')
        #    plt.arrow(c, h*0.5, -20, 0, shape='full', length_includes_head=True, head_width=height*0.05, head_length = 10, fc='k', ec='k')

        # info_x = max(summed_heatmaps)*0.65
        info_x = cutoff[0] * 0.65
        plt.text(
            info_x,
            height * 0.25,
            """Median: {}
Mean: {}
Std. dev.: {} 
Max.: {}(={}xmedian)""".format(
                str(round(np.median(summed_heatmaps), 1)),
                str(round(np.mean(summed_heatmaps), 1)),
                str(round(np.std(summed_heatmaps), 1)),
                str(round(max(summed_heatmaps), 1)),
                str(int(max(summed_heatmaps) / np.median(summed_heatmaps))),
            ),
        )

        print(cutoff)
        plt.xlim([0, cutoff[0]])
        plt.yscale("log")
        plt.title("Histogram of distance to closest prototype")
        plt.xlabel("Summed Euclidian distance to closest prototype")
        plt.ylabel("Number of radio-sources per bin")
        plt.tight_layout()
        plt.show()
        # plt.savefig(outliers_path + '/outliers_histogram.png', transparent=True)
        # plt.close()

    if inverted:
        pd_catalogue = pd_catalogue.assign(Heatmap=lambda x: (1 / x["Heatmap"]))
    return np.array(range(len(heatmaps)))[sort_key]


def load_rotations_bin(bin_path, verbose=False):
    """Load a rotations binary file that store the rotation angle of the mapped sources"""

    # <file format version> 3 <number of entries> <som layout> <data>
    with open(bin_path, "rb") as inputStream:
        (
            file_format_version,
            output_type,
            number_of_entries,
            som_layout,
            som_dimensionality,
        ) = struct.unpack("i" * 5, inputStream.read(4 * 5))
        som_dimensions = struct.unpack(
            "i" * som_dimensionality, inputStream.read(4 * som_dimensionality)
        )

        assert file_format_version == 2
        assert output_type == 3

        if verbose:
            print("number of entries:", number_of_entries)
            print("som_layout:", som_layout)
            print("som dimensionality:", som_dimensionality)
            print("som dimensions:", som_dimensions)

        failed = 0
        flip_list = np.ones(number_of_entries, dtype="int")
        rotations = np.ones(number_of_entries)
        for i in range(number_of_entries):
            try:
                flip_list[i] = int(struct.unpack_from("b", inputStream.read(1))[0])
                rotations[i] = struct.unpack_from("f", inputStream.read(4))[0]
            except:
                failed += 1
    if failed > 0:
        print("Failed:", int(1.0 * failed))
        assert failed == 0, "possibly you are not using cartesian layout"
    return flip_list, rotations, number_of_entries


def map_dataset_to_trained_som(
    som,
    path_to_dataset_bin,
    path_to_mapping_result,
    path_to_trained_som,
    gpu_id,
    use_gpu=True,
    verbose=True,
    version=1,
    alternate_neuron_dimension=None,
    use_cuda_visible_devices=True,
    rotation_path=None,
    circular_shape=False,
):

    cuda_flag = " "
    cuda_visible_devices = f"CUDA_VISIBLE_DEVICES={gpu_id}"
    if not use_gpu:
        cuda_flag = "--cuda-off"
        cuda_visible_devices = ""
    if not use_cuda_visible_devices:
        cuda_visible_devices = ""

    if version == 1:

        bash_string = """{} /home/rafael/data/mostertrij/PINK_old/PINK/bin/Pink --inter-store overwrite \
        --neuron-dimension {} --numrot 360 --progress 0.1 {} \
        --som-width {} --som-height {} --layout {} --map {} {} {}""".format(
            cuda_visible_devices,
            som.rotated_size,
            cuda_flag,
            som.som_width,
            som.som_height,
            som.layout,
            path_to_dataset_bin,
            path_to_mapping_result,
            path_to_trained_som,
        )
    if version == 2:
        neuron_dimension_flag = " "
        rotation_store_flag = " "
        circular_shape_flag = " "

        if not rotation_path is None:
            rotation_store_flag = f"--store-rot-flip {rotation_path}"
        if alternate_neuron_dimension != None:
            neuron_dimension_flag = f"--euclidean-distance-dimension {alternate_neuron_dimension} --neuron-dimension {alternate_neuron_dimension}"
        else:
            neuron_dimension_flag = f"--euclidean-distance-dimension {som.rotated_size} --neuron-dimension {som.fullsize}"
        if circular_shape:
            circular_shape_flag = f"--euclidean-distance-shape circular"
        if som.layout == "quadratic":
            layout_v2 = "cartesian"
        else:
            layout_v2 = "hexagonal"
        bash_string = """{} Pink  {} --euclidean-distance-type float \
        --som-width {} --som-height {} --layout {} {} {} {} --map {} {} {}""".format(
            cuda_visible_devices,
            cuda_flag,
            som.som_width,
            som.som_height,
            layout_v2,
            neuron_dimension_flag,
            circular_shape_flag,
            rotation_store_flag,
            path_to_dataset_bin,
            path_to_mapping_result,
            path_to_trained_som,
        )

    if verbose:
        print(bash_string)
    return bash_string


def map_datasets_to_trained_som(
    som,
    bash_path,
    paths_to_binary_datasets,
    path_to_trained_som,
    paths_to_mapping_results,
    gpu_id,
    use_gpu=True,
    verbose=True,
    version=1,
    overwrite=False,
    **kwargs,
):
    """Given a SOM object and a number of datasets in the form of binary files (.bin),
    write the commands to a bashscript that when executed will map the datasets to the
    specified trained SOM using the specified GPU"""

    if overwrite:
        mapping_strings = [
            map_dataset_to_trained_som(
                som,
                p_bin,
                p_map,
                path_to_trained_som,
                gpu_id,
                verbose=verbose,
                use_gpu=use_gpu,
                version=version,
                **kwargs,
            )
            + "\n"
            for p_bin, p_map in zip(paths_to_binary_datasets, paths_to_mapping_results)
        ]
    else:
        mapping_strings = [
            map_dataset_to_trained_som(
                som,
                p_bin,
                p_map,
                path_to_trained_som,
                gpu_id,
                verbose=verbose,
                version=version,
                use_gpu=use_gpu,
                **kwargs,
            )
            + "\n"
            for p_bin, p_map in zip(paths_to_binary_datasets, paths_to_mapping_results)
            if os.path.exists(p_bin) and not os.path.exists(p_map)
        ]
    # [print('do map and bin paths exist? pbin, pmap', p_bin, p_map )
    #        for p_bin, p_map in zip(paths_to_binary_datasets, paths_to_mapping_results)]

    mapping_strings = np.array(mapping_strings)
    # print("mapping_string:", mapping_strings[0])

    if use_gpu:

        if isinstance(gpu_id, list):
            for i, gpu in enumerate(gpu_id):

                local_bash_path = bash_path.replace(".sh", f"_gpuID{gpu}.sh")
                with open(local_bash_path, "w") as f:
                    locs = list(range(i, len(mapping_strings), len(gpu_id)))
                    for string in mapping_strings[locs]:
                        f.write(f"CUDA_VISIBLE_DEVICES={gpu}" + string)
                        f.write("\nsleep 5\n")

        else:
            with open(bash_path, "w") as f:
                f.writelines(
                    list(
                        map(
                            lambda x: f"CUDA_VISIBLE_DEVICES={gpu}" + x + "\nsleep 3",
                            mapping_strings,
                        )
                    )
                )
    else:
        # print('mapping_strings', mapping_strings)
        with open(bash_path, "w") as f:
            f.writelines(list(map(lambda x: x + "\nsleep 3", mapping_strings)))


def closest_prototypes_to_heatmap(som, closest_prototypes, figsize=10):
    lijstje = np.zeros(som.som_width * som.som_height)
    for i in closest_prototypes:
        lijstje[i] += 1
    plt.figure(figsize=(figsize, figsize))
    lijstje = lijstje.reshape([som.som_width, som.som_height])
    sns.heatmap(lijstje.T, annot=False, square=True, fmt="d", cbar=True, cmap="inferno")
    plt.show()


def make_kdtree(ras, decs):
    """This makes a `scipy.spatial.CKDTree` on (`ra`, `decl`).
    Parameters
    ----------
    ras,decs : array-like
        The right ascension and declination coordinate pairs in decimal degrees.
    Returns
    -------
    `scipy.spatial.CKDTree`
        The cKDTRee object generated by this function is returned and can be
        used to run various spatial queries.
    """

    cosdec = np.cos(np.radians(decs))
    sindec = np.sin(np.radians(decs))
    cosra = np.cos(np.radians(ras))
    sinra = np.sin(np.radians(ras))
    xyz = np.column_stack((cosra * cosdec, sinra * cosdec, sindec))

    # generate the kdtree
    kdt = cKDTree(xyz, copy_data=True)

    return xyz, kdt


def return_nn_distance_in_arcsec(ras, decs, subset_indices=None):
    """Makes a `scipy.spatial.CKDTree` on (`ras`, `decs`)
    and return nearest neighbour distances in degrees.
    Parameters
    ----------
    ras,decs : array-like
        The right ascension and declination coordinate pairs in decimal degrees.
    subset_indices : array-like
        Indices of the subset for which you want nn distances to the rest of the RAs and DECs
    Returns
    -------
    Nearest neighbour distances in arcsec
    Nearest neighbour indices as bonus
    """
    # Create kdtree
    xyz, ra_dec_tree = make_kdtree(ras, decs)

    # Query kdtree for nearest neighbour distances
    if subset_indices is None:
        kd_out = ra_dec_tree.query(xyz, k=2)
    else:
        kd_out = ra_dec_tree.query(xyz[tuple(subset_indices)], k=2)

    nn_distances_rad = kd_out[0][:, 1]
    nn_distances_arcsec = np.rad2deg(nn_distances_rad) * 3600
    return nn_distances_arcsec, kd_out[1][:, 1]


# Execution takes about a minute per thousand sources (all because of great cirlce spatial separation calculation)
def return_nearest_neighbour_ids_and_pairs(pd_dataframe, title=""):
    """Returns id and distance of/to nearest neighbour (nn)
    NOTE: nn-distance is calculated using great circle distance.
    This is triggered by the p=3 in the KDTree_plus query.
    """
    warnings.warn(
        "Deprecated, take a look at the function 'return_nn_distance_in_arcsec' instead.",
        DeprecationWarning,
    )

    # credits for this file go to Erik Osinga and Martijn Oei.
    import ScipySpatialckdTree as great_tree

    if not os.path.exists(
        os.path.join(data_directory, "neighbour_ids{}.npz".format(title))
    ):

        # list of just the right ascencion and declination for all sources
        sources = list(zip(pd_dataframe["RA"].tolist(), pd_dataframe["DEC"].tolist()))

        # Build KD-tree to speed up nearest-neighbour look-up
        sources_tree = great_tree.KDTree_plus(sources, leafsize=10)

        # Query KD-tree for nearest neighbours
        nearest_id = []
        nearest_id_pairs = []
        for i, source in enumerate(sources):
            a = sources_tree.query(source, k=2, p=3)
            nearest_id.append(int(a[1][1]))
            nearest_id_pairs.append(tuple(sorted(a[1])))
        np.savez(
            os.path.join(data_directory, "neighbour_ids{}".format(title)),
            nearest_id=nearest_id,
            nearest_id_pairs=nearest_id_pairs,
        )
    else:
        loaded = np.load(
            os.path.join(data_directory, "neighbour_ids{}.npz".format(title))
        )
        nearest_id = loaded["nearest_id"]
        nearest_id_pairs = loaded["nearest_id_pairs"]

    return nearest_id, nearest_id_pairs


# Execution takes about a minute per thousand sources (all because of great cirlce spatial separation calculation)
def return_nearest_neighbour_separations(pd_dataframe, nearest_neighbour_ids, title=""):
    """
    Returns great angle distance between nearest neighbours.
    NOTE: distance calculation is slow because of dataframe conversions in Astropy
    """
    warnings.warn(
        "Deprecated, take a look at the function 'return_nn_distance_in_arcsec' instead.",
        DeprecationWarning,
    )

    if not os.path.exists(
        os.path.join(data_directory, "separations{}.npy".format(title))
    ):
        ras = pd_dataframe["RA"].tolist()
        decs = pd_dataframe["DEC"].tolist()

        nn_separations = []
        for i, nn_id in enumerate(nearest_neighbour_ids):
            object1 = SkyCoord(ras[i], decs[i], frame="icrs", unit="deg")
            object2 = SkyCoord(ras[nn_id], decs[nn_id], frame="icrs", unit="deg")
            separation = object1.separation(object2).degree
            nn_separations.append(separation)

        # Save to file to save time on next execution
        np.save(
            os.path.join(data_directory, "separations{}".format(title)), nn_separations
        )
    else:
        nn_separations = np.load(
            os.path.join(data_directory, "separations{}.npy".format(title))
        )

    return nn_separations


# Execution takes about a minute per thousand sources (all because of great cirlce spatial separation calculation)


def return_nearest_neighbour_pair_coordinates(pd_dataframe):
    """Returns the point in the sky in between the nearest neighbour pairs in the dataframe."""
    # Difference in RAs and DECs
    delta_ra = (
        pd_dataframe["RA"].values
        - pd_dataframe["RA"][pd_dataframe["Nearest_neighbour_id"].tolist()].values
    )
    delta_dec = (
        pd_dataframe["DEC"].values
        - pd_dataframe["RA"][pd_dataframe["Nearest_neighbour_id"].tolist()].values
    )

    new_ras = pd_dataframe["RA"] - delta_ra / 2
    new_decs = pd_dataframe["DEC"] - delta_dec / 2
    print(len(delta_ra), len(new_ras))

    return new_ras.tolist(), new_decs.tolist()


def get_nn_distances_slow(catalog_ids, ra_list, dec_list, save_name, output_directory):
    """Turns a list of RAs and DECs [in degree] into a skycoord catalogue
    subsequently searches the nearest neighbour for each item in the catalogue.
    Returns a list with ids and distance [in degree] to nn.
    """
    start = time.time()
    ids, distances = [], []
    catalog = SkyCoord(ra=ra_list * u.degree, dec=dec_list * u.degree)
    for i in catalog_ids:
        c = catalog[i]
        idx, d2d, d3d = c.match_to_catalog_sky(catalog, nthneighbor=2)
        ids.append(idx)
        distances.append(d2d.deg)
    np.savez(
        os.path.join(output_directory, save_name), ids=ids, distances_deg=distances
    )
    print(f"Time taken: {time.time()-start:.2f}")
    np.save(os.path.join(output_directory, "nn_time_taken"), time.clock() - start)


def pandas_catalogue_to_fits_paths(
    pandas_catalogue,
    mosaic_id_key,
    data_path,
    fits_filename="mosaic-blanked",
    hdf=False,
):
    """
    Note: This function is specific to the LOFAR DR1 release!
    Takes in a pandas catalogue which has for every entry a source from which,
    we want to know what is the path to the fits file where it resides.
    Also requires the path to the directory in which the subsequent pointing directories
    are stored.
    """
    # Get (truncated) directory names
    directory_names = get_immediate_subdirectories(data_path)
    truncated_directory_names = [
        directory[:8] if len(directory) > 8 else directory
        for directory in directory_names
    ]
    directory_dictionary = {
        (directory[:8] if len(directory) > 8 else directory): directory
        for directory in directory_names
    }
    # Loop over fits column and link them up to the fullname directory
    fits_file_paths = []
    if hdf:
        for pd_index, short_directory_name in zip(
            pandas_catalogue.index.values, pandas_catalogue[mosaic_id_key]
        ):
            fits_file_path = os.path.join(
                data_path,
                directory_dictionary[short_directory_name],
                fits_filename + ".fits",
            )
            assert os.path.exists(fits_file_path)
            fits_file_paths.append([pd_index, fits_file_path])
    else:

        for pd_index, short_directory_name in zip(
            pandas_catalogue["Unnamed: 0"], pandas_catalogue[mosaic_id_key]
        ):
            fits_file_path = os.path.join(
                data_path,
                directory_dictionary[short_directory_name],
                fits_filename + ".fits",
            )
            assert os.path.exists(fits_file_path)
            fits_file_paths.append([pd_index, fits_file_path])

    return fits_file_paths


def som_object_from_path(som_path, layout="quadratic", version=1, replace_nans=False):
    # Unpack trained SOM
    (
        data_som,
        som_width,
        som_height,
        som_depth,
        neuron_width,
        neuron_height,
        number_of_channels,
    ) = unpack_trained_som(som_path, layout, version=version, replace_nans=replace_nans)
    # Create SOM object
    som = SOM(
        data_som,
        number_of_channels,
        som_width,
        som_height,
        som_depth,
        layout,
        None,
        None,
        None,
        neuron_width,
    )
    return som


def retrieve_trained_som_paths(som, output_directory=None):
    """Return paths of trained and corresponding mapped paths.
    Assumes that each trained SOM has its own unique run_ID."""
    if output_directory is None:
        output_dir = som.output_directory
    else:
        output_dir = output_directory
    directory_with_right_run_id = [
        d for d in os.listdir(output_dir) if "ID" + str(som.run_id) in d
    ]
    assert len(directory_with_right_run_id) == 1, (
        f"{output_dir} should contain one item"
        f" only with run ID {som.run_id} but contains {directory_with_right_run_id}"
    )
    trained_subdirectory = directory_with_right_run_id[0]

    # Get all map-filenames
    mapping_prefix = "mapping_"
    f_mapped = [
        f
        for f in os.listdir(os.path.join(output_dir, trained_subdirectory))
        if (mapping_prefix in f)
    ]
    f_trained = [f[len(mapping_prefix) :] for f in f_mapped]
    assert len(f_mapped) == len(f_trained)
    epochs = len(f_trained)

    # Get neighborhood radii
    neighborhood_radii = np.array(
        [float(map_file.split("_")[-3]) for map_file in f_mapped]
    )

    # Get sortindex
    # If neighborhood_radii are variable sort on those, else sort on learning constraint
    map_trained_names = np.array(f_mapped)[np.argsort(neighborhood_radii)][::-1]
    trained_names = np.array(f_trained)[np.argsort(neighborhood_radii)][::-1]
    map_paths = [
        os.path.join(output_dir, trained_subdirectory, n) for n in map_trained_names
    ][:epochs]
    trained_paths = [
        os.path.join(output_dir, trained_subdirectory, n) for n in trained_names
    ][:epochs]
    return trained_subdirectory, trained_paths, map_paths, neighborhood_radii


def load_som_training_dataset_mapping(
    som, trained_paths, map_paths, version=1, replace_nans=False, verbose=True
):
    """For given som en trained paths, return mapping"""
    data_maps = []
    assert version == 1 or version == 2
    for (trained, mapped) in zip(trained_paths, map_paths):
        # Create list of indexes to retrieve coordinates of the cut-outs
        (
            data_map_intermediate,
            number_of_images,
            som_width,
            som_height,
            som_depth,
        ) = load_som_mapping(
            mapped, som, version=version, verbose=verbose, replace_nans=replace_nans
        )
        data_maps.append(data_map_intermediate)
    return data_maps


def calculate_AQEs_and_TEs(
    som,
    output_directory,
    trained_subdirectory,
    data_maps,
    trained_paths,
    verbose=True,
    overwrite=False,
):
    """Get list of AQE and TE for each epoch"""
    store_file = os.path.join(
        output_directory, trained_subdirectory, "aqes_tes_store.npz"
    )
    if overwrite or not os.path.exists(store_file):
        print(
            f"Overwrite is true or savefile {store_file} does not exist for AQEs and TEs."
        )

        # Calculate AQEs and TEs
        (
            aqe_list,
            aqe_std_dev_list,
            aqe_norm_list,
            aqe_norm_std_dev_list,
            aqe_quart_list,
            aqe_fifth_list,
            aqe_sev_list,
        ) = calculate_AQEs(som.rotated_size, data_maps)
        # Get TE list
        print(f"Periodic boundary conditions are {som.pbc}")
        te_list = calculate_TEs(
            data_maps, som.som_width, som.som_height, som.pbc, som.layout
        )
        if verbose:
            print("AQEs:", [a for a in aqe_list])
            # print('AQEs norm:', [a for a in aqe_norm_list])
            print("TEs:", [round(t) for t in te_list])
        np.savez(
            store_file,
            a=aqe_list,
            b=aqe_std_dev_list,
            c=aqe_norm_list,
            d=aqe_norm_std_dev_list,
            e=aqe_quart_list,
            f=aqe_fifth_list,
            g=aqe_sev_list,
            h=te_list,
        )
        return (
            aqe_list,
            aqe_std_dev_list,
            aqe_norm_list,
            aqe_norm_std_dev_list,
            aqe_quart_list,
            aqe_fifth_list,
            aqe_sev_list,
            te_list,
        )
    else:
        print("List of AQEs and TEs already exists. Loading them in...")
        z = np.load(store_file)
        return z["a"], z["b"], z["c"], z["d"], z["e"], z["f"], z["g"], z["h"]


def first_index(iterable, condition=lambda x: True):
    """
    Returns the index of the first item in the `iterable` that
    satisfies the `condition`.
    If the condition is not given, returns 0.
    Raises `StopIteration` if no item satysfing the condition is found.
    """
    return next(i for i, x in enumerate(iterable) if condition(x))


def first_index_start_from_back(iterable, condition=lambda x: True):
    return len(iterable) - next(
        i for i, x in enumerate(iterable[::-1]) if not condition(x)
    )


def plot_aqe_and_te_choose_som(
    som,
    figures_dir,
    aqe_list,
    aqe_std_dev_list,
    aqe_norm_list,
    aqe_norm_std_dev_list,
    aqe_quart_list,
    aqe_fifth_list,
    aqe_sev_list,
    te_list,
    neighborhood_radii,
    focus_on_TE=False,
    allowed_min_TE_offset_in_percent_points=2,
    AQE_stopping_condition_percent=1,
    plot_normed=False,
    version=1,
    force_chosen_som_index=None,
):
    """Plot AQE and TE versus epoch, choose the som that meets a certain condition
    by default this condition is met when the decline of AQE falls below 1%."""
    # Calculate their decline
    aqe_list_decline = []
    if version == 1:
        b = np.power(aqe_list[0], 2)
        for a in np.power(aqe_list[1:], 2):
            aqe_list_decline.append((b - a) / a * 100.0)
            b = a
    else:
        b = aqe_list[0]
        for a in aqe_list[1:]:
            aqe_list_decline.append((b - a) / a * 100.0)
            b = a

    # Plot AQE versus neighbourhood radii
    # this is a debug plot
    plt.figure(figsize=[6, 4])
    plt.plot(np.sort(neighborhood_radii)[::-1][1:], aqe_list_decline, ".")
    # plt.plot(np.linspace(0,3.5,10), np.ones(len(np.linspace(0,3.5,10))))
    plt.hlines(
        [0, AQE_stopping_condition_percent], xmin=0, xmax=max(neighborhood_radii)
    )
    # plt.grid()
    plt.xlim([min(neighborhood_radii) * 0.95, max(neighborhood_radii) * 1.05])
    plt.xlabel("neighbourhood radius")
    plt.ylabel("AQE decline [%]")
    plt.gca().invert_xaxis()
    plt.savefig(
        os.path.join(
            figures_dir,
            "neighbourhoodradius_vs_AQE_decline_ID{}.pdf".format(som.run_id),
        )
    )
    plt.show()

    # Chose the last SOM with a TE within 1 percentpoint close to the lowest TE attained during
    # training
    if focus_on_TE:
        min_TE = min(te_list)
        chosen_som_index = first_index_start_from_back(
            te_list, lambda x: x > (min_TE + allowed_min_TE_offset_in_percent_points)
        )
        print(f"min_TE {min_TE}")
        print(
            "Chosen SOM index is {} out of {} epochs".format(
                chosen_som_index, len(aqe_list_decline)
            )
        )
    else:

        # Chose the SOM where AQE improves less than 1% for the first time
        try:
            chosen_som_index = first_index_start_from_back(
                aqe_list_decline, lambda x: x < AQE_stopping_condition_percent
            )
        except:
            print(
                f"\n\nTraining is still improving more than {AQE_stopping_condition_percent}% per epoch. Consider to continue training!\n\n"
            )
            chosen_som_index = len(aqe_list_decline) - 1
    print(
        "Automatically chosen SOM index is {} out of {} epochs".format(
            chosen_som_index, len(aqe_list_decline)
        )
    )
    if not force_chosen_som_index is None:
        assert isinstance(force_chosen_som_index, int)
        print(f"However, forcing SOM index {force_chosen_som_index}")
        chosen_som_index = force_chosen_som_index
    print(
        "AQE for run ID{} is {} with std. dev {}, normed is {}, and TE is {}".format(
            som.run_id,
            aqe_list[chosen_som_index],
            aqe_std_dev_list[chosen_som_index],
            aqe_norm_list[chosen_som_index],
            te_list[chosen_som_index],
        )
    )

    """
    if version==1:
        # manual fix for the square root
        aqe_list, aqe_std_dev_list, aqe_norm_list, \
            aqe_norm_std_dev_list,aqe_quart_list, aqe_fifth_list, aqe_sev_list = list(map(np.sqrt, [aqe_list, aqe_std_dev_list, aqe_norm_list, \
            aqe_norm_std_dev_list,aqe_quart_list, aqe_fifth_list, aqe_sev_list]))
    """

    radii = np.sort(neighborhood_radii)[::-1]
    # Plot AQE and TE vs neighbourhood radius
    # plt.figure(figsize=[6,4])
    fig, ax1 = plt.subplots(figsize=[6, 4])
    ax1.errorbar(
        radii[: chosen_som_index + 1],
        aqe_list[: chosen_som_index + 1],
        yerr=aqe_std_dev_list[: chosen_som_index + 1],
        capsize=3,
        ls="None",
        marker="s",
    )
    # ax1.plot(radii[:chosen_som_index+1],te_list[:chosen_som_index+1], 'o')
    ax1.set_xlabel("neighbourhood radius ")
    ax1.set_ylabel("AQE", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(
        radii[: chosen_som_index + 1], te_list[: chosen_som_index + 1], "o", c="C1"
    )
    ax2.set_ylabel("TE [%]", color="C1")
    # ax1.set_ylim([min(np.add(aqe_list,np.multiply(aqe_std_dev_list,-1))),.1+max(np.add(aqe_list,aqe_std_dev_list))])
    ax1.set_ylim([0, 0.1 + max(np.add(aqe_list, aqe_std_dev_list))])
    # ax1.xlabel('neighbourhood radius')
    # ax1.ylabel('TE [%]   |   AQE')
    # ax1.legend(['TE [%]','AQE'])
    plt.gca().invert_xaxis()
    # plt.grid()
    plt.savefig(
        os.path.join(
            figures_dir, "neighbourhoodradius_vs_TE_AQE_ID{}.pdf".format(som.run_id)
        ),
        bbox_inches="tight",
    )
    plt.show()

    fig, ax1 = plt.subplots(figsize=[6, 4])
    ax1.errorbar(
        radii[: chosen_som_index + 1],
        aqe_fifth_list[: chosen_som_index + 1],
        yerr=[
            np.subtract(
                aqe_fifth_list[: chosen_som_index + 1],
                aqe_quart_list[: chosen_som_index + 1],
            ),
            np.subtract(
                aqe_sev_list[: chosen_som_index + 1],
                aqe_fifth_list[: chosen_som_index + 1],
            ),
        ],
        capsize=3,
        ls="None",
        marker="s",
    )
    # ax1.plot(radii[:chosen_som_index+1],te_list[:chosen_som_index+1], 'o')
    ax1.set_xlabel("neighbourhood radius ")
    ax1.set_ylabel("QE", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(
        radii[: chosen_som_index + 1], te_list[: chosen_som_index + 1], "o", c="C1"
    )
    ax2.set_ylabel("TE [%]", color="C1")
    ax1.set_ylim(
        [
            min(aqe_quart_list[: chosen_som_index + 1]),
            0.1 + max(aqe_sev_list[: chosen_som_index + 1]),
        ]
    )
    # ax1.xlabel('neighbourhood radius')
    # ax1.ylabel('TE [%]   |   AQE')
    # ax1.legend(['TE [%]','AQE'])
    plt.gca().invert_xaxis()
    # plt.grid()
    plt.savefig(
        os.path.join(
            figures_dir, "neighbourhoodradius_vs_TE_QE_ID{}.pdf".format(som.run_id)
        )
    )
    plt.show()
    if plot_normed:

        # Plot normed AQE and TE vs neighbourhood radius
        fig, ax1 = plt.subplots()
        ax1.errorbar(
            radii[: chosen_som_index + 1],
            aqe_norm_list[: chosen_som_index + 1],
            yerr=aqe_norm_std_dev_list[: chosen_som_index + 1],
            capsize=3,
            ls="None",
            marker="s",
            c="blue",
        )
        ax1.set_xlabel("neighbourhood radius")
        ax1.set_ylabel("AQE / (cutout width)^2", color="blue")
        ax2 = ax1.twinx()
        ax2.plot(
            radii[: chosen_som_index + 1],
            te_list[: chosen_som_index + 1],
            "o",
            c="orange",
        )
        ax2.set_ylabel("TE [%]", color="orange")
        plt.gca().invert_xaxis()
        # plt.grid()
        plt.savefig(
            os.path.join(
                figures_dir,
                "neighbourhoodradius_vs_TE_AQE_norm_ID{}.pdf".format(som.run_id),
            )
        )
        plt.show()

    return chosen_som_index


def unpack_som_and_enrich_catalogue(
    som,
    chosen_som_index,
    data_maps,
    trained_subdirectory,
    data_directory,
    output_directory,
    trained_path,
    replace_nans=False,
    flip_axis0=False,
    flip_axis1=False,
    rot90=False,
    catalogue_name=None,
    version=1,
    verbose=False,
    overwrite=True,
):
    """Given a path to a trained som, creates a SOM object and add closest prototype ID to the catalogue."""

    # Unpack trained SOM
    if overwrite:
        (
            data_som,
            som_width,
            som_height,
            som_depth,
            neuron_width,
            neuron_height,
            number_of_channels,
        ) = unpack_trained_som(
            trained_path,
            som.layout,
            version=version,
            verbose=verbose,
            replace_nans=replace_nans,
        )

        print(
            "w,h,d, channels, rotated",
            som.som_width,
            som_width,
            som.som_height,
            som_height,
            som.som_depth,
            som_depth,
            som.number_of_channels,
            number_of_channels,
            som.rotated_size,
            neuron_width,
            neuron_height,
        )
        assert som.som_width == som_width
        assert som.som_height == som_height
        assert som.som_depth == som_depth
        assert som.number_of_channels == number_of_channels
        if version == 1:
            assert som.rotated_size == neuron_width
            assert som.rotated_size == neuron_height
        som.neuron_width = neuron_width
        som.neuron_height = neuron_height
        som.version = version
        som.data_som = data_som
        som.flip_axis0 = flip_axis0
        som.flip_axis1 = flip_axis1
        som.rot90 = rot90
        som.save()

    # Create list of indexes to retrieve coordinates of the cut-outs
    data_map = data_maps[chosen_som_index]
    distance_to_bmu = np.min(data_map, axis=1)
    distance_to_bmu_sorted_down_id = np.argsort(distance_to_bmu)[::-1]
    closest_prototype_id = np.argmin(data_map, axis=1)
    prototype_x = np.array([int(i % som.som_height) for i in closest_prototype_id])
    prototype_y = np.array([int(i / som.som_height) for i in closest_prototype_id])

    # Load the csv catalogue into pandas
    if isinstance(catalogue_name, pd.DataFrame):
        catalogue_final = catalogue_name
    elif catalogue_name is None:
        output_cat_path = os.path.join(
            output_directory, "catalogue_" + som.training_dataset_name + ".h5"
        )
        data_cat_path = os.path.join(
            data_directory, "catalogue_" + som.training_dataset_name + ".h5"
        )
        print("Trying to open:", output_cat_path, "or", data_cat_path)
        if os.path.exists(output_cat_path):
            catalogue_final = pd.read_hdf(output_cat_path)
        else:
            catalogue_final = pd.read_hdf(data_cat_path)

    # Add mapping to catalogue
    print("len cat final and closest", len(catalogue_final), len(closest_prototype_id))
    print(
        "If the code fails now, you might want to insert the correct catalogue using the"
        " catalogue_name kwarg. This argument can be a pandas catalogue (not just string)"
    )
    catalogue_final["Closest_prototype"] = list(closest_prototype_id)
    catalogue_final["Closest_prototype_x"] = list(prototype_x)
    catalogue_final["Closest_prototype_y"] = list(prototype_y)
    catalogue_final["Heatmap"] = list(data_map)
    catalogue_final["Distance_to_bmu"] = distance_to_bmu
    return (
        catalogue_final,
        distance_to_bmu_sorted_down_id,
        closest_prototype_id,
        prototype_x,
        prototype_y,
    )


def create_website_content_for_som(
    run_id,
    bin_path,
    data_directory,
    output_directory,
    figures_dir,
    fit_threshold=0.077,
    print_rms=False,
    print_peak_flux=False,
    AQE_stopping_condition_percent=1,
    resemblance_division_line=None,
    highlight_cutouts=None,
    flip_axis0=False,
    flip_axis1=False,
    rot90=False,
    highlight_cutouts2=None,
    focus_on_TE=False,
    correction_contribution=0.033,
    number_of_outliers_to_show=100,
    max_number_of_images_to_show=1000,
    verbose=False,
    align_prototypes=False,
    version=1,
    zoom_in=False,
    catalogue_name=None,
    short_stop=False,
    comparative_datamap=[],
    force_chosen_som_index=None,
    overwrite=False,
):

    if version == 1:
        assert align_prototypes == False
    start = time.time()
    if not isinstance(run_id, int):
        som = run_id
        run_id = 48
    else:

        # Get paths and directories for each epoch
        som = load_SOM(output_directory, run_id)
        som.print()
        if version == 2:
            assert (
                not som.pbc
            ), "Periodic boundary conditions are not implemented in PINK version >=2"
        (
            trained_subdirectory,
            trained_paths,
            map_paths,
            neighborhood_radii,
        ) = retrieve_trained_som_paths(som)
        som.trained_subdirectory = trained_subdirectory

    print("retrieved paths. Time taken:", round(time.time() - start))
    start = time.time()

    # Get mapping for each epoch
    data_maps = load_som_training_dataset_mapping(
        som, trained_paths, map_paths, version=version, verbose=verbose
    )
    print("loaded data_maps. Time taken:", round(time.time() - start))
    start = time.time()

    # Get list of AQE and TE for each epoch
    (
        aqe_list,
        aqe_std_dev_list,
        aqe_norm_list,
        aqe_norm_std_dev_list,
        aqe_quart_list,
        aqe_fifth_list,
        aqe_sev_list,
        te_list,
    ) = calculate_AQEs_and_TEs(
        som,
        output_directory,
        trained_subdirectory,
        data_maps,
        trained_paths,
        overwrite=overwrite,
    )
    print("get aqes and tes. Time taken:", round(time.time() - start))
    start = time.time()

    # Plot the AQEs and TEs
    if not force_chosen_som_index is None:
        print(f"Warning: Using forced chosen SOM index {force_chosen_som_index}")
        chosen_som_index = force_chosen_som_index
    else:
        chosen_som_index = plot_aqe_and_te_choose_som(
            som,
            figures_dir,
            aqe_list,
            aqe_std_dev_list,
            aqe_norm_list,
            aqe_norm_std_dev_list,
            aqe_quart_list,
            aqe_fifth_list,
            aqe_sev_list,
            te_list,
            neighborhood_radii,
            plot_normed=False,
            version=version,
            AQE_stopping_condition_percent=AQE_stopping_condition_percent,
            focus_on_TE=focus_on_TE,
        )
        print("Chosen som index", chosen_som_index)
    print("plot aqe and te. Time taken:", round(time.time() - start))

    # Take last trained file to display
    trained_path = trained_paths[chosen_som_index]
    map_path = map_paths[chosen_som_index]
    data_map = data_maps[chosen_som_index]
    print("trained_path:", trained_path)

    # Create content directories
    website_path = os.path.join(
        output_directory, trained_subdirectory, "website_ID{}".format(run_id)
    )
    print(website_path)
    outliers_path = os.path.join(website_path, "outliers")
    os.makedirs(outliers_path, exist_ok=True)

    # Create som object and add unpacked mapping to catalogue
    (
        catalogue_final,
        distance_to_bmu_sorted_down_id,
        closest_prototype_id,
        prototype_x,
        prototype_y,
    ) = unpack_som_and_enrich_catalogue(
        som,
        chosen_som_index,
        data_maps,
        trained_subdirectory,
        data_directory,
        output_directory,
        trained_path,
        version=version,
        verbose=verbose,
        flip_axis0=flip_axis0,
        flip_axis1=flip_axis1,
        rot90=rot90,
        catalogue_name=catalogue_name,
        overwrite=overwrite,
    )

    if short_stop:
        return (
            data_map,
            som.data_som,
            trained_path,
            closest_prototype_id,
            catalogue_final,
        )

    # Plot som
    plot_som(
        som,
        gap=2,
        save=True,
        save_name=f"{som.som_width}_{som.som_height}",
        save_dir=website_path,
        normalize=True,
        cmap="viridis",
        align_prototypes=align_prototypes,
        zoom_in=zoom_in,
        trained_path=trained_path,
        version=version,
    )
    # Plot heatmap
    plot_som_bmu_heatmap(
        som, None, data_map=data_map, cbar=False, save=True, save_dir=website_path
    )

    # Calculate AQE per prototype/node
    if verbose:

        (
            AQE_per_node,
            AQE_std_per_node,
            QE_per_node,
        ) = calculate_normalized_AQE_per_prototype(
            data_map, som.rotated_size, verbose=verbose
        )
        check_percentage_below_threshold(som, data_map, fit_threshold)
        return_cutouts_per_percentile_per_prototype(
            som, data_directory, data_map, QE_per_node, threshold=fit_threshold
        )

    # Generate website content
    print("outliers path:", outliers_path, "version", version)
    outliers = plot_and_save_outliers(
        som,
        outliers_path,
        bin_path,
        number_of_outliers_to_show,
        catalogue_final,
        closest_prototype_id,
        distance_to_bmu_sorted_down_id,
        clip_threshold=False,
        plot_border=False,
        version=version,
        debug=False,
        apply_clipping=False,
        overwrite=overwrite,
        save=True,
    )

    # Plot outliers histogram
    plot_distance_to_bmu_histogram(
        data_map, number_of_outliers_to_show, outliers_path, xmax="", save=True
    )

    # Make and save web compatible catalogue
    pd_catalogue_web_compatible = catalogue_final.drop(columns=["Heatmap"])
    pd_catalogue_web_compatible["cutout_index"] = pd_catalogue_web_compatible.index
    save_path = os.path.join(website_path, "pd_catalog.csv")
    catalogue_final.to_csv(save_path, index=False)

    # Save all max_number_of_images_to_show
    save_all_prototypes_and_cutouts(
        som,
        bin_path,
        website_path,
        max_number_of_images_to_show,
        catalogue_final,
        pd_catalogue_web_compatible,
        data_map,
        version=version,
        print_rms=print_rms,
        print_peak_flux=print_peak_flux,
        resemblance_division_line=resemblance_division_line,
        highlight_cutouts=highlight_cutouts,
        highlight_cutouts2=highlight_cutouts2,
        correction_contribution=correction_contribution,
        figsize=3,
        save=True,
        plot_border=False,
        plot_cutout_index=False,
        apply_clipping=False,
        clip_threshold=3,
        overwrite=overwrite,
    )

    # make AQE hist matrix
    plot_matrix_of_AQE_histograms(
        som,
        data_map,
        som.rotated_size,
        save=True,
        absolute=True,
        save_path=os.path.join(website_path, "histogram"),
        comparative_datamap=comparative_datamap,
        plot_only_comparative=False,
    )
    plot_matrix_of_AQE_histograms(
        som,
        data_map,
        som.rotated_size,
        save=True,
        absolute=False,
        save_path=os.path.join(website_path, "histogram"),
        comparative_datamap=comparative_datamap,
        plot_only_comparative=False,
    )

    # return (data_maps[chosen_som_index], som.data_som, trained_path, closest_prototype_id,
    #        catalogue_final, u_matrix_selected_prototypes)


def inspect_trained_som(
    run_id,
    data_directory,
    output_directory,
    figures_dir,
    fit_threshold=0.077,
    verbose=False,
    align_prototypes=False,
    version=1,
    zoom_in=False,
    catalogue_name=None,
    replace_nans=False,
    short_stop=False,
    save=True,
    overwrite=False,
    normalize=True,
    save_dir=None,
    plot_umap=True,
    compress=False,
    overwrite_SOM_output_dir=False,
    flip_axis0=False,
    flip_axis1=False,
    rot90=False,
    AQE_stopping_condition_percent=1,
    focus_on_TE=False,
    force_chosen_som_index=None,
):

    if version == 1:
        assert align_prototypes == False
    start = time.time()
    # Get paths and directories for each epoch
    som = load_SOM(output_directory, run_id)
    som.print()
    if version == 2:
        assert (
            not som.pbc
        ), "Periodic boundary conditions are not implemented in PINK version >=2"
    if overwrite_SOM_output_dir:
        som.output_directory = output_directory
    (
        trained_subdirectory,
        trained_paths,
        map_paths,
        neighborhood_radii,
    ) = retrieve_trained_som_paths(som)
    som.trained_subdirectory = trained_subdirectory
    print("retrieved paths. Time taken:", round(time.time() - start))
    start = time.time()

    # Get mapping for each epoch
    data_maps = load_som_training_dataset_mapping(
        som,
        trained_paths,
        map_paths,
        version=version,
        verbose=False,
        replace_nans=replace_nans,
    )
    print("Shape of data_maps:", np.shape(data_maps))
    print("Shape of chosen data_map:", np.shape(data_maps[force_chosen_som_index]))
    print("loaded data_maps. Time taken:", round(time.time() - start))
    start = time.time()

    # Get list of AQE and TE for each epoch
    (
        aqe_list,
        aqe_std_dev_list,
        aqe_norm_list,
        aqe_norm_std_dev_list,
        aqe_quart_list,
        aqe_fifth_list,
        aqe_sev_list,
        te_list,
    ) = calculate_AQEs_and_TEs(
        som,
        output_directory,
        trained_subdirectory,
        data_maps,
        trained_paths,
        overwrite=overwrite,
    )
    print("get aqes and tes. Time taken:", round(time.time() - start))
    start = time.time()

    # Plot the AQEs and TEs
    if not force_chosen_som_index is None:
        print(f"Warning: Using forced chosen SOM index {force_chosen_som_index}")
        _ = plot_aqe_and_te_choose_som(
            som,
            figures_dir,
            aqe_list,
            aqe_std_dev_list,
            aqe_norm_list,
            aqe_norm_std_dev_list,
            aqe_quart_list,
            aqe_fifth_list,
            aqe_sev_list,
            te_list,
            neighborhood_radii,
            plot_normed=False,
            version=version,
            AQE_stopping_condition_percent=AQE_stopping_condition_percent,
            focus_on_TE=focus_on_TE,
            force_chosen_som_index=force_chosen_som_index,
        )
        chosen_som_index = force_chosen_som_index
    else:
        chosen_som_index = plot_aqe_and_te_choose_som(
            som,
            figures_dir,
            aqe_list,
            aqe_std_dev_list,
            aqe_norm_list,
            aqe_norm_std_dev_list,
            aqe_quart_list,
            aqe_fifth_list,
            aqe_sev_list,
            te_list,
            neighborhood_radii,
            plot_normed=False,
            version=version,
            AQE_stopping_condition_percent=AQE_stopping_condition_percent,
            focus_on_TE=focus_on_TE,
        )
    print("plot aqe and te. Time taken:", round(time.time() - start))
    start = time.time()

    # Take last trained file to display
    trained_path = trained_paths[chosen_som_index]
    map_path = map_paths[chosen_som_index]
    som.trained_path = trained_path
    som.map_path = map_path
    print("trained_path:", trained_path)
    print("map_path:", map_path)

    # Create som object and add unpacked mapping to catalogue
    (
        catalogue_final,
        distance_to_bmu_sorted_down_id,
        closest_prototype_id,
        prototype_x,
        prototype_y,
    ) = unpack_som_and_enrich_catalogue(
        som,
        chosen_som_index,
        data_maps,
        trained_subdirectory,
        data_directory,
        output_directory,
        trained_path,
        version=version,
        replace_nans=replace_nans,
        flip_axis0=flip_axis0,
        flip_axis1=flip_axis1,
        rot90=rot90,
        catalogue_name=catalogue_name,
    )

    if short_stop:
        return (
            data_maps[chosen_som_index],
            som.data_som,
            trained_path,
            closest_prototype_id,
            catalogue_final,
        )

    # Plot som
    if som.number_of_channels > 1:
        plot_som_3D(
            som,
            gap=2,
            save=save,
            save_name=f"som_ID{run_id}",
            save_dir=figures_dir,
            normalize=normalize,
            overwrite=overwrite,
            cmap="viridis",
            align_prototypes=align_prototypes,
            zoom_in=zoom_in,
            trained_path=trained_path,
            version=version,
        )
        # If SOM has depth==2 (indicative of a two-channel SOM)
        # We also plot the second channel as contourlines on top of the first channel
        plot_som_3D_contour(
            som,
            gap=2,
            save=save,
            save_name=f"som_ID{run_id}",
            save_dir=figures_dir,
            normalize=normalize,
            overwrite=overwrite,
            replace_nans=replace_nans,
            compress=compress,
            cmap="viridis",
            align_prototypes=align_prototypes,
            zoom_in=zoom_in,
            trained_path=trained_path,
            version=version,
        )
    else:
        plot_som(
            som,
            gap=2,
            save=save,
            save_name=f"som_ID{run_id}",
            save_dir=figures_dir,
            normalize=normalize,
            overwrite=overwrite,
            compress=compress,
            replace_nans=replace_nans,
            cmap="viridis",
            align_prototypes=align_prototypes,
            zoom_in=zoom_in,
            trained_path=trained_path,
            version=version,
        )

    # Plot heatmap
    plot_som_bmu_heatmap(
        som,
        None,
        data_map=data_maps[chosen_som_index],
        cbar=False,
        save=save,
        save_dir=figures_dir,
        compress=compress,
        save_name="som_ID{}_heatmap".format(run_id),
    )
    if som.number_of_channels > 1:
        return (
            som,
            data_maps[chosen_som_index],
            som.data_som,
            trained_path,
            distance_to_bmu_sorted_down_id,
            closest_prototype_id,
            catalogue_final,
            None,
        )

    # Calculate AQE per prototype/node
    if verbose:

        (
            AQE_per_node,
            AQE_std_per_node,
            QE_per_node,
        ) = calculate_normalized_AQE_per_prototype(
            data_maps[chosen_som_index], som.rotated_size, verbose=verbose
        )
        check_percentage_below_threshold(
            som, data_maps[chosen_som_index], fit_threshold
        )
        return_cutouts_per_percentile_per_prototype(
            som,
            data_directory,
            data_maps[chosen_som_index],
            QE_per_node,
            threshold=fit_threshold,
        )

    # Plot umatrix
    u_matrix_selected_prototypes = plot_u_matrix(
        som,
        trained_path,
        output_directory,
        cbar=True,
        save=save,
        replace_nans=replace_nans,
        save_dir=figures_dir,
        mask=False,
        version=version,
        mask_threshold=1.3,
        max_tiles=10,
    )

    return (
        som,
        data_maps[chosen_som_index],
        som.data_som,
        trained_path,
        distance_to_bmu_sorted_down_id,
        closest_prototype_id,
        catalogue_final,
        u_matrix_selected_prototypes,
    )


def train_SOM_v2(
    GPU_ID,
    seed,
    GAUSS,
    SIGMA,
    damping,
    RotatedSize,
    EPOCHS_PER_EPOCH,
    som_width,
    som_height,
    LAYOUT,
    PBC,
    init,
    data_bin,
    train_bin,
    simple=False,
):
    if PBC:
        raise NotImplementedError("PBC is not yet implemented in Pinkv2.")
        PBC = "--pbc"
    else:
        PBC = ""
    if simple:
        return f"CUDA_VISIBLE_DEVICES={GPU_ID} Pink  --euclidean-distance-type float --verbose -p 1 --train {data_bin} {train_bin}"
    else:
        return (
            f"CUDA_VISIBLE_DEVICES={GPU_ID} Pink --euclidean-distance-type float --seed {seed} --dist-func gaussian {GAUSS}"
            f" {SIGMA} {damping} --inter-store overwrite --neuron-dimension {RotatedSize} --numrot 360"
            f" --num-iter {EPOCHS_PER_EPOCH} --progress 10 --som-width {som_width}  --som-height"
            f" {som_height} --layout {LAYOUT} {PBC} --init {init} --train {data_bin} {train_bin}"
        )


def flip_datamap(
    data_map, som_width, som_height, flip_axis0=False, flip_axis1=False, rot90=False
):
    """flip datamap"""
    datmap = copy.deepcopy(data_map)
    s1, s2 = datmap.shape
    # Note the transpose
    datmap = datmap.T.reshape(som_width, som_height, s1)
    if flip_axis0:
        print("Note: we are now flipping the data map around axis 0 before plotting!")
        datmap = np.flip(datmap, axis=0)
    if flip_axis1:
        print("Note: we are now flipping the data map around axis 1 before plotting!")
        datmap = np.flip(datmap, axis=1)
    if rot90:
        print("Note: we are now rotating the data map 90deg before plotting!")
        datmap = np.rot90(datmap)
    datmap = datmap.reshape(som_width * som_height, s1)
    return datmap.T


def get_sorted_outliers_df_DR1(
    catalogue,
    output_directory,
    save_name="sorted_outliers_indices_DR1.h5",
    verbose=False,
    overwrite=True,
    version=1,
    metric="Distance_to_bmu",
    masx_store_dir="",
    name_id=None,
):
    """
    Loads in a pandas dataframe catalogue
    Sorts this dataframe based on this distance (descending) and
    saves this dataframe as hdf5 file.
    future: Also checks if radio objects are located in 2MASX object or inside a known cluster
    """
    assert version == 1
    assert save_name.endswith(".h5")
    save_outliers_file = os.path.join(output_directory, save_name)
    # Create dictionary to and from datapaths

    if overwrite or not os.path.exists(save_outliers_file):
        # Initialize dataframe
        # df = pd.DataFrame(columns=['distance_to_bmu','bmu_id','cutout_id','datarow_id',
        #    'dist_to_centre_deg','2MASX','SDSS_galaxy_cluster'])

        # Sort cat on outlier score
        cat = copy.deepcopy(catalogue)
        cat = cat.reset_index()
        sorted_df = cat.sort_values(metric, ascending=False)
        # Save file
        sorted_df.to_hdf(save_outliers_file, "df")


def get_sorted_outliers_df(
    som,
    output_directory,
    paths_to_mapping_results,
    pointings_df,
    save_name="sorted_outliers_indices.h5",
    verbose=False,
    overwrite=True,
    version=1,
    masx_store_dir="",
    name_id=None,
):
    """
    Loads in for every pointing that is mapped to a specific SOM its SOM-mapping.
    Adds all the distances for every source in the pointing to its best matching unit to a
    combined dataframe. Sorts this dataframe based on this distance (descending) and
    saves this combined dataframe as hdf5 file.
    Also checks if radio objects are located in 2MASX object or inside a known cluster
    """
    assert save_name.endswith(".h5")
    save_outliers_file = os.path.join(output_directory, save_name)
    # Create dictionary to and from datapaths
    index_to_datarow_dict = {
        datarow_index: datarow for (datarow_index, datarow) in pointings_df.iterrows()
    }
    datarow_to_index_dict = {
        datarow.dir_path: datarow_index
        for (datarow_index, datarow) in pointings_df.iterrows()
    }

    print(save_outliers_file)
    print(os.path.exists(save_outliers_file))
    if overwrite or not os.path.exists(save_outliers_file):
        # Initialize dataframe
        df = pd.DataFrame(
            columns=[
                "distance_to_bmu",
                "bmu_id",
                "cutout_id",
                "datarow_id",
                "dist_to_centre_deg",
                "2MASX",
                "SDSS_galaxy_cluster",
            ]
        )

        print(f"Iterating over {len(paths_to_mapping_results)} pointings.")

        # Iterate over pointings and append datamaps to the dataframe
        for i, ((datarow_index, datarow), map_path) in enumerate(
            zip(pointings_df.iterrows(), paths_to_mapping_results)
        ):
            # Load som mapping
            data_map, number_of_images, _, _, _ = load_som_mapping(
                map_path, som, verbose=verbose, version=version
            )
            distance_to_bmu = np.min(data_map, axis=1)
            closest_prototype_id = np.argmin(data_map, axis=1)

            # Load pointing header to extract its center
            central_c = hdr_to_central_skycoord(datarow.mosaic_path)
            cat = pd.read_hdf(datarow.cat_path, "df")
            print("cat path", datarow.cat_path)
            # cat['idx_all'] = cat.index
            # cat['skycoord'] = SkyCoord(cat.RA, cat.DEC, unit="degree")
            c = SkyCoord(cat.RA, cat.DEC, unit="degree")
            dists = central_c.separation(c).deg

            # Check which radio objects fall inside a 2MASX object
            MASX_column = match_to_2MASX_catalogue(
                cat, store_dir=masx_store_dir, name_id=name_id
            )

            # Check which radio objects fall inside a galaxy cluster
            SDSS_column = match_to_SDSS_cluster_catalogue(
                cat, store_dir=masx_store_dir, name_id=name_id
            )

            # Create pandas dataframe
            print(
                f"{i}/{len(pointings_df)}: n_images {number_of_images} shape datamap {np.shape(data_map)} p_name   {datarow.pointing_name}"
            )
            print(
                f"{i}/{len(pointings_df)}: n_images {number_of_images} shape datamap "
                f"{np.shape(data_map)} p_name   {datarow.pointing_name}",
                file=open(f"sort_progress_{name_id}.txt", "a"),
            )
            print(f"shape of dists is {np.shape(dists)}")
            try:

                test = pd.DataFrame(
                    data={
                        "distance_to_bmu": distance_to_bmu,
                        "bmu_id": closest_prototype_id,
                        "cutout_id": [t for t in range(number_of_images)],
                        "datarow_id": [datarow_index for _ in range(number_of_images)],
                        "dist_to_centre_deg": dists,
                        "2MASX": MASX_column,
                        "SDSS_galaxy_cluster": SDSS_column,
                    }
                )
                df = pd.concat([df, test], ignore_index=True, sort=False)
            except:
                print(
                    f"lengths do not match, try rerunning the pipeline for this field"
                )
                sdfsdf

        # distance_to_bmu, cutout_id, bin_path
        if verbose:
            som.print()
            print(df.info())
            print(df.head())

        # Sort list on distance_to_bmu
        sorted_df = df.sort_values("distance_to_bmu", ascending=False)
        # Save file
        sorted_df.to_hdf(save_outliers_file, "df")


def get_sorted_outliers_single_field(
    som,
    output_directory,
    map_path,
    version=1,
    save_name="sorted_outliers_indices.h5",
    verbose=True,
    overwrite=True,
):
    """
    Loads in for every pointing that is mapped to a specific SOM its SOM-mapping.
    Adds all the distances for every source in the pointing to its best matching unit to a
    combined dataframe. Sorts this dataframe based on this distance (descending) and
    saves this combined dataframe as hdf5 file.
    """
    assert save_name.endswith(".h5")
    save_outliers_file = os.path.join(output_directory, save_name)

    if overwrite or not os.path.exists(save_outliers_file):
        som.print()

        # Iterate over pointings and append datamaps to the dataframe
        data_map, number_of_images, _, _, _ = load_som_mapping(
            map_path, som, version=version, verbose=verbose
        )
        print("Number of images in field:", number_of_images)
        distance_to_bmu = np.min(data_map, axis=1)
        closest_prototype_id = np.argmin(data_map, axis=1)

        # Create pandas dataframe
        df = pd.DataFrame(
            data={
                "distance_to_bmu": distance_to_bmu,
                "bmu_id": closest_prototype_id,
                "cutout_id": [t for t in range(number_of_images)],
            }
        )

        # Sort list on distance_to_bmu
        sorted_df = df.sort_values("distance_to_bmu", ascending=False)
        # Save file
        sorted_df.to_hdf(save_outliers_file, "df")


def plot_outliers_single_fits(
    som,
    field_name,
    field_local_dir,
    field_dir,
    output_directory,
    fullsize,
    verbose=False,
    exclude_pointings_list=[],
    version=1,
    plot_large=True,
    cutout_size_in_arcsec=400,
    arcsec_per_pixel=1.5,
    plot_bmu_heatmap=True,
    fontsize=20,
    cbar=True,
    suppress_cutouts_closer_than_x_arcseconds=50,
    sorted_name="sorted_outliers_indices.h5",
    save=False,
    heatmap_save_name="heatmap",
    save_dir="",
    stop_at=5,
    lower_sigma_limit=1.5,
    upper_sigma_limit=100,
    overwrite=False,
    sqrt_stretch=True,
    **kwargs,
):
    """
    Given a source catalogue sorted on most outlying sources and a binary file, plots the outliers.
    """

    # Load sorted df
    assert sorted_name.endswith(".h5")
    save_outliers_file = os.path.join(output_directory, sorted_name)
    sorted_df = pd.read_hdf(save_outliers_file, "df")
    if verbose:
        print(sorted_df.head())
    print("Length of catalog is:", len(sorted_df))

    # plot bmu heatmap
    if plot_bmu_heatmap:

        count = np.bincount(pd.to_numeric(sorted_df.bmu_id))
        count = np.concatenate(
            [count, np.zeros(som.som_width * som.som_height - len(count), dtype=int)]
        )
        count = count.reshape([som.som_width, som.som_height])

        # plot and save the heatmap
        fig = plt.figure(figsize=(14, 14))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax = sns.heatmap(
            count.T,
            annot=True,
            fmt="d",
            cmap="inferno",
            square=True,
            annot_kws={"size": fontsize},
            cbar=cbar,
            cbar_kws={"shrink": 0.8},
        )  # , linewidths=.1)
        # cbar_kws={'label':'# of sources assigned to prototype'})#, linewidths=.1)

        if save:
            plt.savefig(os.path.join(save_dir, heatmap_save_name + ".png"))
        else:
            plt.show()
        plt.close()

    # Suppress sources close to each other (mostly duplicates)
    ras, decs = [], []
    duplicate_list = [False]
    excluded = 0
    extraction_failed = 0
    for i, (df_index, df_datarow) in enumerate(sorted_df.iterrows()):
        print(df_datarow)
        if i > stop_at:
            break

        # Skip if pointing is in exclude pointingslist
        if field_name in exclude_pointings_list:
            stop_at += 1
            excluded += 1
            duplicate_list.append(False)
            continue

        # Check for all sources that are greater outliers if they are close to you
        pd_catalogue = pd.read_hdf(
            os.path.join(
                field_local_dir, "cutout_list_clipped_True_mosaic.cat.extracted.h5"
            ),
            "df",
        )
        ra = pd_catalogue["RA"].iloc[int(df_datarow.cutout_id)]
        dec = pd_catalogue["DEC"].iloc[int(df_datarow.cutout_id)]
        rms = pd_catalogue["cutout_rms"].iloc[int(df_datarow.cutout_id)]

        if i > 0:
            cat = SkyCoord(ra=ras, dec=decs, unit=(u.deg, u.deg), frame="fk5")
            c = SkyCoord(ra=ra, dec=decs, unit=(u.deg, u.deg), frame="fk5")
            closest_neighbour_separation_in_arcsec = match_coordinates_sky(c, cat)[
                1
            ].arcsec
            # kdtree = cKDTree(ra_dec_list)
            # close_cutouts = kdtree.query_ball_point([ra,dec],
            #        suppress_cutouts_closer_than_x_arcseconds/3600, eps=0/3600) # ~arcseconds, Keep in mind that we are looking at Euclidean
            # distance while we should actually be looking for great circle distance
            # True if close_cutouts is an empty list
            if (
                closest_neighbour_separation_in_arcsec
                > suppress_cutouts_closer_than_x_arcseconds
            ):
                duplicate_list.append(False)
            else:
                duplicate_list.append(True)
                stop_at += 1

        ras.append(ra)
        decs.append(dec)

        if not duplicate_list[i] and plot_large:
            # Create larger cutouts

            image_list = []
            outlier_cutout_size = cutout_size_in_arcsec / arcsec_per_pixel
            extract_attempt_counter = 0
            while len(image_list) == 0 and outlier_cutout_size > 0.8 * fullsize:
                single_source_catalogue = pd.DataFrame(
                    data={"RA": [ra], "DEC": [dec], "cutout_rms": [rms]}
                )
                (
                    image_list,
                    single_source_catalogue,
                ) = single_fits_to_numpy_cutouts_using_astropy_better(
                    outlier_cutout_size,
                    single_source_catalogue,
                    "RA",
                    "DEC",
                    field_dir,
                    "mosaic-blanked.fits",
                    apply_clipping=True,
                    apply_mask=False,
                    verbose=False,
                    store_directory=field_local_dir,
                    mode="partial",
                    store_file=f"outlier_{int(df_datarow.cutout_id)}",
                    dimensions_normal=True,
                    variable_size=False,
                    hdf=True,
                    rescale=True,
                    sqrt_stretch=sqrt_stretch,
                    destination_size=None,
                    lower_sigma_limit=lower_sigma_limit,
                    upper_sigma_limit=upper_sigma_limit,
                    arcsec_per_pixel=arcsec_per_pixel,
                    overwrite=overwrite,
                )
                outlier_cutout_size *= 0.8
                if len(image_list) == 0:
                    print(
                        f"Cutout extraction failed, trying again with dimensions"
                        f"{outlier_cutout_size*arcsec_per_pixel:.2f}x{outlier_cutout_size*arcsec_per_pixel:.2f}"
                        f" arcsec."
                    )
                extract_attempt_counter += 1

            if len(image_list) == 1:
                fig, ax = plt.subplots(figsize=(10, 10))
                # plt.figure(figsize=(10,10))
                ax.imshow(image_list[0], interpolation="nearest")
                # Plot square view window of size equal to original SOM-fed cutout
                f = (
                    outlier_cutout_size * 1.25
                )  # as we downscaled with 0.8 in the while loop above
                plot_red_square = False
                if plot_red_square:
                    ax.plot(
                        [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                        [f / 2 + fullsize / 2, f / 2 + fullsize / 2],
                        "-r",
                        linewidth=1,
                    )
                    ax.plot(
                        [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                        [f / 2 - fullsize / 2, f / 2 - fullsize / 2],
                        "-r",
                        linewidth=1,
                    )
                    ax.plot(
                        [f / 2 + fullsize / 2, f / 2 + fullsize / 2],
                        [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                        "-r",
                        linewidth=1,
                    )
                    ax.plot(
                        [f / 2 - fullsize / 2, f / 2 - fullsize / 2],
                        [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                        "-r",
                        linewidth=1,
                    )

                def format_func(value, tick_number):
                    # Go from pixelsize to arcmin
                    return value * arcsec_per_pixel / 60

                ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
                ax.set_xlabel("arcmin")
                ax.set_ylabel("arcmin")

                enable_hmsdms_label = False
                if enable_hmsdms_label:
                    hmsdms_label = SkyCoord(ra=ra, dec=dec, unit=u.degree).to_string(
                        "hmsdms"
                    )
                    ax.text(
                        f - 2,
                        30,
                        hmsdms_label,
                        color="white",
                        horizontalalignment="right",
                    )
                digits = 3
                ax.text(
                    f - 2,
                    20,
                    f"Field name: {field_name}\nRA & DEC in degree:\n{ra:.{digits}f};"
                    f" {dec:.{digits}f}",
                    color="white",
                    horizontalalignment="right",
                )

                if save:
                    plt.savefig(os.path.join(save_dir, f"{i:04d}_{field_name}.png"))
                else:
                    plt.show()
                plt.close()
            elif len(image_list) == 0:
                stop_at += 1
                extraction_failed += 1
            if verbose:
                print("RA & DEC (degree):", f"{ra:.4f}, {dec:.4f}")
                print(
                    "Mosaic path:", field_dir, "Cutout-ID:", int(df_datarow.cutout_id)
                )

    if verbose:
        print("duplicate_list:", duplicate_list)
    print(
        (
            f"Suppressed {sum(duplicate_list)} cutouts because they are approximately closer than "
            f"{suppress_cutouts_closer_than_x_arcseconds} arcsec to more outlying cutouts.\n"
            f"Suppressed {excluded} cutouts because they are in the exclude pointing list.\n"
            f"Suppressed {extraction_failed} cutouts because their cutout extraction failed.\n"
            f"Considered {stop_at} catalogue entries in total.\n\n"
        )
    )


def plot_outliers_DR1(
    output_directory,
    verbose=False,
    max_outliers_to_return=5,
    suppress_cutouts_closer_than_x_arcseconds=200,
    suppress_cutouts_fainter_than_x_peak_flux=0,
    suppress_cutouts_fainter_than_x_total_flux=0,
    suppress_total_to_peak_flux_ratio=-1e9,
    groups=[],
    outliers_per_group=10,
    som_height=None,
    sorted_name="sorted_outliers_indices_DR1.h5",
    overwrite=False,
):
    """
    Given a source catalogue sorted on most outlying sources returns outliers with a separation
    bigger than suppress_cutouts_closer_than_x_arcseconds
    """

    # Load sorted df
    assert sorted_name.endswith(".h5")
    save_outliers_file = os.path.join(output_directory, sorted_name)
    sorted_df = pd.read_hdf(save_outliers_file, "df")
    if verbose:
        print(sorted_df.head())
    print("Length of catalog is:", len(sorted_df))
    index_path = f"outliers_DR1_with{suppress_cutouts_closer_than_x_arcseconds}arcsec_separation.npy"
    dup_path = f"outliers_DR1_with{suppress_cutouts_closer_than_x_arcseconds}arcsec_separation_duplicate_list.npy"
    faint_path = f"outliers_DR1_with{suppress_cutouts_closer_than_x_arcseconds}arcsec_separation_faint_list.npy"
    groups_path = f"outliers_DR1_per_group_with{suppress_cutouts_closer_than_x_arcseconds}arcsec_separation.npy"

    if not os.path.exists(index_path) or overwrite:
        # Suppress sources close to each other (mostly duplicates)
        ra_dec_list = []
        group_outliers = [[] for _ in range(len(groups))]
        groups = [[(y * som_height) + x for x, y in group] for group in groups]
        duplicate_list = [False]
        duplicate_indices, faint_indices = [], []
        success = -1
        faint_sum = 0
        success_list = []
        for i, (df_index, df_datarow) in enumerate(sorted_df.iterrows()):
            if verbose:
                print(f"Consider entry {i}")
            if i >= max_outliers_to_return:
                break

            # Skip this datarow if the radiosource best belongs to a archetype of sources
            # for which we already found enough outliers
            if not groups == []:
                group_full = False
                for g_i, group in enumerate(groups):
                    if df_datarow["Closest_prototype"] in group:
                        g_now = g_i
                        if len(group_outliers[g_i]) >= outliers_per_group:
                            group_full = True
                        break
                if group_full:
                    duplicate_list.append(False)
                    max_outliers_to_return += 1
                    continue

            ra = df_datarow.RA
            dec = df_datarow.DEC

            # Check for all sources that are greater outliers if they are close to you
            if success > -1:
                if verbose:
                    print(ra, dec)
                cat = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="fk5")

                kdtree = cKDTree(ra_dec_list)
                close_cutouts = kdtree.query_ball_point(
                    [ra, dec],
                    suppress_cutouts_closer_than_x_arcseconds / 3600,
                    eps=0 / 3600,
                )  # ~arcseconds, Keep in mind that we are looking at Euclidean
                # distance while we should actually be looking for great circle distance
                if not close_cutouts:  # True if close_cutouts is an empty list
                    duplicate_list.append(False)
                    if (
                        (
                            df_datarow.Peak_flux
                            < suppress_cutouts_fainter_than_x_peak_flux
                        )
                        or (
                            df_datarow.Total_flux
                            < suppress_cutouts_fainter_than_x_total_flux
                        )
                        or (
                            df_datarow.Total_flux / df_datarow.Peak_flux
                            < suppress_total_to_peak_flux_ratio
                        )
                    ):
                        faint_sum += 1
                        faint_indices.append(df_index)
                        max_outliers_to_return += 1
                        continue
                else:
                    duplicate_list.append(True)
                    duplicate_indices.append(df_index)
                    max_outliers_to_return += 1
                    continue
            if not duplicate_list[i]:
                success_list.append(df_index)
                ra_dec_list.append([ra, dec])
                if not groups == []:
                    group_outliers[g_now].append(df_index)
            success += 1

        print(f"Considered {max_outliers_to_return} catalogue entries in total.")
        print(
            f"{faint_sum} sources were skipped because their peak flux was <"
            f" {suppress_cutouts_fainter_than_x_peak_flux} mJy/beam and their total flux <"
            f"{suppress_cutouts_fainter_than_x_total_flux} mJy."
        )
        # Save cat with successfully extracted entries
        success_list = np.array(success_list)
        np.save(index_path, success_list)
        np.save(dup_path, duplicate_indices)
        np.save(faint_path, duplicate_indices)
        if not groups == []:
            np.save(groups_path, group_outliers)
    else:
        print(
            "Index list with outliers with minimum distance already exists. Loading it in..."
        )
        success_list = np.load(index_path)
        duplicate_indices = np.load(dup_path)
        faint_indices = np.load(faint_path)
        if not groups == []:
            group_outliers = np.load(groups_path)

    print(
        (
            f"Suppressed {len(duplicate_indices)} cutouts because they are approximately closer than "
            f"{suppress_cutouts_closer_than_x_arcseconds} arcsec to more outlying cutouts.\n"
        )
    )
    return success_list, duplicate_indices, faint_indices, group_outliers


def plot_combined_outliers(
    som,
    pointings_df,
    output_directory,
    cutouts_bin_name,
    fullsize,
    verbose=False,
    include_pointings_list=None,
    exclude_pointings_list=[],
    cutout_size_in_arcsec=400,
    arcsec_per_pixel=1.5,
    only_ra_dec=False,
    ra_dec_path=None,
    field_image_name="mosaic-blanked",
    dimensions_normal=True,
    max_distance_to_centre_deg=1e9,
    name_id=1,
    plot_bmu_heatmap=True,
    fontsize=20,
    cbar=True,
    suppress_cutouts_closer_than_x_arcseconds=50,
    sorted_name="sorted_outliers_indices.h5",
    save=False,
    heatmap_save_name="heatmap",
    save_dir="",
    stop_at=5,
    lower_sigma_limit=1.5,
    upper_sigma_limit=100,
    overwrite=False,
    sqrt_stretch=True,
    **kwargs,
):
    """
    Given a source catalogue sorted on most outlying sources and a binary file, plots the outliers.
    If include_pointings_list is not None, pointing must me inside this list.
    if exclude_pointings_list is not empty, pointings may not be in this list (higher prio than
        include list).
    """

    # Load sorted df
    assert sorted_name.endswith(".h5")
    save_outliers_file = os.path.join(output_directory, sorted_name)
    sorted_df = pd.read_hdf(save_outliers_file, "df")
    if verbose:
        print(sorted_df.head())
    print("Length of catalog is:", len(sorted_df))

    # plot bmu heatmap
    if plot_bmu_heatmap:

        count = np.bincount(pd.to_numeric(sorted_df.bmu_id))
        count = np.concatenate(
            [count, np.zeros(som.som_width * som.som_height - len(count), dtype=int)]
        )
        count = count.reshape([som.som_width, som.som_height])

        # plot and save the heatmap
        fig = plt.figure(figsize=(14, 14))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax = sns.heatmap(
            count.T,
            annot=True,
            fmt="d",
            cmap="inferno",
            square=True,
            annot_kws={"size": fontsize},
            cbar=cbar,
            cbar_kws={"shrink": 0.8},
        )  # , linewidths=.1)
        # cbar_kws={'label':'# of sources assigned to prototype'})#, linewidths=.1)

        if save:
            plt.savefig(os.path.join(save_dir, heatmap_save_name + ".png"))
        else:
            plt.show()
        plt.close()

    # Create dictionary to and from datapaths
    index_to_dir_path_dict = {
        datarow_id: datarow.dir_path
        for (datarow_id, datarow) in pointings_df.iterrows()
    }
    index_to_pointing_name_dict = {
        datarow_id: datarow.pointing_name
        for (datarow_id, datarow) in pointings_df.iterrows()
    }

    # Suppress sources close to each other (mostly duplicates)
    ra_dec_list = []
    duplicate_list = [False]
    success = -1
    success_list = []
    excluded = 0
    excluded_to_far_from_center = 0
    extraction_failed = 0
    with open(ra_dec_path, "w") as f:
        f.writelines(["RA, DEC, Field_name"])
    for i, (df_index, df_datarow) in enumerate(sorted_df.iterrows()):
        if verbose:
            print(f"Consider entry {i}")
        if i > stop_at:
            break

        # Skip if pointing is not in include list or if its in exclude pointingslist
        current_pointing = index_to_pointing_name_dict[df_datarow.datarow_id]
        if not include_pointings_list is None:
            if not current_pointing in include_pointings_list:
                stop_at += 1
                excluded += 1
                duplicate_list.append(False)
                continue
        if current_pointing in exclude_pointings_list:
            stop_at += 1
            excluded += 1
            duplicate_list.append(False)
            continue

        # print(os.path.join(index_to_dir_path_dict[df_datarow.datarow_id],
        #        'cutout_list_clipped_True_mosaic.cat.extracted.h5'))
        pd_catalogue = pd.read_hdf(
            os.path.join(
                index_to_dir_path_dict[df_datarow.datarow_id],
                "cutout_list_clipped_True_mosaic.cat.extracted.h5",
            ),
            "df",
        )
        # print('extracted cat len:',len(pd_catalogue))
        # print('cutout_id:',df_datarow.cutout_id)
        ra = pd_catalogue["RA"].iloc[df_datarow.cutout_id]
        dec = pd_catalogue["DEC"].iloc[df_datarow.cutout_id]
        rms = pd_catalogue["cutout_rms"].iloc[df_datarow.cutout_id]
        dist_to_centre_deg = df_datarow.dist_to_centre_deg

        # Reject sources at certain distance from the field centre
        if dist_to_centre_deg > max_distance_to_centre_deg:
            duplicate_list.append(False)
            excluded_to_far_from_center += 1
            stop_at += 1
            continue

        # Check for all sources that are greater outliers if they are close to you
        if success > -1:
            cat = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="fk5")

            kdtree = cKDTree(ra_dec_list)
            close_cutouts = kdtree.query_ball_point(
                [ra, dec],
                suppress_cutouts_closer_than_x_arcseconds / 3600,
                eps=0 / 3600,
            )  # ~arcseconds, Keep in mind that we are looking at Euclidean
            # distance while we should actually be looking for great circle distance
            if not close_cutouts:  # True if close_cutouts is an empty list
                duplicate_list.append(False)
            else:
                duplicate_list.append(True)
                stop_at += 1
                continue

        if verbose:
            print(f"Checked to see if outlier close to prev outlier")

        if not duplicate_list[i]:
            ra_dec_list.append([ra, dec])
            if only_ra_dec:
                with open(ra_dec_path, "a") as f:
                    f.writelines(
                        [
                            f"{ra:.4f}, {dec:.4f}, {index_to_pointing_name_dict[df_datarow.datarow_id]}\n"
                        ]
                    )
                continue

            # Create larger cutouts
            image_list = []
            outlier_cutout_size = cutout_size_in_arcsec / arcsec_per_pixel
            extract_attempt_counter = 0
            while (
                len(image_list) == 0
                and outlier_cutout_size > 0.8 * fullsize
                and extract_attempt_counter < 6
            ):
                # if len(image_list) == 0 and outlier_cutout_size > 0.8*fullsize and extract_attempt_counter<6:
                with timeout(seconds=10):

                    if verbose:
                        print(
                            f"if len ima {len(image_list)}==0 and outlier_cutout_size {outlier_cutout_size} > 0.8*fullsize {0.8*fullsize}"
                        )
                    single_source_catalogue = pd.DataFrame(
                        data={"RA": [ra], "DEC": [dec], "cutout_rms": [rms]}
                    )
                    (
                        image_list,
                        single_source_catalogue,
                    ) = single_fits_to_numpy_cutouts_using_astropy_better(
                        outlier_cutout_size,
                        single_source_catalogue,
                        "RA",
                        "DEC",
                        index_to_dir_path_dict[df_datarow.datarow_id],
                        field_image_name + ".fits",
                        apply_clipping=True,
                        apply_mask=False,
                        verbose=False,
                        store_directory=index_to_dir_path_dict[df_datarow.datarow_id],
                        mode="partial",
                        store_file=f"outlier_{df_datarow.cutout_id}",
                        dimensions_normal=dimensions_normal,
                        variable_size=False,
                        hdf=True,
                        rescale=True,
                        sqrt_stretch=sqrt_stretch,
                        destination_size=None,
                        lower_sigma_limit=lower_sigma_limit,
                        upper_sigma_limit=upper_sigma_limit,
                        arcsec_per_pixel=arcsec_per_pixel,
                        overwrite=overwrite,
                    )
                    outlier_cutout_size *= 0.8
                    if len(image_list) == 0:
                        print(
                            f"Cutout extraction failed, trying again with dimensions"
                            f"{outlier_cutout_size*arcsec_per_pixel:.2f}x{outlier_cutout_size*arcsec_per_pixel:.2f}"
                            f" arcsec."
                        )
                extract_attempt_counter += 1

            if len(image_list) == 1:
                fig, ax = plt.subplots(figsize=(10, 10))
                # plt.figure(figsize=(10,10))
                ax.imshow(image_list[0], interpolation="nearest", origin="lower")
                # Plot square view window of size equal to original SOM-fed cutout
                f = (
                    outlier_cutout_size * 1.25
                )  # as we downscaled with 0.8 in the while loop above
                ax.plot(
                    [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                    [f / 2 + fullsize / 2, f / 2 + fullsize / 2],
                    "-r",
                    linewidth=1,
                )
                ax.plot(
                    [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                    [f / 2 - fullsize / 2, f / 2 - fullsize / 2],
                    "-r",
                    linewidth=1,
                )
                ax.plot(
                    [f / 2 + fullsize / 2, f / 2 + fullsize / 2],
                    [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                    "-r",
                    linewidth=1,
                )
                ax.plot(
                    [f / 2 - fullsize / 2, f / 2 - fullsize / 2],
                    [f / 2 - fullsize / 2, f / 2 + fullsize / 2],
                    "-r",
                    linewidth=1,
                )

                def format_func(value, tick_number):
                    # Go from pixelsize to arcmin
                    return value * arcsec_per_pixel / 60

                ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
                ax.set_xlabel("arcmin")
                ax.set_ylabel("arcmin")
                # Store info about successful extraction for printing table in paper
                success += 1
                success_list.append(
                    (
                        df_index,
                        ra,
                        dec,
                        index_to_pointing_name_dict[df_datarow.datarow_id],
                        df_datarow.distance_to_bmu,
                        df_datarow.bmu_id,
                        df_datarow["2MASX"],
                        df_datarow.SDSS_galaxy_cluster,
                    )
                )

                enable_hmsdms_label = True
                if enable_hmsdms_label:
                    hmsdms_label = SkyCoord(ra=ra, dec=dec, unit=u.degree).to_string(
                        "hmsdms"
                    )
                    ax.text(
                        f - 2,
                        30,
                        f"Pointing {index_to_pointing_name_dict[df_datarow.datarow_id]}\n"
                        + hmsdms_label,
                        color="white",
                        horizontalalignment="right",
                    )
                    digits = 3
                    ax.text(
                        f - 2,
                        70,
                        f"(RA; DEC) in degree\n({ra:.{digits}f};" f" {dec:.{digits}f})",
                        color="white",
                        horizontalalignment="right",
                    )

                if save:
                    plt.savefig(
                        os.path.join(
                            save_dir,
                            f"{success:04d}_{index_to_pointing_name_dict[df_datarow.datarow_id]}.png",
                        )
                    )
                else:
                    plt.show()

                with open(ra_dec_path, "a") as f:
                    f.writelines(
                        [
                            f"{ra:.4f}, {dec:.4f}, {index_to_pointing_name_dict[df_datarow.datarow_id]}\n"
                        ]
                    )

                plt.close()
            elif len(image_list) == 0:
                stop_at += 1
                extraction_failed += 1
            if verbose:
                print("RA & DEC (degree):", f"{ra:.4f}, {dec:.4f}")
                print(
                    "Mosaic path:",
                    index_to_dir_path_dict[df_datarow.datarow_id],
                    "Cutout-ID:",
                    df_datarow.cutout_id,
                )

    # Save cat with successfully extracted entries
    success_list = np.array(success_list)
    successfully_extracted = pd.DataFrame(
        {
            "N": [j for j in range(len(success_list))],
            "df_index": success_list[:, 0],
            "RA": success_list[:, 1],
            "DEC": success_list[:, 2],
            "Field name": success_list[:, 3],
            "Outlier score": success_list[:, 4],
            "BMU": success_list[:, 5],
            "2MASX": success_list[:, 6],
            "SDSS galaxy cluster": success_list[:, 7],
        }
    )
    # (df_index,ra,dec,index_to_pointing_name_dict[df_datarow.datarow_id],
    #            df_datarow.distance_to_bmu, df_datarow.bmu_id,df_datarow.2MASX,
    #            df_datarow.SDSS_galaxy_cluster))
    successfully_extracted.to_hdf(f"successfully_extracted_{name_id}.h5", "df")

    if verbose:
        print("duplicate_list:", duplicate_list)
    print(
        (
            f"Suppressed {sum(duplicate_list)} cutouts because they are approximately closer than "
            f"{suppress_cutouts_closer_than_x_arcseconds} arcsec to more outlying cutouts.\n"
            f"Suppressed {excluded} cutouts because they are in the exclude pointing list.\n"
            f"Suppressed {excluded_to_far_from_center} cutouts because they were located more than"
            f" {max_distance_to_centre_deg} deg from the center of the field.\n"
            f"Suppressed {extraction_failed} cutouts because their cutout extraction failed.\n"
            f"Considered {stop_at} catalogue entries in total.\n\n"
        )
    )


def return_nn_distance_in_arcsec(ras, decs, subset_indices=None):
    """Makes a `scipy.spatial.CKDTree` on (`ras`, `decs`)
    and return nearest neighbour distances in degrees.
    Assuming small angles, we make use of the approximation:
    angle = 2arcsin(a/2) ~= a
    For the LoTSS cat, the errors introduced with this assumption are of the order 1e-6 arcsec.
    Parameters
    ----------
    ras,decs : array-like
        The right ascension and declination coordinate pairs in decimal degrees.
    subset_indices : array-like
        Indices of the subset for which you want nn distances to the rest of the RAs and DECs
    Returns
    -------
    Nearest neighbour distances in arcsec
    Nearest neighbour indices as bonus
    """
    # Create kdtree
    xyz, ra_dec_tree = make_kdtree(ras, decs)

    # Query kdtree for nearest neighbour distances
    if subset_indices is None:
        kd_out = ra_dec_tree.query(xyz, k=2)
    else:
        kd_out = ra_dec_tree.query(xyz[tuple(subset_indices)], k=2)

    an_distances_rad = kd_out[0][:, 1]
    nn_distances_arcsec = np.rad2deg(nn_distances_rad) * 3600
    return nn_distances_arcsec, kd_out[1][:, 1]


def make_kdtree(ras, decs):
    """This makes a `scipy.spatial.CKDTree` on (`ra`, `decl`).
    Parameters
    ----------
    ras,decs : array-like
        The right ascension and declination coordinate pairs in decimal degrees.
    Returns
    -------
    `scipy.spatial.CKDTree`
        The cKDTRee object generated by this function is returned and can be
        used to run various spatial queries.
    """

    cosdec = np.cos(np.radians(decs))
    sindec = np.sin(np.radians(decs))
    cosra = np.cos(np.radians(ras))
    sinra = np.sin(np.radians(ras))
    xyz = np.column_stack((cosra * cosdec, sinra * cosdec, sindec))

    # generate the kdtree
    kdt = cKDTree(xyz, copy_data=True)

    return xyz, kdt


def return_nn_within_radius(
    ras, decs, search_around_ras, search_around_decs, radius_in_arcsec
):
    """Makes a `scipy.spatial.CKDTree` on (`ras`, `decs`)
    and return nearest neighbours within the given radius.
    Parameters
    ----------
    ras,decs : array-like
        The right ascension and declination coordinate pairs in decimal degrees.
    radius_in_arcsec : int
        search radisu for nns in arcsec
    Returns
    -------
    indexes of nearest neighbours within the given radius in arcsec
    """
    # Create kdtree
    xyz, ra_dec_tree = make_kdtree(ras, decs)

    cosdec = np.cos(np.radians(search_around_decs))
    sindec = np.sin(np.radians(search_around_decs))
    cosra = np.cos(np.radians(search_around_ras))
    sinra = np.sin(np.radians(search_around_ras))
    search_xyz = np.column_stack((cosra * cosdec, sinra * cosdec, sindec))

    # Query kdtree for nearest neighbour distances
    radius_in_rad = np.deg2rad(radius_in_arcsec / 3600)
    kd_out = ra_dec_tree.query_ball_point(search_xyz, 2 * np.sin(radius_in_rad / 2))

    return kd_out


# Get 2MASX cat


def match_to_2MASX_catalogue(
    cat,
    store_dir="",
    min_masx_radius_arcsec=15.0,
    name_id=None,
    verbose=False,
    overwrite=False,
):
    start = time.time()
    masx_path = os.path.join(store_dir, f"masx_{name_id}.csv")
    if overwrite or not os.path.exists(masx_path):
        v = Vizier(
            columns=["RAJ2000", "DEJ2000", "2MASX", "Kpa", "r.ext", "Kb/a"],
            column_filters={"r.ext": f">{min_masx_radius_arcsec}"},
            catalog="2MASX",
        )
        # catalog="2MASX")
        v.ROW_LIMIT = -1
        cat_2MASX = v.get_catalogs("VII/233")[0]
        cat_2MASX = cat_2MASX.to_pandas()
        cat_2MASX.to_csv(masx_path)
    else:
        cat_2MASX = pd.read_csv(masx_path)

    if verbose:
        print(f"2masx cat ({len(cat_2MASX)} entries) loaded:", time.time() - start)
        start = time.time()

    # Create kdtree from cat
    _, ra_dec_tree = make_kdtree(cat.RA, cat.DEC)
    if verbose:
        print("Created kdtree:", time.time() - start)
        start = time.time()

    # Transform 2masx coordinates to xyz
    masx_ras, masx_decs = cat_2MASX["RAJ2000"], cat_2MASX["DEJ2000"]
    cosdec = np.cos(np.radians(masx_decs))
    sindec = np.sin(np.radians(masx_decs))
    cosra = np.cos(np.radians(masx_ras))
    sinra = np.sin(np.radians(masx_ras))
    search_xyzs = np.column_stack((cosra * cosdec, sinra * cosdec, sindec))

    # turn search radii into radians
    radii_in_rad = np.radians(cat_2MASX["r.ext"] / 3600)
    # turn search radii into cartesian space
    radii_cartesian = 2 * np.sin(radii_in_rad / 2)
    if verbose:
        print("Transformed 2masx coordinates to Cartesian:", time.time() - start)
        start = time.time()

    # Query kdtree for nearest neighbour distances
    kd_outs = [
        ra_dec_tree.query_ball_point(search_xyz, radius_cartesian)
        for search_xyz, radius_cartesian in zip(search_xyzs, radii_cartesian)
    ]
    if verbose:
        print("Searched around all kdtree:", time.time() - start)
        start = time.time()

    # Loop over kdtree results to match radio cat to 2masx
    index_name_pairs = [
        [i, name[2:-1]] for i, name in zip(kd_outs, cat_2MASX["_2MASX"]) if not not i
    ]
    identifier = ["" for i in range(len(cat))]
    for i, name in index_name_pairs:
        for j in i:
            if isinstance(name, str):
                identifier[j] = "2MASX " + name
            else:
                identifier[j] = "2MASX " + name.decode("utf-8")
    tally = sum([1 for i in identifier if not i == ""])

    print(
        f"Created column with 2masx names ({tally}) if present for each cat entry:",
        time.time() - start,
    )
    return identifier


# SDSS galaxy cluster catalog


def match_to_SDSS_cluster_catalogue(
    cat, store_dir="", name_id=None, verbose=False, overwrite=False
):
    # Using The radius within which the mean density of a cluster is
    # 200 times of the critical density of the Universe.
    start = time.time()

    # J/ApJS/199/34/table1 Clusters of Galaxies Identified from the SDSS-III (132684 rows)
    # Get SDSS galaxy cluster catalog
    sdss_path = os.path.join(store_dir, f"sdss_{name_id}.csv")
    if overwrite or not os.path.exists(sdss_path):

        v = Vizier(
            columns=["*"],  # 'RAJ2000', 'DEJ2000',"2MASX","Kpa","r.ext","Kb/a"],
            # column_filters={"r.ext":">10.0"}, catalog="2MASX")
            catalog="J/ApJS/199/34/table1",
        )
        v.ROW_LIMIT = -1
        cat2 = v.get_catalogs("J/ApJS/199/34/table1")[0]
        cat2 = cat2.to_pandas()
        cat2.to_csv(sdss_path)
    else:
        cat2 = pd.read_csv(sdss_path)

    # Get spectoscopiredshifts and radii in Mpc
    radii_Mpc = cat2["r200"]
    redshifts = [
        zph if np.isnan(zsp) else zsp for zph, zsp in zip(cat2["zph"], cat2["zsp"])
    ]

    if verbose:
        print(f"sdss cat ({len(cat2)} entries) loaded:", time.time() - start)
        start = time.time()

    # Convert radii to angular size
    # Angular separation in arcsec corresponding to a proper kpc at redshift z.
    arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(z=redshifts)
    radii_arcsec = arcsec_per_kpc * radii_Mpc * 1e3 * u.kpc

    # Transform sdss coordinates to xyz
    cat2_ras, cat2_decs = cat2["RAJ2000"], cat2["DEJ2000"]
    cosdec = np.cos(np.radians(cat2_decs))
    sindec = np.sin(np.radians(cat2_decs))
    cosra = np.cos(np.radians(cat2_ras))
    sinra = np.sin(np.radians(cat2_ras))
    search_xyzs = np.column_stack((cosra * cosdec, sinra * cosdec, sindec))

    # turn search radii into radians
    radii_in_rad = np.radians(radii_arcsec.value / 3600)
    print(radii_in_rad)
    # turn search radii into cartesian space
    radii_cartesian = 2 * np.sin(radii_in_rad / 2)
    if verbose:
        print("Transformed sdss coordinates to Cartesian:", time.time() - start)
        start = time.time()

    # Create kdtree from cat
    _, ra_dec_tree = make_kdtree(cat.RA, cat.DEC)
    if verbose:
        print("Created kdtree:", time.time() - start)
        start = time.time()

    # Query kdtree for nearest neighbour distances
    kd_outs = [
        ra_dec_tree.query_ball_point(search_xyz, radius_cartesian)
        for search_xyz, radius_cartesian in zip(search_xyzs, radii_cartesian)
    ]
    if verbose:
        print("Searched around all kdtree:", time.time() - start)
        start = time.time()

    # Loop over kdtree results to match radio cat to 2masx
    index_name_pairs = [
        [i, name[2:-1]] for i, name in zip(kd_outs, cat2["WHL"]) if not not i
    ]
    identifier = ["" for i in range(len(cat))]
    for i, name in index_name_pairs:
        for j in i:
            if isinstance(name, str):
                identifier[j] = "WHL " + name
            else:
                identifier[j] = "WHL " + name.decode("utf-8")

    tally = sum([1 for i in identifier if not i == ""])

    print(
        f"Created column with sdss names ({tally}) if present for each cat entry:",
        time.time() - start,
    )
    return identifier


def interpolate_to_higher_resolution_array(
    high_res_array: np.ndarray, low_res_array: np.ndarray, kind: str = "linear"
) -> np.ndarray:
    """Interpolate low res 2D array to higher res 2D array.

    :param np.ndarray high_res_array: High resolution 2D array
    :param np.ndarray low_res_array: Low resolution 2D array
    :param str kind, optional: Interpolation type, can be 'linear', 'cubic', 'quintic'
        default is 'linear'.

    :raises assert: if dimensions of low_res_array > those of high_res_array

    :return np.ndarray: Interpolated version of low_res_array
    """
    w_high, h_high = np.shape(high_res_array)
    w_low, h_low = np.shape(low_res_array)
    assert w_high > w_low
    assert h_high > h_low
    f = interp2d(
        np.linspace(0, w_high, w_low),
        np.linspace(0, h_high, h_low),
        low_res_array,
        kind=kind,
        bounds_error=True,
    )
    return f(np.arange(w_high), np.arange(h_high))


def plot_cutout2D(
    cutout: np.ndarray,
    wcs=None,
    sqrt: bool = True,
    colorbar: bool = True,
    cmap: str = "viridis",
    return_fig=False,
):
    """Plot cutout2D object.

    Optionally with World Coordinate system, square-root scaling, colorbar
    or custom colormap

    :param np.ndarray cutout: 2D cutout
    :param wcs: World coordinate system, from astropy.wcs.WCS(fits_header, naxis=2), optional
    :param sqrt bool, optional: scale intensity using square root. Default is True.
    :param colorbar bool, optional: show colorbar. Default is True.
    :param cmap str, optional: colormap. Default is viridis.

    :return: figure
    """
    fig = plt.figure(figsize=(12, 12))
    if wcs is None:
        ax = plt.subplot()
        plt.xlabel("pixels")
        plt.ylabel("pixels")
    else:
        ax = plt.subplot(projection=wcs)
        plt.xlabel("RA")
        plt.ylabel("DEC")

    # Create interval object
    if sqrt:
        interval = vis.MinMaxInterval()
        vmin, vmax = interval.get_limits(cutout)
        norm = vis.ImageNormalize(vmin=vmin, vmax=vmax, stretch=vis.SqrtStretch())
    else:
        norm = None

    # Display the image
    plt.imshow(cutout, norm=norm, cmap=cmap, origin="lower")

    if colorbar:
        plt.colorbar(label="Jy/beam")

    if return_fig:
        return fig, ax
    else:
        plt.show()


def fits_to_hdf(cat_path, overwrite=False, debug=False):
    """Convert a fits catalogue to hdf5 format"""
    cat_path_hdf = cat_path.replace(".fits", ".h5")
    if overwrite or not os.path.exists(cat_path_hdf):
        # Load Fits cat
        start = time.time()
        cat = Table.read(cat_path).to_pandas()
        str_df = cat.select_dtypes([np.object])
        str_df = str_df.stack().str.decode("utf-8").unstack()
        for col in str_df:
            cat[col] = str_df[col]

        # Write to hdf5
        cat.to_hdf(cat_path_hdf, "df")

        # Test result
        if debug:
            start = time.time()
            cat2 = pd.read_hdf(cat_path_hdf, "df")
            print(cat2.info())


def RA_DEC_to_cat_entries(ra, dec, cat, nclosest=1, square=0.5):
    """Given an RA and DEC in degree return the nclosest cat entries"""
    # Limit search to small area around given RA and DEC
    subset = cat[
        (cat.DEC < dec + square)
        & (cat.DEC > dec - square)
        & (cat.RA < ra + square)
        & (cat.RA > ra - square)
    ]

    # Create kdtree
    xyz, ra_dec_tree = make_kdtree(subset.RA.values, subset.DEC.values)

    # Convert ra, dec to cartesian
    cosdec = np.cos(np.radians(dec))
    sindec = np.sin(np.radians(dec))
    cosra = np.cos(np.radians(ra))
    sinra = np.sin(np.radians(ra))
    # x = np.column_stack((cosra * cosdec, sinra * cosdec, sindec))
    x = (cosra * cosdec, sinra * cosdec, sindec)

    # Query kdtree
    kd_out = ra_dec_tree.query(x, k=nclosest, eps=0, p=2)
    # Return cat entries
    return subset.iloc[kd_out[1]]


def visualise_data_som_numpy_array(data_som):
    """Show data_som without zoomin or realign"""
    dat = copy.deepcopy(data_som)
    s = dat.shape
    r = int(np.sqrt(s[-1]))
    fig, ax = plt.subplots(s[0], s[1], figsize=(12, 12))
    for row in range(s[0]):
        for col in range(s[1]):
            ax[row, col].imshow(dat[col, row].reshape(r, r), origin="lower")
            ax[row, col].axis("off")
    plt.show()
    return dat


def plot_som_and_two_heatmaps(
    settings1,
    settings2,
    output_directory,
    map_dir,
    save=False,
    save_dir=None,
    specific_som=None,
    highlight=[],
    highlight_colors=[],
    highlight_rotatedsize=True,
    legend_list=[],
    title_append="",
    caption1=None,
    return_n_closest=1,
    annotations_for_paper=False,
    compress=True,
    caption2=None,
    caption3=None,
    save_path=None,
):
    """Plot a SOM accompanied by two heatmaps and corresponding captions"""
    # Create som object and add unpacked mapping to catalogue
    if specific_som is None:
        print("Loading large SOM mapping from path:", settings1.map_path)
        settings1.map_path = settings1.map_path.replace(
            "data2", "home/rafael/data"
        ).replace("data1", "home/rafael/data")
        (
            data_map_large,
            numberOfImages_large,
            som_width,
            som_height,
            som_depth,
        ) = load_som_mapping(
            settings1.map_path,
            settings1.som,
            version=2,
            verbose=False,
            compress=compress,
        )
    else:
        print("Loading large SOM mapping from path:", specific_som.map_path)
        (
            data_map_large,
            numberOfImages_large,
            som_width,
            som_height,
            som_depth,
        ) = load_som_mapping(
            specific_som.map_path,
            specific_som.som,
            version=2,
            verbose=False,
            compress=compress,
        )

    print("Loading remnant SOM mapping from path:", settings1.map_path)
    if settings2.map_path is None:
        settings2.map_path = os.path.join(
            map_dir,
            f"{settings2.store_filename}_ID{settings2.run_id}_mapped_to_ID{settings2.map_to_run_id}.bin",
        )
    settings2.map_path = settings2.map_path.replace(
        "data2", "home/rafael/data"
    ).replace("data1", "home/rafael/data")
    (
        data_map_remnant,
        numberOfImages_remnant,
        som_width,
        som_height,
        som_depth,
    ) = load_som_mapping(
        settings2.map_path, settings2.som, version=2, verbose=False, compress=compress
    )

    print("\nExperiment:", settings1.experiment)
    print("We mapped this many remnants to the SOM:", np.shape(data_map_remnant))
    print("We mapped this many large sources to the SOM:", np.shape(data_map_large))
    # Create heatmap to show where on the map the remnants end up
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 15))
    plt.subplots_adjust(wspace=0.04)

    if not specific_som is None:
        trained_som = copy.deepcopy(specific_som.som)
    else:
        try:
            trained_som = load_SOM(
                "/home/rafael/data/mostertrij/data/lockman/output",
                settings1.map_to_run_id,
            )
        except:
            trained_som = load_SOM(output_directory, settings1.map_to_run_id)
    print("Normalize som:", settings1.normalize)
    if trained_som.number_of_channels == 1:
        plot_som(
            trained_som,
            gap=2,
            save=False,
            save_name=f"som_ID{settings1.map_to_run_id}",
            save_dir=save_dir,
            ax=ax1,
            normalize=settings2.normalize,
            overwrite=True,
            compress=compress,
            # highlight=highlight,highlight_colors=highlight_colors,
            legend_list=legend_list,
            highlight_rotatedsize=highlight_rotatedsize,
            cmap="viridis",
            align_prototypes=True,
            zoom_in=settings1.zoom_in,
            trained_path=settings1.map_to_binpath,
            version=2,
        )
    else:
        print("Plotting multichannel SOM using contours for the second channel.")
        plot_som_3D_contour(
            trained_som,
            gap=2,
            save=False,
            save_name="",
            save_dir="",
            ax=ax1,
            normalize=settings2.normalize,
            overwrite=True,
            replace_nans=True,
            compress=compress,
            cmap="viridis",
            align_prototypes=True,
            zoom_in=settings1.zoom_in,
            trained_path=settings1.map_to_binpath,
            version=2,
        )
    print("Plotting remnant mapping")
    trained_som.som_width = som_width
    trained_som.som_height = som_height
    _, counts1 = plot_som_bmu_heatmap(
        trained_som,
        None,
        data_map=data_map_large,
        ax=ax2,
        cbar=False,
        save=False,
        return_n_closest=return_n_closest,
        highlight=highlight,
        highlight_colors=highlight_colors,
        compress=False,
        save_dir=None,
        save_name="heatmap",
        fontsize=20,
        debug=False,
    )
    print("Plotting large mapping")
    _, counts2 = plot_som_bmu_heatmap(
        trained_som,
        None,
        data_map=data_map_remnant,
        ax=ax3,
        cbar=False,
        save=False,
        return_n_closest=return_n_closest,
        highlight=highlight,
        highlight_colors=highlight_colors,
        compress=False,
        save_dir=None,
        save_name="heatmap",
        fontsize=16,
        debug=False,
    )
    if annotations_for_paper:
        ax2.text(
            1 / 5 - 0.1,
            1.02,
            "A",
            fontsize=30,
            ha="center",
            transform=ax2.transAxes,
            color="r",
        )
        ax2.text(
            2 / 5 - 0.1,
            1.02,
            "B",
            fontsize=30,
            ha="center",
            transform=ax2.transAxes,
            color="r",
        )
        ax2.text(
            0.9, 1.02, "C", fontsize=30, ha="center", transform=ax2.transAxes, color="r"
        )
        ax2.text(
            1.04,
            4 / 5 - 0.11,
            "D",
            fontsize=30,
            ha="center",
            transform=ax2.transAxes,
            color="r",
        )
    if caption1 is None:
        ax1.set_title(f"[Experiment: {settings1.experiment}]" + title_append)
    else:
        ax1.set_title(caption1)
    if caption2 is None:
        ax2.set_title('All sources >60" \nin HETDEX\nmapped to SOM')
    else:
        ax2.set_title(caption2)
    if caption3 is None:
        ax3.set_title("Visually inspected \nAGN remnants\nmapped to SOM")
    else:
        ax3.set_title(caption3)
    if save:
        if save_path is None:
            # plt.savefig(os.path.join(save_dir,f"Mapping_{settings1.experiment.replace(' ','_').replace(';','')}.pdf"),bbox_inches='tight')
            save_path = os.path.join(
                save_dir,
                f"Mapping_{settings1.run_id}_and_{settings2.run_id}_to_{settings1.experiment.replace(' ','_').replace(';','')}_compressed_{compress}.pdf",
            )
        plt.savefig(save_path, bbox_inches="tight")
        print("Saved two heatmaps to:", save_path)

    plt.show()
    return data_map_large, data_map_remnant, counts1, counts2


def plot_som_and_three_heatmaps(
    settings1,
    settings2,
    settings3,
    output_directory,
    map_dir,
    save=False,
    save_dir=None,
    specific_som=None,
    highlight=[],
    highlight_colors=[],
    highlight_rotatedsize=True,
    legend_list=[],
    title_append="",
    caption1=None,
    annotations_for_paper=False,
    compress=True,
    caption2=None,
    caption3=None,
    caption4=None,
    save_path=None,
):
    """Plot a SOM accompanied by two heatmaps and corresponding captions"""
    # Create som object and add unpacked mapping to catalogue
    if settings2.map_path is None:
        settings2.map_path = os.path.join(
            map_dir,
            f"{settings2.store_filename}_ID{settings2.run_id}_mapped_to_ID{settings2.map_to_run_id}.bin",
        )

    settings2.map_path = settings2.map_path.replace(
        "data2", "home/rafael/data"
    ).replace("data1", "home/rafael/data")
    (
        data_map_remnant,
        numberOfImages_remnant,
        som_width,
        som_height,
        som_depth,
    ) = load_som_mapping(
        settings2.map_path, settings2.som, version=2, verbose=False, compress=compress
    )
    print("Loading remnant SOM mapping from path:", settings2.map_path)

    settings3.map_path = settings3.map_path.replace(
        "data2", "home/rafael/data"
    ).replace("data1", "home/rafael/data")
    (
        data_map_nonremnant,
        numberOfImages_nonremnant,
        som_width,
        som_height,
        som_depth,
    ) = load_som_mapping(
        settings3.map_path, settings3.som, version=2, verbose=False, compress=compress
    )
    print("Loading nonremnant SOM mapping from path:", settings3.map_path)

    if specific_som is None:
        print("Loading large SOM mapping from path:", settings1.map_path)
        settings1.map_path = settings1.map_path.replace(
            "data2", "home/rafael/data"
        ).replace("data1", "home/rafael/data")
        (
            data_map_large,
            numberOfImages_large,
            som_width,
            som_height,
            som_depth,
        ) = load_som_mapping(
            settings1.map_path,
            settings1.som,
            version=2,
            verbose=False,
            compress=compress,
        )
    else:
        print("Loading large SOM mapping from path:", specific_som.map_path)
        (
            data_map_large,
            numberOfImages_large,
            som_width,
            som_height,
            som_depth,
        ) = load_som_mapping(
            specific_som.map_path,
            specific_som.som,
            version=2,
            verbose=False,
            compress=compress,
        )

    print("\nExperiment:", settings1.experiment)
    print("We mapped this many large sources to the SOM:", np.shape(data_map_large))
    print("We mapped this many remnants to the SOM:", np.shape(data_map_remnant))
    print("We mapped this many nonremnants to the SOM:", np.shape(data_map_nonremnant))
    # Create heatmap to show where on the map the remnants end up
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(35, 15))

    if not specific_som is None:
        trained_som = specific_som.som
    else:
        try:
            trained_som = load_SOM(
                "/home/rafael/data/mostertrij/data/lockman/output",
                settings1.map_to_run_id,
            )
        except:
            trained_som = load_SOM(output_directory, settings1.map_to_run_id)
    old_w = trained_som.som_width
    old_h = trained_som.som_height
    print("Normalize som:", settings1.normalize)
    if trained_som.number_of_channels == 1:
        plot_som(
            trained_som,
            gap=2,
            save=False,
            save_name=f"som_ID{settings1.map_to_run_id}",
            save_dir=save_dir,
            ax=ax1,
            normalize=settings1.normalize,
            overwrite=True,
            compress=compress,
            # highlight=highlight,highlight_colors=highlight_colors,
            legend_list=legend_list,
            highlight_rotatedsize=highlight_rotatedsize,
            cmap="viridis",
            align_prototypes=True,
            zoom_in=settings1.zoom_in,
            trained_path=settings1.map_to_binpath,
            version=2,
        )
    else:
        print("Plotting multichannel SOM using contours for the second channel.")
        plot_som_3D_contour(
            trained_som,
            gap=2,
            save=False,
            save_name="",
            save_dir="",
            ax=ax1,
            normalize=settings2.normalize,
            overwrite=True,
            replace_nans=True,
            compress=compress,
            cmap="viridis",
            align_prototypes=True,
            zoom_in=settings1.zoom_in,
            trained_path=settings1.map_to_binpath,
            version=2,
        )

    print("Plotting large mapping")
    trained_som.som_width = som_width
    trained_som.som_height = som_height
    _, counts1 = plot_som_bmu_heatmap(
        trained_som,
        None,
        data_map=data_map_large,
        ax=ax2,
        cbar=False,
        save=False,
        highlight=highlight,
        highlight_colors=highlight_colors,
        compress=False,
        save_dir=None,
        save_name="heatmap",
        fontsize=16,
        debug=False,
    )

    print("Plotting remnant mapping")
    _, counts2 = plot_som_bmu_heatmap(
        trained_som,
        None,
        data_map=data_map_remnant,
        ax=ax3,
        cbar=False,
        save=False,
        highlight=highlight,
        highlight_colors=highlight_colors,
        compress=False,
        save_dir=None,
        save_name="heatmap",
        fontsize=20,
        debug=False,
    )

    print("Plotting nonremnant mapping")
    _, counts3 = plot_som_bmu_heatmap(
        trained_som,
        None,
        data_map=data_map_nonremnant,
        ax=ax4,
        cbar=False,
        save=False,
        highlight=highlight,
        highlight_colors=highlight_colors,
        compress=False,
        save_dir=None,
        save_name="heatmap",
        fontsize=20,
        debug=False,
    )

    if annotations_for_paper:
        ax2.text(
            1 / 5 - 0.1,
            1.02,
            "A",
            fontsize=30,
            ha="center",
            transform=ax2.transAxes,
            color="r",
        )
        ax2.text(
            2 / 5 - 0.1,
            1.02,
            "B",
            fontsize=30,
            ha="center",
            transform=ax2.transAxes,
            color="r",
        )
        ax2.text(
            0.9, 1.02, "C", fontsize=30, ha="center", transform=ax2.transAxes, color="r"
        )
        ax2.text(
            1.04,
            4 / 5 - 0.11,
            "D",
            fontsize=30,
            ha="center",
            transform=ax2.transAxes,
            color="r",
        )
    if caption1 is None:
        ax1.set_title(f"[Experiment: {settings1.experiment}]" + title_append)
    else:
        ax1.set_title(caption1)
    if caption2 is None:
        ax2.set_title("Visually inspected \nAGN remnants\nmapped to SOM")
    else:
        ax2.set_title(caption2)
    if caption3 is None:
        ax3.set_title('All sources >60" \nin HETDEX\nmapped to SOM')
    else:
        ax3.set_title(caption3)
    if caption4 is None:
        ax4.set_title('All sources >60" \nin HETDEX\nmapped to SOM')
    else:
        ax4.set_title(caption4)
    if save:
        if save_path is None:
            # plt.savefig(os.path.join(save_dir,f"Mapping_{settings1.experiment.replace(' ','_').replace(';','')}.pdf"),bbox_inches='tight')
            save_path = os.path.join(
                save_dir,
                f"Mapping_{settings1.run_id}_and_{settings2.run_id}_and_{settings3.run_id}_to_{settings1.experiment.replace(' ','_').replace(';','')}_compressed_{compress}.pdf",
            )
        plt.savefig(save_path, bbox_inches="tight")
        print("Saved three heatmaps to:", save_path)

    trained_som.som_width = old_w
    trained_som.som_height = old_h
    plt.show()
    return (
        data_map_large,
        data_map_remnant,
        data_map_nonremnant,
        counts1,
        counts2,
        counts3,
    )


def get_heatmap_distribution_differences(
    counts1,
    counts2,
    mask_insignificant=True,
    mask_negative=True,
    save=False,
    save_dir=None,
    save_path=None,
    save_label=None,
    debug=False,
    title=None,
):
    """Get heatmap distribution differences"""

    # Get predicted distribution if counts1 would follow the distribution of counts2
    total_counts1 = np.sum(counts1)
    total_counts2 = np.sum(counts2)
    std_dev_counts2 = np.sqrt(counts2)
    predicted_counts1 = counts2 * (total_counts1 / total_counts2)
    predicted_std = std_dev_counts2 * (total_counts1 / total_counts2)

    # Get difference between predicted and actual counts
    difference = counts1 - predicted_counts1
    relative_difference = counts1 / predicted_counts1
    # Check if difference is significant
    non_significant = np.abs(difference) < np.abs(predicted_std)
    if debug:
        print("totalcoutns", total_counts1, total_counts2)
        print("predocted_counts1", predicted_counts1.T)
        print("difference", difference.T)
        print("non_significant", non_significant.T)
        plt.figure(figsize=(10, 10))
        plt.title("predicted_counts1")
        sns.heatmap(
            predicted_counts1.T,
            annot=False,
            square=True,
            fmt="d",
            cbar=True,
            cmap="viridis",
        )
        plt.show()
        plt.figure(figsize=(10, 10))
        plt.title("predicted_std")
        sns.heatmap(
            predicted_std.T,
            annot=False,
            square=True,
            fmt="d",
            cbar=True,
            cmap="viridis",
        )
        plt.show()
        plt.figure(figsize=(10, 10))
        plt.title("non_significant")
        sns.heatmap(
            non_significant.T,
            annot=False,
            square=True,
            fmt="d",
            cbar=True,
            cmap="viridis",
        )
        plt.show()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    if title is None:
        plt.title("Difference between predicted and actual counts")
    else:
        plt.title(title)
    if mask_insignificant:
        mask_non_significant = non_significant.T
    else:
        mask_non_significant = np.ones_like(non_significant) < 0
    if mask_negative:
        mask_negative = difference.T < 0
    else:
        mask_negative = np.ones_like(difference) < 0
    sns.heatmap(
        relative_difference.T,
        annot=True,
        square=True,
        mask=mask_non_significant | mask_negative,
        cbar=False,
        cmap="viridis",
    )
    if save:
        if save_path is None:
            plt.savefig(
                os.path.join(
                    save_dir,
                    f"Mapping_{save_label.replace(' ','_').replace(';','')}_relative_abundance.pdf",
                ),
                bbox_inches="tight",
            )
        else:
            plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def create_new_files_after_filtering(
    accepted_ids,
    old_cutouts,
    old_cat,
    new_settings,
    data_directory,
    output_directory,
    gpu_id=0,
    map_dir="/home/rafael/data/mostertrij/data/remnants/output/maps",
):
    # Creat new paths
    new_bin_path = os.path.join(data_directory, new_settings.store_filename + ".bin")
    new_cat_path = os.path.join(
        output_directory, "catalogue_" + new_settings.store_filename + ".h5"
    )
    new_cutouts_path = cutouts_path = os.path.join(
        output_directory, new_settings.store_filename + ".npy"
    )
    new_map_path = os.path.join(
        map_dir,
        f"{new_settings.store_filename}_ID{new_settings.run_id}_mapped_to_ID{new_settings.map_to_run_id}.bin",
    )
    new_settings.map_path = new_map_path
    new_settings.save(output_directory)
    # Store new filtered files
    new_cat = old_cat.iloc[accepted_ids]
    new_cat.to_hdf(new_cat_path, "df")
    new_cutouts = old_cutouts[accepted_ids]
    np.save(new_cutouts_path, new_cutouts)
    write_numpy_to_binary_v2(
        new_bin_path,
        new_cutouts,
        new_settings.som.layout,
        verbose=False,
        overwrite=True,
    )
    # Output code to map new cutouts to trained SOM
    mapstring = map_dataset_to_trained_som(
        new_settings.som,
        new_bin_path.replace("data2", "home/rafael/data"),
        new_map_path.replace("data2", "home/rafael/data"),
        new_settings.map_to_binpath.replace("data2", "home/rafael/data"),
        gpu_id,
        use_gpu=True,
        verbose=True,
        version=2,
        alternate_neuron_dimension=None,
        use_cuda_visible_devices=True,
        rotation_path=None,
        circular_shape=False,
    )
    return (
        new_settings,
        new_cat,
        new_cat_path,
        new_cutouts,
        new_cutouts_path,
        new_bin_path,
    )


def map_to_som_using_settingsfile(
    settingsfile,
    data_directory,
    output_directory,
    gpu_id=0,
    map_dir="/home/rafael/data/mostertrij/data/remnants/output/maps",
):
    # Creat new paths
    bin_path = os.path.join(data_directory, settingsfile.store_filename + ".bin")
    map_path = os.path.join(
        map_dir,
        f"{settingsfile.store_filename}_ID{settingsfile.run_id}_mapped_to_ID{settingsfile.map_to_run_id}.bin",
    )
    settingsfile.map_path = map_path
    settingsfile.save(output_directory)

    # Output code to map new cutouts to trained SOM
    mapstring = map_dataset_to_trained_som(
        settingsfile.som,
        bin_path.replace("data2", "home/rafael/data"),
        map_path.replace("data2", "home/rafael/data"),
        settingsfile.map_to_binpath.replace("data2", "home/rafael/data"),
        gpu_id,
        use_gpu=True,
        verbose=True,
        version=2,
        alternate_neuron_dimension=None,
        use_cuda_visible_devices=True,
        rotation_path=None,
        circular_shape=False,
    )
    return settingsfile
