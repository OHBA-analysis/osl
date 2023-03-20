"""Example script for maxfiltering raw data recorded at Oxford.

Note: this script needs to be run on a computer with a MaxFilter license.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl.maxfilter import run_maxfilter_batch

# Setup paths to raw (pre-maxfiltered) fif files
input_files = [
    "/ohba/pi/knobre/dgresch/Internal_External/data/meg/00_raw/s01/InEx_s01_block_01.fif",
    "/ohba/pi/knobre/dgresch/Internal_External/data/meg/00_raw/s01/InEx_s01_block_02.fif",
]

# Directory to save the maxfiltered data to
output_directory = "/ohba/pi/knobre/cgohil/dg_int_ext/maxfilter"

# Run MaxFiltering
run_maxfilter_batch(
    input_files,
    output_directory,
    "--maxpath /neuro/bin/util/maxfilter --scanner Neo --tsss --mode multistage --headpos --movecomp",
)
