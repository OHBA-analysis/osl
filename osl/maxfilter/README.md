# OHBA Maxfilter

A python batch processing script for Maxfilter preprocessing of MEG files.

- [Usage](#usage)
  - [Customising Options](#customising-options)
  - [Multistage Options](#multistage-options)
- [Options Arguments](#optional-arguments)


## Usage

We can call `osl_maxfilter` via the command line (make sure you have activated the osl conda environment first). `osl_maxfilter` requires at least 2 positional inputs to run. These are

```
  files                 plain text file containing full paths to files to be
                        processed
  outdir                Path to output directory to save data in
```

For example:

```
osl_maxfilter input_files.txt /path/to/my/output/dir/
```

will run each fif file in `input_files.txt` through maxfilter with default options and store the outputs in `/path/to/my/output/dir/`

### Customising Options

Maxfilter processing can be customised using command line flags which are (mostly) mapped to the options in Maxfilter itself. For exmaple, we can specify that `autobad` be included by adding the `--autobad` command line flag.

```
osl_maxfilter input_files.txt /path/to/my/output/dir/ --autobad
```

Similarly, we can include movement compensation and head-position computation by adding their respective options.

```
osl_maxfilter input_files.txt /path/to/my/output/dir/ --autobad --movecomp --headpos
```

Some options take additional arguments. Here we specify that temporal extension SSS should be applied with a 20 second data buffer (using `--tsss` and `-st 20`) and that two specific channels should be removed from the analysis (`--bads 1722 1723`).

```
osl_maxfilter input_files.txt /path/to/my/output/dir/ --movecomp --headpos --tsss --st 20 --bads 1722 1723
```

A [complete list of customisation options](#optional-arguments) is included at the bottom of this page.

#### Temporal Extension

The temporal extension can be turned on with the `--tsss` flag, the buffer length and correlation threshold can then be cusomised using the `--st` and `--corr` options. This is slightly different to main maxfilter which only requires you to specify -st to turn on the temporal extension.

This example specifys a temporal extension with a twenty second buffer window and a correlation threshold of 0.9

```
osl_maxfilter input_files.txt /path/to/my/output/dir/ --movecomp --headpos --tsss --st 20 --corr 0.9
```

#### Position Translation

There are several ways to customised head position translation to align head position between two recordings. One option is to align both scans to the same pre-specified position. This is done by specifying `--trans` to default and providing a head origin co-ordinate. For example:

```
osl_maxfilter input_files.txt /path/to/my/output/dir/ --trans default --origin 0 0 40 --frame head --force
```

Will move the point 0,0,40 in head space to the device origin and then align the device and head coordinate systems.

We can also align one scan to match the head position of a reference scan. This is done by specifying the path to a reference fif file in the `--trans` option. For example:

```
osl_maxfilter input_files.txt /path/to/my/output/dir/ --trans /path/to/reference.fif
```

Will align all files with the head position from the `/path/to/reference.fif` file.


### Multistage Options

More complex maxfilter workflows are implemented as specific 'modes'. Two modes are implemented at the moment.

#### Multistage

The multistage maxfilter is selected using `--mode multistage`. This will first run:

1) Maxfilter with limited customisation, no movement compensation and autobad on to identify bad channels. 
2) Maxfilter with full customisation and movement compensation with the specific bad channels from stage 1
3) Optional [position translation](#position-translation) (this requires that the `--trans` options are specified)

#### CBU

The CBU maxfilter processing chain is selected using `--mode cbu`. This will first run:

1) A custom head-origin co-ordinate is estimated from the headshape points with any nose points removed.
2) Maxfilter with limited customisation, no movement compensation and autobad on to identify bad channels. 
3) Maxfilter with full customisation and movement compensation with the specific bad channels from stage 1
4) Position translation to a default head position.

### Optional Arguments

```
optional arguments:
  -h, --help            show this help message and exit
  --maxpath MAXPATH     Path to maxfilter command to use
  --mode MODE           Running mode for maxfilter. Either 'standard' or
                        'multistage'
  --headpos             Output additional head movement parameter file
  --movecomp            Apply movement compensation
  --movecompinter       Apply movement compensation on data with intermittent
                        HPI
  --autobad             Apply automatic bad channel detection
  --autobad_dur AUTOBAD_DUR
                        Set autobad on with a specific duration
  --bad BAD [BAD ...]   Set specific channels to bad
  --badlimit BADLIMIT   Set upper limit for number of bad channels to be
                        removed
  --trans TRANS         Transforms the data to the head position in defined
                        file
  --origin ORIGIN [ORIGIN ...]
                        Set specific sphere origin
  --frame FRAME         Set device/dead co-ordinate frame
  --force               Ignore program warnings
  --tsss                Apply temporal extension of maxfilter
  --st ST               Data buffer length for TSSS processing
  --corr CORR           Subspace correlation limit for TSSS processing
  --inorder INORDER     Set the order of the inside expansion
  --outorder OUTORDER   Set the order of the outside expansion
  --hpie HPIE           sets the error limit for hpi coil fitting (def 5 mm)
  --hpig HPIG           ets the g-value limit (goodness-of-fit) for hpi coil
                        fitting (def 0.98))
  --scanner SCANNER     Set CTC and Cal for the OHBA scanner the dataset was
                        collected with (VectorView, VectorView2 or Neo). This
                        overrides the --ctc and --cal options.
  --ctc CTC             Specify cross-talk calibration file
  --cal CAL             Specify fine-calibration file
  --overwrite           Overwrite previous output files if they're in the way
  --dryrun              Don't actually run anything, just print commands that
                        would have been run
```
