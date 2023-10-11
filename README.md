# MEDIC paper analysis scripts
Analysis scripts for the medic paper.

There are several hard coded paths that may prevent you from running some scripts from this repo immediately. I will make an attempt to fix these over time. If you find something not working, feel free to create an issue!

## Usage

To use this repo, first install it:

```
pip install -e ./ -v --config-settings editable_mode=strict
```

this will give you several scripts to run on your `PATH`.

### Python scripts

Once you've installed this repo, you can run any of the scripts in the `medic_analysis/scripts` folder.

To call them, simply type the name of the script without the `.py` extension.

The two main scripts used in the paper are:

`alignment_metrics.py`: Generates alignment metrics for figures 5 and 6 in paper.

`paper_figures.py`: Generates actual paper figures + supplemental figures.

Note that this repo expects outputs from our [lab processing pipeline](https://github.com/DosenbachGreene/processing_pipeline).

### MATLAB scripts

For scripts written in MATLAB, you will need to navigate to the `matlab_scripts` directory and run them from there.

There are two main scripts and auxiliary python script that goes along these scripts:

`explore_MEDIC.m`: This will run the ABCD group average comparison found in figure 3 of the paper.

`split_groups.py`: This python script will split the ABCDavg dscalar into MEDIC and TOPUP groups. You must run `explore_MEDIC.m` beforehand.

`cluster_correct.m`: This will run the multiple comparisons cluster correction for the t-statistic map. You should have run `split_groups.py` beforehand.
