
# Transcription 3D Folding

A project to investigate the impact of 3D genome folding on transcriptional regulation. The project includes modules for processing a variety of genomic and transcriptional data-types, plus modules for generating visualizations and analyses.


## Credits

This work was done under Geoff Fudenberg's guidance, as research for the Fudenberg Lab at USC, in collaboration with Nora Lab at UCSF.

Continuing updates to this project can be found at [Fudenberg Research Group's Github](https://github.com/Fudenberg-Research-Group/transcription_3Dfolding/)

Unpublished RNA-sequencing data was provided by collaborators at UCSF's Nora Lab. 

## Methods

Most of this work is focused on the role of the cohesin-associated protein NipBL. We use an RNA-sequencing dataset from a degraded NipBL cell line to identify up-regulated and down-regulated genes, using `DESeq2`. From there, we look for patterns that separate these differentially-expressed genes in relation to other genomic features -- including overall genomic folding structure and TAD domains (from Hi-C data) and in conjunction with other genome-associating proteins.

All of this work can be found through vizualizations in the project file notebooks.

## Development

### Environment

For easy installation of scientific packages, we recommend building the environment using [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).

With conda installed, build environment by running:

`conda env create --file environment.yml`


### Project Structure

Note: this project used [a python project template](https://github.com/rochacbruno/python-project-template/generate), but was not developed intended for distribution. As such, some of the template files are left either empty or unchanged from the original template (denoted below).

Overall Structure of files and locations.


```text
├── docs                     # [empty] Documentation site (add more .md files here)
│   └── index.md             # [empty] The index page for the docs site
├── .github                  # Github metadata for repository
│   ├── release_message.sh   # A script to generate a release message
│   └── workflows            # The CI pipeline for Github Actions
├── .gitignore               # A list of files to ignore when pushing to Github
├── HISTORY.md               # Auto generated list of changes to the project
├── LICENSE                  # [empty] The license for the project
├── Makefile                 # [empty] A collection of utilities to manage the project
├── mkdocs.yml               # Configuration for documentation site
├── transcription_3dfolding             # The main python package for the project
│   ├── feature_generation   # Notebooks for vizualizing genomic features generated from Hi-C boundary peak-calling
│   ├── trancription_data_analysis
│   │   ├── chip_seq_agg     # Notebooks to vizualize for Chip-Seq signals around disregulated genes
│   │   ├── chip_seq_peak    # Notebooks to vizualize Chip-Seq peaks in relation to disregulated genes
│   │   ├── feature_histograms # notebooks with histograms for exploratory data analysis of features
│   │   └── utils            # Functions for processing data or generating plots
│   ├── base.py              # [empty] The base module for the project
│   ├── __init__.py          # [empty] This tells Python that this is a package
│   ├── __main__.py          # [empty] The entry point for the project
│   └── VERSION              # [empty] The version for the project is kept in a static file
├── README.md                # The main readme for the project
├── environment.yml          # The full file for generating a conda environment
└── setup.py                 # [empty] The setup.py file for installing and packaging the project
```



