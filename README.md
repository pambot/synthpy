# Synthpy
Synthetic data generation is required for rigourous algorithm development, especially algorithms of messy fields where they cannot be mathematically proved to be correct. High quality synthetic data needs to not only resemble features of real world data, incorporating a signifcant amount of domain knowledge, but it also needs to be statistically tunable so as to allow the researcher to deduce which parameters affect whichever component of their algorithm. Currently, synthetic data generation scripts are made independently by each research group and are often inextricably tied to their particular research project. Many of them rely on MATLAB, which is not open source. These factors limit the reproducibility and usability of synthetic data. We are creating Synthpy, an open source module for the generation of synthetic data built on the Python scientific computing stack. Synthpy is beginning with the generation of biological imaging and sequencing data, but it could also potentially span multiple fields in the future.

For an overview of the `Image` module, see the [tutorial](https://github.com/pambot/synthpy/blob/master/scipy2017.ipynb), which is also the beginnings of a lightning talk for Scipy 2017.

## Future Plans
* Docstrings have begun! Finish all docstrings for `Image` and generate Sphinx docs.
* Re-think the FASTA representation in `Sequence` to accomodate structural variations.
* Make `Sequence` example using real biological data.
* And more! So much to do...
