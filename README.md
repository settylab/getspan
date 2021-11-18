# getspan

This package will allow you to compute a gene's span of expression along a pre-computed pseudo-axis (ie. pseudotime, pseudospace). getspan is designed to work with single cell RNA-seq data.

It is recommended to define a specific gene set of interest to calculate spans and trends for, as computing trends and spans for all genes in a dataset will take a significant amount of time.


## Installation

getspan is implemented in Python3. It can be installed with `pip` via one of the following methods, dependiing on how your credentials are setup:

1. With a SSH key (recommended):

```
pip install git+ssh://git@github.com/settylab/getspan.git
```
2. With just your username and password:

```
pip install git+https://github.com/settylab/getspan.git
```
You will be prompted for your username and password.


3. With a personal authenticator token:

*See [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) for details*

Here the token is saved in an environmental variable `GITHUB_TOKEN`

```
pip install git+https://${GITHUB_TOKEN}@github.com/settylab/getspan.git
```

4. Clone the repo and install:
```
pip install git+file:///path/to/your/git/repo
```

5. **To uninstall:**

```
pip uninstall getspan
```

## Dependencies:

Dependencies that the package relies on are found in `setup.py`


## Usage

A tutorial on `getspan` usage and demonstration of visualizations can be found in this notebook:

https://github.com/settylab/getspan/blob/main/notebooks/tutorial.ipynb

Data used in the notebook can be found in `/data/`



