import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

# Setting plot parameters
sns.set_style('ticks')
matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['image.cmap'] = 'Spectral_r'
warnings.filterwarnings(action="ignore", module="matplotlib", message="findfont")


def plot_reg(reg_dict, genes, adata, expr_df, 
             plot_span=False, gene_spans=None, ncols=3, 
             normalized=False, pseudo_axis_key=None, 
             color_cells=False, color_by=None, cpalette=None, 
             save=False, outfile=None):
    """
    Plots the regression line for the given genes
    For a normalized regression plot, individual cells will not be plotted
    If not normalized, can plot and color cells by key in adata.obs
    
    Parameters
    ----------
    reg_dict: dict
        gene:DataFrame containing predicted expression values against pseudo-axis
    genes: list 
        gene names contained in ``reg_dict``
    expr_df: pandas.DataFrame
        cell by gene expression values
    plot_span: bool
        Whether to plot the gene span 
    gene_spans: pandas.DataFrame
        Columns with gene span, threshold and derivative points
    
    # following are only needed if 'normalized=False'
    normalized: bool
        Whether plotting a normalized regression line
    adata: AnnData
        Cells by genes
    pseudo_axis_key: String
        Key in adata.obs specifying the pseudo-axis values
    color_cells: bool
        Whether to color the cells by an ``adata.obs`` annotation
    color_by: string
        Key in adata.obs by which to color cells
    cpalette_key: string
        If ``color_by`` is categorical, the key in ``adata.uns`` for color palette
    save: bool, default: False
        Whether to save the outputted plot
    outfile: String, default: None
        If ``save``, path to outfile, ending in .png    
    
    """
    
    # set up figure
    nrows = int(np.ceil(len(genes) / ncols))

    fig, axes = plt.subplots(nrows, ncols, sharey=True, sharex=True, 
                             figsize=(3*ncols, 3*nrows), squeeze=False)
        
    for ax, g in zip(axes.flat, genes): 
    
        if not normalized:
            if color_cells:
                if adata.obs[color_by].dtype.name == 'category':
                    cpalette = adata.uns[cpalette_key]
                else:
                    cpalette = 'Spectral_r'
                sns.scatterplot(data=adata.obs, x=adata.obs[pseudo_axis_key], y=expr_df[g], hue=color_by, palette=cpalette, alpha=0.75, ax=ax)
            else:
                sns.scatterplot(x=adata.obs[pseudo_axis_key], y=expr_df[g], color='gainsboro', alpha=0.75, ax=ax)
            
        if plot_span:
            _single_span_plot(g, gene_spans, ax)
        
        _single_reg_plot(g,reg_dict,ax)
        
    if save:
        fig.savefig(outfile, bbox_inches='tight', dpi=200)

    
def plot_span(genes, span_df, reg_dict, 
              inflect=False, thresh=False, 
              save=False, outfile=None):

    """
    Plots the gene span behind the regression line for given gene(s)
    
    Parameters
    ----------
    genes: list, array-like
        Genes to plot span for
    span_df: pandas.DataFrame
        gene by columns: ['span', 'first_deriv', 'sec_deriv', 'thresh']
    reg_dict: dict
        gene:DataFrame containing predicted expression values against pseudo-axis
    inflect: bool, default: False
        Whether to plot the inflection points of the first and second derivative
        Salmon: first derivative
        Teal  : second derivative
    thresh: bool, default: False
        Whether to plot the threshold line
    save: bool, default: False
        Whether to save the outputted plot
    outfile: String, default: None
        If ``save``, path to outfile, ending in .png
    """
    # set up figure
    nrows = int(np.ceil(len(genes) / ncols))

    fig, axes = plt.subplots(nrows, ncols, sharey=True, sharex=True, 
                             figsize=(3*ncols, 3*nrows), squeeze=False)
        
    for ax, g in zip(axes.flat, genes): 
        _single_span_plot(g, span_df, ax, inflect, thresh)

        _single_reg_plot(g, reg_dict, ax, interval=False)
        
    if save:
        fig.savefig(outfile, bbox_inches='tight', dpi=200)

def _single_reg_plot(gene,reg_dict, ax, interval=True):
    
    sns.lineplot(x=reg_dict[g]['pseudo_axis'], y=reg_dict[g]['expression'], 
                     color='darkslategray', lw=1, ax=ax)
        
    # shade the area within one standard deviation of the trend line
    if interval:
        ax.fill_between(reg_dict[g]['pseudo_axis'], reg_dict[g]['low_b'], reg_dict[g]['up_b'],
                        alpha=0.3, color='cadetblue')

    # plot annotations
    ax.set_title(g)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    sns.despine()
    
    
def _single_span_plot(g, span_df, ax, inflect, thresh):
    
    # Draw gene span
    ax.axvspan(span_df.loc[g, 'span'][0], span_df.loc[g, 'span'][1], alpha=0.3, color='darkseagreen', label='gene span')
    
    # plot inflection points
    if inflect:
        for k, boundary in enumerate(span_df.loc[gene, 'first_deriv'], 1):
            ax.axvline(x=boundary, color='salmon', label='first deriv')
        
        for k, boundary in enumerate(span_df.loc[gene, 'sec_deriv'], 1):
            ax.axvline(x=boundary, color='teal', label='sec deriv')
    
    # plot threshold used for span
    if thresh:
        ax.axhline(y=span_df.loc[gene, 'threshold'], label='threshold', ls='--', color='gray')       
    
    sns.despine()
    
        

