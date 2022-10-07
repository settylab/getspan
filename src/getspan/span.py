import pandas as pd
import numpy as np

import scanpy as sc
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from scipy.sparse import issparse
from tqdm.auto import tqdm

import pickle


def calc_reg(adata, genes,  pseudo_axis_key, 
             imputed=True, scaled=False, res=200, std=True,
             smooth=False, length_scale=0.2, ls_bounds=(1e-2,1e2),
             constant_bounds='fixed', noise_level=(1e-5, 1e5), 
             save=True, pickle_file='trends.pickle', atac = False ):
    
    """
    Function to calulate a Gaussian Process Regression for given list of genes
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix with pseudo axis and imputed gene expression
    genes: List, array-like
        genes to compute regression for. Must be in ``adata.var_names``
    pseudo_axis_key: String
        Key in ``adata.obs`` annotating each cell's pseudo-axis value
    imputed: bool, default: True
        Whether to use imputed gene expression
    scaled: bool, default: False
        When ``imputed=True``, indicates whether to use scaled data
    res: int, default: 200
        Resolution of Gaussian Process Regression
    std: bool, default: True
        Whether to calculate standard deviate and confidence interval
    smooth: bool, default: False
        Whether to compute a smoother regression line at a fixed length scale
    length_scale: float
        If ``smooth``, fixed length_scale for RBF kernel, determines length of deviation from data
    ls_bounds: pair of floats, default: (1e-2, 1e2)
        The lower and upper bound on ``length_scale`` for the RBF kernel when ``smooth=False``
    constant_bounds: pair of floats or "fixed", default="fixed"
        The lower and upper bound on ``constant_value`` for the Constant kernel. If set to "fixed", remains unchanged during tuning
    noise_level: pair of floats or "fixed", default: (1e-5, 1e5)
        The lower and upper bound on ``noise_level``. If set to "fixed", remains unchanged during tuning
    save: bool, default: True
        Whether to save the resulting dictionary to a pickle file after each GPR calculation. Recommended for long lists of genes 
    pickle_file: String, default: trends.pickle
        If ``save``, file path ending in .pickle to write to
    atac: bool, default: False
        indicates if the input data is scATAC-seq gene scores instead of scRNA-seq gene expression and annotates dataframe columns with correct labels.
        No option for using scaled data at this time.
    Returns
    -------
    results: dict
        Dictionary of DataFrames containing predicted regression and confidence interval
    """
    
    results = {}
    
    # Retrieve the imputed gene expression
    if atac:
        if imputed:
            expr_df = pd.DataFrame(adata.obsm['MAGIC_imputed_GeneScores'], columns = adata.uns['GeneScoresColumns'], index = adata.obs_names)
        else:
            expr_df = pd.DataFrame(adata.obsm['GeneScores'], columns = adata.uns['GeneScoresColumns'], index = adata.obs_names)
    elif imputed:
    if imputed:
        if scaled:
            expr_df = pd.DataFrame(adata.obsm['scaled_MAGIC_imputed_data'], columns=adata.var_names, index=adata.obs_names)
        else:
            expr_df = pd.DataFrame(adata.obsm['MAGIC_imputed_data'], columns=adata.var_names, index=adata.obs_names)
    else:
        if issparse(adata.X):
            expr_df = pd.DataFrame(adata.X.A, columns=adata.var_names, index=adata.obs_names)
        else:
            expr_df = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
    # Initialize kernel specifying covariance function for the GaussianProcessRegressor
    
    if smooth:
        kernel = C(1.0, constant_value_bounds=constant_bounds) * RBF(length_scale, 'fixed') + WhiteKernel(noise_level_bounds=noise_level)
    else:
        kernel = C(1.0, constant_value_bounds=constant_bounds) * RBF(1, ls_bounds) + WhiteKernel(noise_level_bounds=noise_level) 
    
    
    # retrieve the pseudo axis from adata
    ps_ax = adata.obs[pseudo_axis_key].values
    ps_ax = ps_ax.reshape((ps_ax.shape[0],1)) # reshape into 2D array
    
    # initialize values for predicted axis
    pred_ax = np.linspace(ps_ax.min(), ps_ax.max(), res)
    pred_ax_2d = pred_ax.reshape(pred_ax.shape[0], 1) # reshape into 2D array
     
    
    # Calculate GPR for each gene
    for gene in tqdm(genes, total=len(genes)):
        gp = GaussianProcessRegressor(kernel=kernel)
        expr = expr_df[gene]
       
        gp.fit(ps_ax, expr)
        
        # predict expression values 
        if std:
            predictions, stds = gp.predict(pred_ax_2d, return_std=True)
            df = pd.DataFrame({'pseudo_axis': pred_ax,
                               'expression': predictions,
                               'std': stds,
                               'low_b': predictions - stds, 
                               'up_b': predictions + stds})
        else:
            predictions = gp.predict(pred_ax_2d, return_std=False)
            df = pd.DataFrame({'pseudo_axis': pred_ax,
                               'expression': predictions})
    
        results[gene] = df
        
        if save:
            with open(pickle_file, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return results

def calc_span(gexpr_dict, thresh=0.2):
    
    """
    Identifies the span of expression along the pseudo-axis for each gene. 
    Recommened to calculate span based on smoother gene regression trend
    
    Parameters
    ----------
    gexpr_dict: dict
        Contains DataFrames with predicted expression. Recommended: smoother regression 
      
    thresh: float, default: 0.2
        Value from 0.0 - 1.0 specifying preliminary threshold to consider a gene expressed
    
    Returns
    -------
    gene_span: pandas.DataFrame
        Columns with gene span, threshold, and derivative points
    
    """

    norm_gexpr = normalize_regs(gexpr_dict)
    
    gene_span = pd.DataFrame(index=norm_gexpr.keys(), columns=['span', 'first_deriv', 'sec_deriv', 'threshold'])
    
    
    for gene in norm_gexpr:

        gdf = norm_gexpr[gene]
        gene_span.loc[gene, 'threshold'] = thresh

        step_size = (gdf['pseudo_axis'][1] - gdf['pseudo_axis'][0]).round(6)        
        
        valid_locs = gdf.loc[gdf['expression'] >= thresh]['pseudo_axis'].values

        # first derivative inflection points:
        first_deriv = np.gradient(gdf['expression'], step_size)   
        fd_zero = np.where(np.diff(np.sign(first_deriv)))[0]       
        fd_zero_p = gdf.iloc[fd_zero]['pseudo_axis'].values
        
        gene_span.loc[gene, 'first_deriv'] = fd_zero_p


        # second derivative inflection points:
        sec_deriv = np.gradient(first_deriv, step_size)
        sd_zero = np.where(np.diff(np.sign(sec_deriv)))[0] 
        sd_zero_p = gdf.iloc[sd_zero]['pseudo_axis'].values
        
        gene_span.loc[gene, 'sec_deriv'] = sd_zero_p

        
        # Look for a non-continuous range:
        break_points = list(np.where(np.diff(valid_locs).round(6) > step_size)[0])
        
        bps=[]
        for bp in break_points:
            bps.append(valid_locs[bp])
        
        # choose the span with the highest average expression
        if len(break_points) > 0:
            ranges = [valid_locs[0]]            
            
            for i in range(len(break_points)):
                range_break = break_points[i]

                ranges.append(valid_locs[range_break])
                ranges.append(valid_locs[range_break + 1])

            ranges.append(valid_locs[-1])
            lb = 0
            hb = 1
            
            mean = gdf.loc[gdf['pseudo_axis'].apply(lambda x: np.logical_and(x >= ranges[0], x <= ranges[1]))]['expression'].mean()
            #dist = ranges[1] - ranges[0]
            
            for i in range(2, len(ranges), 2):
                new_mean = gdf.loc[gdf['pseudo_axis'].apply(lambda x: np.logical_and(x >= ranges[i], x <= ranges[i+1]))]['expression'].mean()    
                if new_mean > mean:
                    mean = new_mean
                    lb = i
                    hb = i+1
            ranges = [ranges[lb], ranges[hb]]
        else:
            # just take the only span that exists
            ranges = [valid_locs[0], valid_locs[-1]]
        
        # Use first derivative as span boundaries
        fd_zero_p = np.sort(fd_zero_p)
        fdp_l = fd_zero_p[fd_zero_p < ranges[0]]
        fdp_h = fd_zero_p[fd_zero_p > ranges[1]]
    
        
        # Use sec deriv to reduce span:
        sd_zero_p = np.sort(sd_zero_p)
        sdp_l = sd_zero_p[sd_zero_p < ranges[0]]
        sdp_h = sd_zero_p[sd_zero_p > ranges[1]]   
        
        
        if len(fdp_l) > 0:
            low_bound = fdp_l[-1]
            if len(sdp_l) > 0:
                if sdp_l[-1] > low_bound:
                    low_bound = sdp_l[-1]
                
            ranges[0] = low_bound 


        if len(fdp_h) > 0:
            high_bound = fdp_h[0]  
            if len(sdp_h) > 0:
                if sdp_h[0] < high_bound:
                    high_bound = sdp_h[0]
            
            ranges[1] = high_bound
        
        # store span and return
        gene_span.loc[gene,'span'] = ranges
    
    return gene_span
    
def normalize_regs(reg_dict,pseudo_axis_key='pseudo_axis'):
    """
    Performs max-min normalization of gene trend
    
    Parameters
    ----------
    reg_dict: dict
        Contains DataFrames with predicted gene expression along pseudo axis
    
    Returns
    -------
    norm_gexpr: dict
        normalized expression of ``reg_dict``
    """
    
    norm_reg_dict = {}
    
    for gene in reg_dict:
        
        df = pd.DataFrame(columns=[pseudo_axis_key, 'expression'])
        df[pseudo_axis_key] = reg_dict[gene][pseudo_axis_key]

        max_expr = reg_dict[gene]['expression'].max()
        min_expr = reg_dict[gene]['expression'].min()
        norm_expr = (reg_dict[gene]['expression'] - min_expr) / (max_expr - min_expr)

        df['expression'] = norm_expr

        norm_reg_dict[gene] = df
    
    return norm_reg_dict

