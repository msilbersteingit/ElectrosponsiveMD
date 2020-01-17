from analib import extract
from analib import compute
from analib import plotting_functions
import pandas as pd

def rdf(coordinates, box_l, simname, coords_loc=[3,6], nhis = 200,
    save=False,style='matplotlib'):
    """ Radial distribution function"""
    g,r = compute.radial_distribution_function(coordinates, box_l, simname, 
        coords_loc, nhis)
    plotting_functions.radial_distribution_function_plotting(g,r,simname,save,style)

def sk(coordinates, rdf_fname,box_l,simname,coords_loc=[3,6],
    save=False,style='matplotlib'):
    " Structure factor from rdf"
    k,sk = compute.structure_factor(coordinates, rdf_fname,box_l,simname,coords_loc)
    plotting_functions.structure_factor_plotting(k,sk,simname,save,style)

def standard_analysis(simname, nfiles):
    """
    Extract variables from multiple log files and save them in one single 
    pandas dataframe. 

    Args:
    simname (string): Name of the simulation (end name till density (uk)).
    Do not add simulation number (eg: uk_1)
    nfiles (int): Number of simulations.
    save (bool): Whether to save the plots in a directory.
    style (string): style of the plots e.g.: 'seaborn','fivethirtyeight',etc.

    Returns:
    aggregate_data_pd: nicely organized one single pandas dataframe containing data from ALL
    the simulations (depending on nfiles) in 12 columns.
    """
    aggregate_data=pd.DataFrame()
    for sims in range(1,nfiles+1):
        curr_fname=simname+'_'+str(sims)
        print(curr_fname)
        aggregate_data[sims] = [extract.extract_log_thermo(curr_fname)]
    aggregate_data_list=[aggregate_data[1][0][key] for key in aggregate_data[1][0]]

    for sims in range(2,nfiles+1):
        for key in aggregate_data[sims][0]:
            aggregate_data_list.append(aggregate_data[sims][0][key])
    aggregate_data_pd = pd.concat(aggregate_data_list)
    return aggregate_data_pd



