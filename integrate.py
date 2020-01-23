from analib import extract
from analib import compute
from analib import plotting_functions
import pandas as pd
import hickle as hkl

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

def chain_orientation_parameter_combined(simname, nfiles, nc, dp):
    """
    Compute chain orientation parameter from multiple unwrapped files 
    (different parts of the same simulation) and combine them in one
    array.
        
    Args:
    simname (string): Name of the simulation (type the name of the simulation
    only till density which is mostly written as 'uk'). Do not add simulation
    number (e.g. uk_1).
    nfiles (int): Number of simulation parts for this specific simulation

    Returns:
    aggregate_cop_array: One array containing the evolution of chain
    orientation parameter from initial topology file to the last
    equilibration step.
    """
    cop_x = []
    cop_y = []
    cop_z = []
    for sims in range(1, nfiles+1):
        curr_fname = simname + '_' + str(sims)
        coord = extract.extract_unwrapped(curr_fname)
        for key in coord:
            cop_x.append(compute.chain_orientation_parameter(coord[key],
                [1,0,0],nc,dp))
            cop_y.append(compute.chain_orientation_parameter(coord[key],
                [0,1,0],nc,dp))
            cop_z.append(compute.chain_orientation_parameter(coord[key],
                [0,0,1],nc,dp))
    dumpnamex = simname + '_cop_x'
    dumpnamey = simname + '_cop_y'
    dumpnamez = simname + '_cop_z'
    print(cop_x)
    print(cop_y)
    print(cop_z)

    hkl.dump(cop_x, dumpnamex)
    hkl.dump(cop_y, dumpnamey)
    hkl.dump(cop_z, dumpnamez)
    return zip(cop_x, cop_y, cop_z)