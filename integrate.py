from analib import extract
from analib import compute
from analib import plotting_functions
import pandas as pd
import hickle as hkl
import numpy as np

import matplotlib.pyplot as plt

def rdf(coordinates, box_l, simname, coords_loc=[3,6], nhis = 200,
    save=False,style='matplotlib'):
    """ Radial distribution function"""
    g,r = compute.radial_distribution_function(coordinates, box_l, simname, 
        coords_loc, nhis)
    plotting_functions.radial_distribution_function_plotting(g,r,simname,save,style)

def compare_rdf(simname,nhis,bool_first,coords_loc=[3,6],linewidthv=2,save=False):
    fig = plt.figure(figsize=(8,8))
    plt.xlabel('$r$',fontsize=20)
    plt.ylabel('$g(r)$',fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    for index,filename in enumerate(simname):
        if bool_first[index]==0:
            df,bs=extract.extract_unwrapped(filename,last_only=True,boxsize=True)
        elif bool_first[index]==1:
            df,bs=extract.extract_unwrapped(filename,first_only=True,boxsize=True)
        coord=df['timestep_0']
        del df
        g,r = compute.radial_distribution_function(coord,bs,filename,
        coords_loc,nhis)
        plt.plot(r,g,linewidth=linewidthv,label=filename)
    plt.legend()
    if save:
        plt.savefig(simname[index] +'_comparison.png',dpi=300)

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

    hkl.dump(cop_x, dumpnamex)
    hkl.dump(cop_y, dumpnamey)
    hkl.dump(cop_z, dumpnamez)
    return zip(cop_x, cop_y, cop_z)

def radius_of_gyration_squared_integrated(*args,mass=1):
    """
    Enter the filenames of multiple parts of the same simulation
    and this file will return the cumulative R_g^2 for all of them
    combined.

    Args:
    *args (string): filenames of the simulation.

    Returns:
    Plot of R_g ^2vs time and R_g^2
    """
    rog_sq=[]
    rend_sq=[]
    for filename in args:
        df_unwrap=extract.extract_unwrapped(filename)
        rend_sq_curr,gauss_curr = compute.end_to_end_distance_squared(df_unwrap)
        rog_sq_curr=compute.radius_of_gyration_squared(df_unwrap,mass)
        rog_sq=rog_sq+rog_sq_curr
        rend_sq=rend_sq+rend_sq_curr
    return rog_sq,rend_sq

def msd_integrated(*args,coord_loc=[3,6]):
    """
    Args:
    *args (string): filenames of the simulation.

    Returns:
    msd (cumulative list)
    """ 
    msd=[]
    df_unwrap=extract.extract_unwrapped(args[0])
    r_0=df_unwrap['timestep_0']
    r_0.sort_values(by=['id'],inplace=True)
    r_0=r_0.iloc[:,coord_loc[0]:coord_loc[1]].values
    for filename in args:
        df_unwrap=extract.extract_unwrapped(filename)
        msd_curr=compute.msd(df_unwrap,r_0,coord_loc)
        msd = msd + msd_curr
    return msd

def msd_molecular_com_integrated(*args,coord_loc=[3,6]):
    """
    Args:
    *args (string): filenames of the different sequences/parts of the SAME
    simulation.

    Returns:
    msd(cumulative list)
    """
    msd=[]
    df_unwrap=extract.extract_unwrapped(args[0])
    r_0=df_unwrap['timestep_0']
    total_chains=sorted(r_0.mol.unique())
    r_0_com=np.empty((len(total_chains),3),dtype=None)
    for chain in total_chains:
            curr_chain=r_0[r_0['mol']==chain].iloc[:,coord_loc[0]:coord_loc[1]].values
            curr_com=np.mean(curr_chain,axis=0)
            r_0_com[int(chain-1),:]=curr_com
    for filename in args:
        df_unwrap=extract.extract_unwrapped(filename)
        msd_curr=compute.msd_molecular_com(df_unwrap,r_0_com,coord_loc)
        msd=msd +msd_curr
    return msd

def msd_com_inner_integrated(*args,coord_loc=[3,6]):
    """
    """
    msd=[]
    df_unwrap=extract.extract_unwrapped(args[0])
    r_0=df_unwrap['timestep_0']
    total_chains=sorted(r_0.mol.unique())
    r_0_com=np.empty((len(total_chains),3),dtype=None) #Initialize a None array
    for chain in total_chains:
            curr_chain=r_0[r_0['mol']==chain].iloc[:,coord_loc[0]:coord_loc[1]].values
            curr_com=np.mean(curr_chain,axis=0)
            r_0_com[int(chain-1),:]=curr_com
    r_0.sort_values(by=['id'],inplace=True)
    r_0=r_0.iloc[:,coord_loc[0]:coord_loc[1]].values        
    for filename in args:
        df_unwrap=extract.extract_unwrapped(filename)
        msd_curr=compute.msd_com_inner(df_unwrap,r_0,r_0_com,coord_loc)
        msd=msd +msd_curr
    return msd