import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import signal
import seaborn as sns
import matplotlib as mpl

from analib import fileIO
from analib import extract
from analib import integrate

imagesavepath=r'c:/Users/Raiter/OneDrive - Cornell University/Thesis/Results/images_from_jupyter_notebook/'

def find_cumulative_pairs(distances,cut_off):
    """Return the number of pairs below the specified cut-off at each timestep

    Args:
    distances (dict): A dictionary with timesteps as keys and the distances
        between positive and negative beads (magnitude) as values at that
        timestep as a pandas dataframe.
    cut_off(float): The designated limit of the distances between pairs.

    Returns:
    num_of_pairs (np array): Count of distances below cut-off value at each
        timestep.
    """
    num_of_pairs=np.zeros(len(distances.keys()))
    for index,key in enumerate(distances):
        num_of_pairs[index]=np.asarray(
        np.where(distances[key]<cut_off)).shape[1]
    normalizer=np.amax(num_of_pairs)
    num_of_pairs_cumulative=num_of_pairs/normalizer
    return num_of_pairs_cumulative

def find_hypothetical_pairs(dist_vec,distances,prop_constant,cut_off):
    """Find the number of hypothetical pairs below the specified cutoff"""
    hypothetical_pair={}
    index_list = np.asarray(np.where(distances['timestep_0'] < cut_off))
    hypothetical_pair[0] = np.take(dist_vec['timestep_0'],index_list,axis=0)
    initial_distribution = hypothetical_pair[0][0]
    hpair=np.zeros(len(dist_vec.keys()))
    for hyp_index in range(len(dist_vec.keys())):
        hypothetical_pair[hyp_index] = np.multiply(
            initial_distribution,prop_constant[hyp_index])
        hypothetical_pair[hyp_index]=np.linalg.norm(
            hypothetical_pair[hyp_index],axis=1)
        hpair[hyp_index]=np.asarray(
            np.where(hypothetical_pair[hyp_index] < cut_off)).shape[1]
    normalizer=np.amax(hpair)
    num_of_pairs_hypothetical=hpair/normalizer
    return num_of_pairs_hypothetical

def find_tracked_pairs(distances,cut_off):
    """Find the number of tracked pairs below the specified cutoff"""
    num_of_pairs=np.zeros(len(distances.keys()))
    index_list = np.asarray(np.where(distances['timestep_0'] < cut_off))
    for index,key in enumerate(distances):
        num_of_pairs[index] = np.asarray(
            np.where(np.take(distances[key],index_list) < cut_off)).shape[1]
    normalizer=np.amax(num_of_pairs)
    num_of_pairs_tracked=num_of_pairs/normalizer
    return num_of_pairs_tracked

def set_strain_props(ax):
    """Set properties for strain (x-axis) for the given axis object"""
    fontProperties = {'family':'serif','size':20}
    ax.set_xlim(0,1.71)
    ax.set_xticklabels(ax.get_yticks(),{'family':'serif','size':14})
    ax.set_xlabel('Strain',fontProperties)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    return ax

def set_y_props(ax,label):
    """"Set properties for y-axis with the given name"""
    fontProperties = {'family':'serif','size':20}
    ax.set_yticklabels(ax.get_yticks(),{'family':'serif','size':14})
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_ylabel(label,fontProperties)
    return ax

def set_x_props(ax,label):
    """"Set properties for y-axis with the given name"""
    fontProperties = {'family':'serif','size':20}
    ax.set_xticklabels(ax.get_xticks(),{'family':'serif','size':14})
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_xlabel(label,fontProperties)
    return ax

def plot_cutoff_pairs(distances,*args,save=False,dotsperinch=300):
    """Find out the number of pairs in the distances dictionary below the
    cut-off value (at each timestep) and plot them against strain 
    (or timestep).

    Args:
    distances (dict): A dictionary with timesteps as keys and the distances
        between positive and negative beads (magnitude) as values at that
        timestep as a pandas dataframe.
    *args (float): Cut-off values for pairs. For multiple values, the function
    
    Returns: 
    Plot with an independent curve for each cut-off.
    """
    num_of_pairs=np.zeros(len(distances.keys()))
    strain=np.arange(0,len(distances.keys())/100,0.01)
    fig1=plt.figure(figsize=(8,6),dpi=150)
    ax=plt.gca()
    ax=set_strain_props(ax)
    ax=set_y_props(ax,'Number of pairs')
    for index_b,arg in enumerate(args):
        num_of_pairs=find_cumulative_pairs(distances,arg)
        ax.plot(strain,num_of_pairs,label='cut-off = '+str(arg),linewidth=2)
    ax.legend()
    plt.show()
    if save:

        fname = fileIO.default_path + 'cutoffs_' + str(args)
        plt.savefig(fname,dotsperinch)
    #return num_of_pairs

def specific_cutoff_pairs(distances,dist_vec,cut_off,
                          prop_constant,pngname,save=False):
    """Plot tracked pairs, hypothetical pairs and cumulative pairs for one
    set. Check ____ for multiple plots.
    Args:
    distances (dict): a dictionary containing the distances between all the
        type 2 and type 3 (charged) beads.
    distances_vec (dict): A dictionary with timesteps as keys and the
        distances between positive and negative beads (vector) as values at
        that timestep as a pandas dataframe. 
    cut_off (float): The designated limit of the distances between pairs.
    
    _constant (np array): an array  containing the proportionality
        constant pertaining to the increase in size of
        the box (along x, y and z) at every timestep.
    
    Returns:
    Plot with tracked, hypothetical and cumulative pairs.
    """
    num_of_pairs_cumulative = find_cumulative_pairs(distances,cut_off)
    num_of_pairs_tracked = find_tracked_pairs(distances,cut_off)
    num_of_pairs_hypothetical = find_hypothetical_pairs(dist_vec,distances,
                                                        prop_constant,cut_off)

    strain=np.linspace(0,1.71,len(distances.keys()))
    fig1=plt.figure(figsize=(8,6),dpi=150)
    ax=plt.gca()
    ax=set_strain_props(ax)
    ax=set_y_props(ax,'Number of pairs')
    ax.plot(strain,num_of_pairs_cumulative,
        label='cut-off (cumulative pairs)',linewidth=2)
    ax.plot(strain,num_of_pairs_hypothetical,
        label='cut-off (hypothetical pairs)',linewidth=2)
    ax.plot(strain,num_of_pairs_tracked,color='r',
        label='cut-off (tracked pairs)',linewidth=2)
    title='Cut-off = ' + str(cut_off)
    fontProperties = {'family':'serif','size':20}
    ax.set_title(title,fontProperties)
    ax.legend()
    plt.show()
    if save:
        saveloc= fileIO.default_path + pngname + '_charge_pair_analysis_cut_off_'\
            + str(cut_off)
        fig1.savefig(saveloc,dpi=300)

def plot_charged_compars(*args,cut_off=2,pngname='plot_charged_compars',save=False,dotsperinch=300):
    """ Similar to specific_cutoff_pairs for multiple simulations"""
    distances=args[0]
    strain=np.linspace(0,1.71,len(distances.keys()))
    fig1=plt.figure(figsize=(8,6),dpi=150)
    ax=plt.gca()
    index_out=0
    ax=set_strain_props(ax)
    ax=set_y_props(ax,'Number of pairs')
    c=['C3','C0']
    while index_out<len(args):
        distances=args[index_out]
        prop_constant=args[index_out+1]
        dist_vec=args[index_out+2]
        num_of_pairs_tracked = find_tracked_pairs(distances,cut_off)
        num_of_pairs_hypothetical=find_hypothetical_pairs(dist_vec,distances,
                                                          prop_constant,
                                                          cut_off)
        num_of_pairs_cumulative = find_cumulative_pairs(distances,cut_off)
        ax.plot(strain,num_of_pairs_tracked,'-',color=c[index_out//3])
        ax.plot(strain,num_of_pairs_cumulative,'--',color = c[index_out//3])
        ax.plot(strain,num_of_pairs_hypothetical,':',color=c[index_out//3])
        index_out+=3
    red_patch = mpatches.Patch(color='C3',label = 'v20')
    green_patch = mpatches.Patch(color='C0',label = 'v20 (last stage efield')
    first_legend = plt.legend(handles=[red_patch,green_patch],loc='lower left')
    axl = plt.gca().add_artist(first_legend)
    custom_lines = [Line2D([0], [0], color='k', lw=1),
                Line2D([0], [0], color='k', linestyle='--',lw=1),
                Line2D([0], [0], color='k', linestyle = ':',lw=1)]
    ax.legend(custom_lines,['Tracked','Cumulative','Hypothetical'])
    title = "Cut off = " + str(cut_off)
    ax.set_title(title,fontsize=20,fontfamily='Serif')
    if save:
        fname = fileIO.default_path + pngname + '_comparison_charged_cutoff_' + str(cut_off) +'.png'
        fig1.savefig(fname,dpi=dotsperinch)

def plot_charged_compars_2(cut_off,*args,save=False,dotsperinch=300):
    """ Similar to specific_cutoff_pairs for multiple simulations"""
    distances=args[0]
    strain=np.arange(0,len(distances.keys())/100,0.01)
    fig1=plt.figure(figsize=(8,6),dpi=150)
    ax=plt.gca()
    index_out=0
    ax=set_strain_props(ax)
    ax=set_y_props(ax,'Number of pairs')
    c=['C3','g','C0']
    while index_out<len(args):
        distances=args[index_out]
        dist_vec=args[index_out+1]
        num_of_pairs_tracked = find_tracked_pairs(distances,cut_off)
        num_of_pairs_hypothetical= dist_vec
        num_of_pairs_cumulative = find_cumulative_pairs(distances,cut_off)
        ax.plot(strain,num_of_pairs_tracked,'-',color=c[index_out])
        ax.plot(strain,num_of_pairs_cumulative,'--',color = c[index_out])
        ax.plot(strain,num_of_pairs_hypothetical,':',color=c[index_out])
        index_out+=2
    red_patch = mpatches.Patch(color='C3',label = '0.05 V/A')
    green_patch = mpatches.Patch(color='C0',label = '0.00 V/A')
    first_legend = plt.legend(handles=[red_patch,green_patch])
    axl = plt.gca().add_artist(first_legend)
    custom_lines = [Line2D([0], [0], color='k', lw=1),
                Line2D([0], [0], color='k', linestyle='--',lw=1),
                Line2D([0], [0], color='k', linestyle = ':',lw=1)]
    ax.legend(custom_lines,['Tracked','Cumulative','Hypothetical'])
    title = "Cut off = " + str(cut_off)
    ax.set_title(title,fontsize=20,fontfamily='Serif')
    plt.show()
    if save:
        fname = fileIO.default_path + '_comparison_charged_' + str(cut_off) + '.png'
        fig1.savefig(fname,dpi=dotsperinch)

def understand_variation(*args,smoothed=False,save=False,msd=False,
    strain_end=1.718,units='lj'):
    """
    Plot data for repeats of same simulation to understand intra-model
    variation.

    Args:
    fname (str) : Names of the simulation.
    smoothed (=False) : Apply a Savgol filter to smooth curves
    """
    fig1=plt.figure(figsize=[8,8],dpi=100)
    for index,arg in enumerate(args):
        df1,df2=extract.extract_def(arg)
        df1=df1.values
        df2=df2.values
        deformation_along = np.argmax(np.array([np.std(df1[:,1]),
                                      np.std(df1[:,2]),
                                      np.std(df1[:,3])]))
        plt.figure(1)
        till = len(df1[:,deformation_along+1])
        strain=np.linspace(0,strain_end,till)
        if smoothed:
            if units=='real':
                y_noise = df1[:till,deformation_along+1]*1000
            elif units=='lj':
                y_noise = df1[:till,deformation_along+1]
            y=signal.savgol_filter(y_noise,
                                   357, # window size used for filtering
                                   1,
                                   mode='nearest') # order of polynomial
            y=np.asarray(y)
            plt.plot(strain[:till],y.T,linewidth=2,label=arg)
        else:
            #strain=np.arange(0,1.718,0.00)
            if units=='real':
                plt.plot(strain[:till],df1[:till,deformation_along+1]*1000,
                     linewidth=2,label=arg)
            elif units=='lj':
                plt.plot(strain[:till],df1[:till,deformation_along+1],
                     linewidth=2,label=arg)
        if msd:
            plt.figure(2)
            ax2=plt.gca()
            ax2.plot(strain,df2[:,4],label='positive')
            ax2.plot(strain,df2[:,8],label='negative')
            ax2.plot(strain,df2[:,12],label='ions')
            ax2.plot(strain,df2[:,16],label='neutral',linestyle=':',
            linewidth=2)
        #ax2.plot(strain,df2[:,20],label=arg,linewidth=2)
        #ax3.plot(df2[:,0],df2[:,19],label='positive')
        #ax3.plot(df2[:,0],df2[:,22],label='negative')
        #ax3.plot(df2[:,0],df2[:,25],label='ions')
        #ax3.plot(df2[:,0],df2[:,28],label='neutral')
        #plt.figure(3)
        #plt.plot(strain,df2[:,35],label=arg)
    plt.figure(1)
    plt.xlabel('Strain',fontsize=20,fontfamily='serif')
   
    if units=='real':
        plt.ylabel('Stress (MPa)',fontsize=20,fontfamily='serif')
    elif units=='lj':
        plt.ylabel('Stress',fontsize=20,fontfamily='serif')
    ax = plt.gca()
    ax = set_strain_props(ax)
    fontProperties = {'family':'serif','size':16}
    ax.set_yticklabels(ax.get_yticks(), fontProperties)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.legend()
    plt.tight_layout()
    if msd:
        ax2=set_strain_props(ax2)
        plt.ylabel('MSD ($\AA ^2$)',fontsize=20,fontfamily='serif')
        plt.legend()
        plt.close()
        plt.figure(3)
        plt.xlabel('Strain',fontsize=20,fontfamily='Serif')
        plt.ylabel('Non Gaussian Parameter',fontsize=20,fontfamily='Serif')
        plt.legend()
        plt.close()
    full_path = imagesavepath + str(args[0])[:-4]
    if save:
        if smoothed:
            fname=full_path +'_smoothed.png'
            print(fname)
            fig1.savefig(fname,dpi=300)
        else:
            fname=full_path + '.png'
            print(fname)
            fig1.savefig(fname,dpi=300)

def plot_multiple_numpy(*args,smoothed=False):
    """Plot graphs for given arguments
    
    Args:
    *args (str): numpy arrays for each separate simulation containing the
    def 1 file information"""
    fig1=plt.figure(figsize=[8,8],dpi=150)        
    plt.figure(1)
    strain=np.arange(0,1.718,0.001)
    labels=['1000 DP','2000 DP','3000 DP']
    till = len(strain)
    for index,arg in enumerate(args):
        deformation_along = np.argmax(np.array([np.std(arg[:,1]),np.std(arg[:,2]),np.std(arg[:,3])]))
        if smoothed:
            y_noise = arg[:till,deformation_along+1]*1000
            y=signal.savgol_filter(y_noise,
                                   357, # window size used for filtering
                                   1,
                                   mode='nearest') # order of fitted polynomial
            y=np.asarray(y)
            plt.plot(strain[:till],y.T,linewidth=2,label=labels[index])
            
        else:
            plt.plot(strain[:till],arg[:till,deformation_along+1]*1000,linewidth=2,label=labels[index])
    plt.xlabel('Strain',fontsize=20,fontfamily='serif')
    plt.legend()
    plt.ylabel('Stress (MPa)',fontsize=20,fontfamily='serif')
    ax=plt.gca()
    ax=set_strain_props(ax)
    ax.legend(prop=dict(size=16))
    fontProperties = {'family':'serif','size':16}
    ax.set_yticklabels(ax.get_yticks(), fontProperties)
    plt.tight_layout()
    fname = str(labels)
    fname = fname.replace(',', '').replace('\'','').replace(' ','').strip('[]\'')
    full_path = imagesavepath + fname
    if smoothed:
        plt.savefig(full_path+'_smoothed.png',dpi=300)
    else:
        plt.savefig(full_path+'.png',dpi=300)

def structure_factor_plotting(k,sk,simname,save,style):
    if style=='matplotlib':
        mpl.style.use('default')
    elif style == 'sns':
        sns.set()
    fig = plt.figure(figsize=(8,6))
    ax = plt.gca()
    ax.plot(k,sk,linewidth=2) #Plot rdf and set chart properties
    set_y_props(ax,'$s(k)$')
    set_x_props(ax,'$k$')
    figname=simname+'_sk.png'
    title='Structure Factor'
    ax.set_title(title,fontsize=20,fontfamily='Serif')
    if save:
        plt.savefig(figname,dpi=300)

def radial_distribution_function_plotting(g,r,simname,save,style):
    """Plotting the radial distribution function"""
    if style=='matplotlib':
        mpl.style.use('default')
    elif style == 'sns':
        sns.set()
    fig = plt.figure(figsize=(8,6))
    ax = plt.gca()
    ax.plot(r,g,linewidth=1) #Plot rdf and set chart properties
    set_y_props(ax,'$g(r)$')
    set_x_props(ax,'$r$')
    figname=simname+'_rdf.png'
    title='Radial Distribution Function'
    ax.set_title(title,fontsize=20,fontfamily='Serif')
    if save:
        plt.savefig(figname,dpi=300)
    
def plot_standard_analysis(simname, nfiles, start=1000,save=False, style='matplotlib'):
    """
    Plot all the variables extracted integrate.standard_analysis
    Args:
    simname (string): Name of the simulation (end name till density (uk)).
    Do not add simulation number (eg: uk_1)
    nfiles (int): Number of simulation parts for this specific simulation
    save (bool): Whether to save the plots in a directory.
    style (string): style of the plots e.g.: 'seaborn','fivethirtyeight',etc.
    """

    aggregate_data_pd = integrate.standard_analysis(simname, nfiles)
    fig=plt.figure(figsize=(18,35))
    #time=np.arange((start/10000) +0.0001,(len(aggregate_data_pd)/10000)+0.0001,0.0001)
    #time=np.arange((start/10000) +0.0001,(len(aggregate_data_pd)/10000),0.0001)
    time_length,_=aggregate_data_pd.shape
    time=np.arange(start,time_length)
    for index, column in enumerate(aggregate_data_pd):
        if index!=0:
            ax = fig.add_subplot(5,2,index)
            ax.plot(time,aggregate_data_pd[column].iloc[start:])
            ax=set_x_props(ax,'Time')
            ax=set_y_props(ax, str(column))
    curr_fname = simname + '_time_vs_all_parameters_starting_at_timestep_' + str(start)  + '.png'
    if save:
        full_path = imagesavepath + curr_fname
        plt.savefig(full_path,dpi=100)

def plot_chain_orientation_parameter(simname, nc, dp, nfiles, 
    save = False,style='matplotlib'):
    """
    Compute chain orientation parameter from each unwrapped file involved in the
    simulation and concatenate the results into one array. Plot the array
    against time (ns)

    Args:
    simname (string): Name of the simulation (type the name of the simulation
    only till density which is mostly written as 'uk'). Do not add simulation
    number (e.g. uk_1).
    nfiles (int): Number of simulation parts for this specific simulation
    save (bool): Whether to save the plot in a specified directory
    style (string): style of the plots e.g.: 'seaborn','fivethirtyeight',etc.
    """
    aggregate_cop_array = integrate.chain_orientation_parameter_combined(simname,
    nfiles,nc,dp)
    cop_x, cop_y, cop_z = zip(*aggregate_cop_array)
    fig=plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    time = np.arange(0, len(cop_x), 1)
    ax.plot(time, cop_x, label ='x')
    ax.plot(time, cop_y, label ='y')
    ax.plot(time, cop_z, label ='z')
    ax.legend()
    ax = set_x_props(ax, 'Time')
    ax = set_y_props(ax, 'Chain orientation parameter')
    if save:
        curr_name = simname + '_cop_vs_time.png'
        full_path = imagesavepath + curr_name
        plt.savefig(full_path, dpi=100)


def energy_evolution_single(simname, save=False):
    """
    Find out energy evolution during the deformation of the given simulation.

    Args:
    simname (string): Name of the simulation
    save (bool): Whether to save the plot
    """
    def1, def2 = extract.extract_def(simname)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    #epair = (def1.iloc[:,8] - def1.iloc[0,8]).values
    ebond = (def1.iloc[:,9] - def1.iloc[0,9]).values
    ecoul = (def1.iloc[:,12] - def1.iloc[0,12]).values
    evdwl = (def1.iloc[:,13] - def1.iloc[0,13]).values
    etotal = (def1.iloc[:,14] - def1.iloc[0,14]).values
    strain=np.linspace(0,1.718,len(etotal))
    #ax.plot(strain, epair, label = 'pairwise')
    ax.plot(strain, ebond, label = 'bonded')
    ax.plot(strain, ecoul, label = 'coulombic')
    ax.plot(strain, evdwl, label = 'Van der Waals')
    ax.plot(strain, etotal, label = 'total')
    ax = set_x_props(ax, 'Strain')
    ax = set_y_props(ax, 'Energy')
    ax.set_xlim(0,1.71)
    ax.legend(fontsize = 14)
    plt.tight_layout()
    if save:
        curr_name = simname + '_energy_evolution_deformation.png'
        full_path = imagesavepath + curr_name
        plt.savefig(full_path)


def energy_evolution_comparison(simname, ids, save = False):
    """
    Compare energy evolution along different directions for the 
    given simulation.

    Args:
    simname (string): Name of the simulation
    ids: if '2_x', '3_x' then [2,3]
    save (bool): Whether to save the plot
    """
    dirs = ['x','y','z','x','y','z']
    df1 = {}
    df2 = {}
    col_nums = [8,9,12,13,14,17, 7, 15,16]
    col_labels = ['pairwise','bonded','coulombic','Van der Waals','total','density', 'temp','pe','ke']
    for index in range(6):
        fname = simname + '_' + str(ids[int(index/3)]) + '_' + dirs[index]
        df1[index], df2[index] = extract.extract_def(fname)
    fig = plt.figure(figsize=(20,18))
    strain=np.arange(0,1.718,0.001)
    for index in range(9):
        ax = fig.add_subplot(3,3,index+1)
        for inner_index in range(6):
            #curr_energy = (df1[inner_index].iloc[:,col_nums[index]] - df1[inner_index].iloc[0,col_nums[index]]).values
            #ax.plot(strain, curr_energy, label = dirs[inner_index])
            ax.plot(strain, df1[inner_index].iloc[:,col_nums[index]], label = dirs[inner_index])
            ax.legend(fontsize = 16)
            ax.set_title(col_labels[index],fontsize=20)
            ax = set_x_props(ax, 'Strain')
            ax = set_y_props(ax, 'Energy')
            ax.set_xlim(0,1.71)
    if save:
        curr_name = simname + '_energy_evolution_deformation_comparison.png'
        full_path = imagesavepath + curr_name
        plt.tight_layout()
        plt.savefig(full_path)
    
    
    

