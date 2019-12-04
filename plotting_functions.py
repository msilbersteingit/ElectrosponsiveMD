import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import signal

from analib import fileIO
from analib import extract

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
    strain=np.arange(0,2.0,0.25)
    fontProperties = {'family':'serif','size':20}
    ax.set_xticklabels(strain,{'family':'serif','size':14})
    ax.set_xlabel('Strain',fontProperties)
    ax.set_xlim(0,1.71)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    return ax

def set_y_props(ax,label):
    """"Set properties for y-axis with the given name"""
    fontProperties = {'family':'serif','size':20}
    ax.set_yticklabels(ax.get_yticks(),{'family':'serif','size':14})
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_ylabel(label,fontProperties)
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
                          prop_constant,save=False):
    """Plot tracked pairs, hypothetical pairs and cumulative pairs for one
    set. Check ____ for multiple plots.
    Args:
    distances (dict): a dictionary containing the distances between all the
        type 2 and type 3 (charged) beads.
    distances_vec (dict): A dictionary with timesteps as keys and the
        distances between positive and negative beads (vector) as values at
        that timestep as a pandas dataframe. 
    cut_off (float): The designated limit of the distances between pairs.
    prop_constant (np array): an array  containing the proportionality
        constant pertaining to the increase in size of
        the box (along x, y and z) at every timestep.
    
    Returns:
    Plot with tracked, hypothetical and cumulative pairs.
    """
    num_of_pairs_cumulative = find_cumulative_pairs(distances,cut_off)
    num_of_pairs_tracked = find_tracked_pairs(distances,cut_off)
    num_of_pairs_hypothetical = find_hypothetical_pairs(dist_vec,distances,
                                                        prop_constant,cut_off)

    strain=np.arange(0,len(distances.keys())/100,0.01)
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
        saveloc= fileIO.default_path + 'charge_pair_analysis_cut_off_'\
            + str(cut_off)
        plt.savefig(saveloc,dpi=300)

def plot_charged_compars(cut_off,*args,save=False,dotsperinch=300):
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
        prop_constant=args[index_out+1]
        dist_vec=args[index_out+2]
        num_of_pairs_tracked = find_tracked_pairs(distances,cut_off)
        num_of_pairs_hypothetical=find_hypothetical_pairs(dist_vec,distances,
                                                          prop_constant,
                                                          cut_off)
        num_of_pairs_cumulative = find_cumulative_pairs(distances,cut_off)
        ax.plot(strain,num_of_pairs_tracked,'-',color=c[index_out])
        ax.plot(strain,num_of_pairs_cumulative,'--',color = c[index_out])
        ax.plot(strain,num_of_pairs_hypothetical,':',color=c[index_out])
        index_out+=3
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
    if save:
        fname = fileIO.default_path + '_comparison_charged.png'
        plt.savefig(fname,dotsperinch)

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

def understand_variation(*args,smoothed=False,save=False):
    """
    Plot data for repeats of same simulation to understand intra-model
    variation.

    Args:
    fname (str) : Names of the simulation.
    smoothed (=False) : Apply a Savgol filter to smooth curves
    """
    fig1=plt.figure(figsize=[8,8],dpi=100)
    fig2=plt.figure(figsize=[8,8],dpi=100)
    fig3=plt.figure(figsize=[8,8],dpi=100)
    strain=np.arange(0,1.718,0.001)
    till = len(strain)
    for index,arg in enumerate(args):
        df1,df2=extract.extract_def(arg)
        df1=df1.values
        df2=df2.values
        deformation_along = np.argmax(np.array([np.std(df1[:,1]),
                                      np.std(df1[:,2]),
                                      np.std(df1[:,3])]))
        plt.figure(1)
        if smoothed:
            y_noise = df1[:till,deformation_along+1]*1000
            y=signal.savgol_filter(y_noise,
                                   357, # window size used for filtering
                                   1,
                                   mode='nearest') # order of polynomial
            y=np.asarray(y)
            plt.plot(strain[:till],y.T,linewidth=2,label=arg)
        else:
            plt.plot(strain[:till],df1[:till,deformation_along+1]*1000,
                     linewidth=2,label=arg)
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
    plt.ylabel('Stress (MPa)',fontsize=20,fontfamily='serif')
    strain_labels=np.arange(0,1.75,0.25)
    ax = plt.gca()
    ax = set_strain_props(ax)
    fontProperties = {'family':'serif','size':16}
    ax.set_yticklabels(ax.get_yticks(), fontProperties)
    plt.legend()
    plt.show()
    ax2=set_strain_props(ax2)
    plt.ylabel('MSD ($\AA ^2$)',fontsize=20,fontfamily='serif')
    plt.legend()
    plt.close()
    plt.figure(3)
    plt.xlabel('Strain',fontsize=20,fontfamily='Serif')
    plt.ylabel('Non Gaussian Parameter',fontsize=20,fontfamily='Serif')
    plt.legend()
    plt.close()
    path=r'c:/Users/Raiter/OneDrive - Cornell University/Thesis/Results/11052019/atg/'
    #arg_name = str(args[0])[:-4]
    full_path = path+ str(args[0])[:-4]
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
    """Plot graphs for given """
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
    path=r'c:\Users\Raiter\OneDrive - Cornell University\Thesis\Results\11052019\atg\\'
    fname = str(labels)
    fname = fname.replace(',', '').replace('\'','').replace(' ','').strip('[]\'')
    full_path = path + fname
    if smoothed:
        plt.savefig(full_path+'_smoothed.png',dpi=300)
    else:
        plt.savefig(full_path+'.png',dpi=300)