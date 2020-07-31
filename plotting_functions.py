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
from analib import compute

imagesavepath=r'c:/Users/Raiter/OneDrive - Cornell University/Thesis/Results/images_from_jupyter_notebook/'

def find_cumulative_pairs(distances,cut_off, normalized=True):
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
    if normalized:
        normalizer=np.amax(num_of_pairs)
        num_of_pairs_cumulative=num_of_pairs/normalizer
        return num_of_pairs_cumulative
    else:
        return num_of_pairs

def find_hypothetical_pairs(dist_vec,distances,prop_constant,cut_off,normalized=True):
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
    if normalized:
        normalizer=np.amax(hpair)
        num_of_pairs_hypothetical=hpair/normalizer
        return num_of_pairs_hypothetical
    else:
        return hpair

def find_tracked_pairs(distances,cut_off,normalized=True):
    """Find the number of tracked pairs below the specified cutoff"""
    num_of_pairs=np.zeros(len(distances.keys()))
    index_list = np.asarray(np.where(distances['timestep_0'] < cut_off))
    for index,key in enumerate(distances):
        num_of_pairs[index] = np.asarray(
            np.where(np.take(distances[key],index_list) < cut_off)).shape[1]
    if normalized:
        normalizer=np.amax(num_of_pairs)
        num_of_pairs_tracked=num_of_pairs/normalizer
        return num_of_pairs_tracked
    else:
        return num_of_pairs

def find_tracked_pairs_unique(distances,simname):
    """Find the number of tracked pairs below the specified cutoff"""
    coord,bs=extract.extract_unwrapped(simname,boxsize_whole=True)
    pairs_under_cutoff = np.zeros(len(coord.keys()))
    charged_pair_distances={}
    for index,key in enumerate(coord):
        box_l=bs[key]
        sidehx=box_l[0]/2
        sidehy=box_l[1]/2
        sidehz=box_l[2]/2
        chargedpairdistance=np.zeros((1634,1))
        for index_inner, atomid in enumerate(distances['timestep_0']):
            current_timestep = coord[key]
            pcharge=current_timestep[current_timestep['id']==atomid[0]].iloc[:,3:6].values
            ncharge=current_timestep[current_timestep['id']==atomid[1]].iloc[:,3:6].values
            dx = pcharge[0][0] - ncharge[0][0]
            dy = pcharge[0][1] - ncharge[0][1]
            dz = pcharge[0][2] - ncharge[0][2]
            if (dx < -sidehx):   dx = dx + box_l[0]
            if (dx > sidehx):    dx = dx - box_l[0]
            if (dy < -sidehy):   dy = dy + box_l[1]
            if (dy > sidehy):    dy = dy - box_l[1]
            if (dz < -sidehz):   dz = dz + box_l[2]
            if (dz > sidehz):    dz = dz - box_l[2]
            distAB = [dx,dy,dz]
            chargedpairdistance[index_inner] = np.linalg.norm(distAB)
        chargedpairdistance = chargedpairdistance[chargedpairdistance!=0]
        charged_pair_distances[index] = chargedpairdistance 
        #pairs_under_cutoff[index] = np.asarray(np.where(chargedpairdistance<cut_off)).shape[1]
    return charged_pair_distances

def wrapper_unique_pairs(simname):
    ps = compute.find_pairs_unique(simname)
    cpd = find_tracked_pairs_unique(ps,simname)
    return cpd

def autcorrelation_lifetime_correlation(simname,cutoff):
    ps = compute.find_pairs_unique(simname)
    cpd = find_tracked_pairs_unique(ps, simname)
    pairs_under_cutoff=np.zeros(len(cpd.keys()))
    for index,element in enumerate(cpd):
        pairs_under_cutoff[index] = np.asarray(np.where(cpd[element]<cutoff)).shape[1]
    for index, element in enumerate(pairs_under_cutoff):
        pairs_under_cutoff[index]=pairs_under_cutoff[0]*pairs_under_cutoff[index]
    return pairs_under_cutoff

def set_strain_props(ax):
    """Set properties for strain (x-axis) for the given axis object"""
    fontProperties = {'size':22}
    ax.set_xlim(0,1.71)
    ax.set_xticklabels(ax.get_yticks(),{'size':18})
    ax.set_xlabel('Strain',fontProperties)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.tick_params(direction='in')
    return ax

def set_y_props(ax,label):
    """"Set properties for y-axis with the given name"""
    fontProperties = {'family':'Arial','size':22}
    ax.set_yticklabels(ax.get_yticks(),{'family':'Arial','size':16})
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_ylabel(label,fontProperties)
    ax.tick_params(direction='in')
    return ax

def set_x_props(ax,label):
    """"Set properties for y-axis with the given name"""
    fontProperties = {'family':'Arial','size':22}
    ax.set_xticklabels(ax.get_xticks(),{'family':'Arial','size':16})
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.set_xlabel(label,fontProperties)
    ax.tick_params(direction='in')
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

def plot_charged_compars(*args,cut_off=2,pngname='plot_charged_compars',save=False,
labels,dotsperinch=300,normalized=True,deformation_strain_end=1.72002,plot_strain_end=1.0):
    """ Similar to specific_cutoff_pairs for multiple simulations"""
    distances=args[0]
    strain=np.linspace(0,deformation_strain_end,len(distances.keys())-1)
    fig1=plt.figure(figsize=(8,8),dpi=150)
    ax=plt.gca()
    index_out=0
    ax=set_x_props(ax,'Strain')
    ax.set_xlim(0,plot_strain_end)
    ax=set_y_props(ax,'Number of pairs')
    c=['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    colornames=['red','blue','green','yellow']
    while index_out<len(args):
        distances=args[index_out]
        #prop_constant=args[index_out+1]
        #dist_vec=args[index_out+2]
        if normalized:
            num_of_pairs_tracked = find_tracked_pairs(distances,cut_off)
            #num_of_pairs_hypothetical=find_hypothetical_pairs(dist_vec,distances,
            #                                                prop_constant,
            #                                                cut_off)
            num_of_pairs_cumulative = find_cumulative_pairs(distances,cut_off)
            ax.plot(strain,num_of_pairs_tracked,'-',color=c[index_out//3])
            ax.plot(strain,num_of_pairs_cumulative,'--',color = c[index_out//3])
            #ax.plot(strain,num_of_pairs_hypothetical,':',color=c[index_out//3])
        else:
            num_of_pairs_tracked = find_tracked_pairs(distances,cut_off,normalized=False)
            #num_of_pairs_hypothetical=find_hypothetical_pairs(dist_vec,distances,
            #                                                prop_constant,
            #                                                cut_off,
            #                                               normalized=False)
            num_of_pairs_cumulative = find_cumulative_pairs(distances,cut_off,
            normalized=False)
            ax.plot(strain,num_of_pairs_tracked[1:],'-',color=c[index_out//3])
            ax.plot(strain,num_of_pairs_cumulative[1:],'--',color = c[index_out//3])
            #ax.plot(strain,num_of_pairs_hypothetical,':',color=c[index_out//3])
        index_out+=1
    patches={}
    for index in range(len(labels)):
        patches[index] = mpatches.Patch(color=c[index],label = labels[index])
    first_legend = plt.legend(handles=[patches[0],patches[1],patches[2],patches[3]],loc='lower left')
    axl = plt.gca().add_artist(first_legend)
    custom_lines = [Line2D([0], [0], color='k', lw=1),
                Line2D([0], [0], color='k', linestyle='--',lw=1),
                Line2D([0], [0], color='k', linestyle = ':',lw=1)]
    ax.legend(custom_lines,['Tracked','Cumulative'],loc='lower right')
    title = "Cut off = " + str(cut_off)[:4]
    #ax.set_title(title,fontsize=20,fontfamily='Serif')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
        'Perpendicular electric field',
        'Electric field = 1.2'))
    #ax.text(0.03,0.08,textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props)
    if save:
        plt.tight_layout()
        fname = imagesavepath + pngname + '_comparison_charged_cutoff_' + str(cut_off)[:4] +'.png'
        fig1.savefig(fname,dpi=dotsperinch)

def plot_tracked_pairs(*args,cut_off=2,pngname='tracked-pairs_',save=False,
labels,dotsperinch=300,deformation_strain_end=1.72002,plot_strain_end=1.0):
    """ Similar to specific_cutoff_pairs for multiple simulations"""
    distances=args[0]
    strain=np.linspace(0,deformation_strain_end,len(distances.keys()))
    fig1=plt.figure(figsize=(8,8),dpi=150)
    ax=fig1.add_subplot(1,1,1)
    index_out=0
    ax=set_x_props(ax,'Strain')
    ax.set_xlim(0,plot_strain_end)
    ax=set_y_props(ax,'Number of tracked pairs')
    while index_out<len(args):
        distances=args[index_out]
        num_of_pairs_tracked = find_tracked_pairs(distances,cut_off,normalized=False)
        ax.plot(strain,num_of_pairs_tracked,label=labels[index_out])
        index_out+=1
    ax.legend(fontsize=16)
    title = "Cut off = " + str(cut_off)
    ax.set_title(title,fontsize=20,fontfamily='Serif')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.tight_layout()
    if save:
        fname = imagesavepath + pngname + '_comparison_charged_cutoff_' + str(cut_off) +'.png'
        fig1.savefig(fname,dpi=dotsperinch)

def plot_unique_tracked_pairs(*args,cut_off=2,pngname='tracked-pairs_',save=False,
labels,dotsperinch=300,deformation_strain_end=1.72002,plot_strain_end=1.0):
    """ Similar to specific_cutoff_pairs for multiple simulations"""
    fig1=plt.figure(figsize=(8,8),dpi=150)
    strain = np.linspace(0,deformation_strain_end,)

    ax=fig1.add_subplot(1,1,1)
    index_out=0
    while index_out<len(args):
        matched_pairs = compute.find_pairs_unique(args[0])
        pairs_under_cutoff = compute.find_tracked_pairs_unique(matched_pairs,args[0],cut_off=cut_off)
        ax.plot(strain,pairs_under_cutoff,label=labels[index_out])
        index_out+=1
    ax=set_x_props(ax,'Strain')
    ax.set_xlim(0,plot_strain_end)
    ax=set_y_props(ax,'Number of tracked pairs')
    ax.legend(fontsize=16)
    if save:
        fname = imagesavepath + pngname + '_comparison_charged_cutoff_' + str(cut_off) +'.png'
        fig1.savefig(fname,dpi=dotsperinch)


def understand_variation(*args,labels,smoothed=False,save=False,msd=False,
    deformation_strain_end=1.72002,plot_strain_end=1.0,units='lj'):
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
        strain=np.linspace(0,deformation_strain_end,till)
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
                     linewidth=2,label=labels[index])
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
    plt.legend(fontsize=16)
    ax.set_xlim(0,plot_strain_end)
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

def return_mean_stress(*args):
    """Read stress for the each of the given files and return the mean stress.
    Typically used for finding out the mean stress for a deformation of
    the same system along different directions or at different equilibration
    times.

    Args:
    *args (str): file name for the deformation.

    Returns:
    m_stress (list): a list containing the mean stress for the given deformations.
    """
    df1,df2=extract.extract_def(args[0])
    m_stress=np.zeros((df1.shape[0],len(args)))
    for index, arg in enumerate(args):
        df1,df2=extract.extract_def(arg)
        df1=df1.values
        df2=df2.values
        deformation_along = np.argmax(np.array([np.std(df1[:,1]),
                                      np.std(df1[:,2]),
                                      np.std(df1[:,3])]))
        m_stress[:,index]=df1[:,deformation_along+1]
    m_stress = np.mean(m_stress,axis=1)
    return m_stress

def return_mean_temperature(*args):
    """Read temperature for the each of the given files and return the mean temperature.
    Typically used for finding out the mean temperature for a deformation of
    the same system along different directions or at different equilibration
    times.

    Args:
    *args (str): file name for the deformation.

    Returns:
    m_temperature (list): a list containing the mean stress for the given deformations.
    """
    df1,df2=extract.extract_def(args[0])
    m_temperature=np.zeros((df1.shape[0],len(args)))
    for index, arg in enumerate(args):
        df1,df2=extract.extract_def(arg)
        df1=df1.values
        df2=df2.values
        m_temperature[:,index]=df1[:,7]
    m_temperature = np.mean(m_temperature,axis=1)
    return m_temperature

def plot_mean_quantities(*args,ylabel,colors,labels,title,
deformation_strain_end=1.72002,plot_strain_end=1.0,ncol=1,ylim=None,
loc='best',save=False,legend_font=14,linewidth=1.0):
    """Plot mean quantities """
    fig = plt.figure(figsize=[5,4],dpi=100)
    ax=fig.add_subplot(1,1,1)
    strain=np.linspace(0,deformation_strain_end,len(args[0]))
    for index, arg in enumerate(args):
        ax.plot(strain,arg,label=labels[index],color=colors[index],linewidth=linewidth)
    ax = set_x_props(ax, 'Strain')
    fontProperties = {'family':'Arial','size':legend_font}
    fontPropertieslabel = {'family':'Arial','size':22}
    ax.set_yticklabels(ax.get_yticks(),{'family':'Arial','size':16})
    ax.set_ylabel(ylabel,fontPropertieslabel)
    ax.tick_params(direction='in')
    ax.set_xlim(0,plot_strain_end)
    if ylim:
        ax.set_ylim(top=ylim)
    leg=ax.legend(prop=fontProperties,frameon=False,ncol=ncol,loc=loc)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    if save:
        plt.tight_layout()
        full_path = imagesavepath + title + '.png'
        plt.savefig(full_path,dpi=300)
    return ax

def plot_multiple_numpy(*args, labels,colors, loc='best',deformation_strain_end=1.72002, 
    plot_strain_end=1.0,save=False,linewidth=1,axes_width=1,legend_font=14,
    ylim=None,ncol=1):
    """Plot graphs for given arguments
    
    Args:
    *args (str): numpy arrays for each separate simulation containing the
    def 1 file information"""
    fig1=plt.figure(figsize=[5,4],dpi=100)        
    ax = fig1.add_subplot(1,1,1)
    strain=np.linspace(0.00016345,deformation_strain_end,len(args[0]))
    till = len(strain)
    fontProperties = {'family':'Arial','size':legend_font}
    for index,arg in enumerate(args):
        ax.plot(strain,arg,linewidth=linewidth,label=labels[index],color=colors[index])
    ax.set_xlim(0,plot_strain_end)
    if ylim:
        ax.set_ylim(0,ylim)
    else:
        ax.set_ylim(bottom=0)
    ax=set_x_props(ax,'Strain')
    ax=set_y_props(ax,'Stress (LJ units)')
    leg=ax.legend(prop=fontProperties,frameon=False,ncol=ncol,loc=loc)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    plt.tight_layout()
    plt.setp(ax.spines.values(), linewidth=axes_width)
    ax.xaxis.set_tick_params(width=axes_width)
    ax.yaxis.set_tick_params(width=axes_width)
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    if save:
        fname = str(labels)
        fname = fname.replace(',', '').replace('\'','').replace(' ','').strip('[]\'')
        full_path = imagesavepath + fname
        plt.savefig(full_path+'.png',dpi=300)
    return ax

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

def radial_distribution_function_plotting(*args,labels,figname,save=False):
    """Plotting the radial distribution function"""
    fig = plt.figure(figsize=(8,6),dpi=100)
    ax = fig.add_subplot(1,1,1)
    rdf_dict={}
    for simfile in args:
        rdf_dict[simfile] = pd.read_csv(simfile,delim_whitespace=True)
    for index,simfile in enumerate(args):
        ax.plot(rdf_dict[simfile].iloc[:,0],rdf_dict[simfile].iloc[:,1],label=labels[index]) #Plot rdf and set chart properties
    ax = set_y_props(ax,'$g(r)$')
    ax = set_x_props(ax,'$r$')
    ax.set_xlim(0,3)
    ax.set_ylim(0,10)
    ax.legend(fontsize=13)
    title='Radial Distribution Function'
    #ax.set_title(title,fontsize=20,fontfamily='Serif')
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
    
def density_evolution(*args,labels,title,save=False):
    """Plot density for the given simnames"""
    fig=plt.figure(figsize=(8,8),dpi=150)
    ax=fig.add_subplot(1,1,1)
    df1, df2 = extract.extract_def(args[0])
    strain=np.linspace(0,1.718,df1.shape[0])
    for index, filename in enumerate(args):
        df1, df2 = extract.extract_def(filename)
        ax.plot(strain, df1.iloc[:,17],label=labels[index],linewidth=2)
    ax.legend(fontsize=16)
    ax = set_x_props(ax, 'Strain')
    ax = set_y_props(ax, 'Density')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
        'Parallel electric field',
        'Electric field = 1.2'))
    ax.text(0.30, 0.30, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
    ax.set_xlim(0,1.71)
    if save:
        plt.tight_layout()
        full_path = imagesavepath + 'density-' +title + '.png'
        plt.savefig(full_path,dpi=300)

def ngp_evolution(*args,labels,title,save=False):
    """Plot NGP for the given simnames"""
    fig=plt.figure(figsize=(8,8),dpi=150)
    ax=fig.add_subplot(1,1,1)
    df1, df2 = extract.extract_def(args[0])
    strain=np.linspace(0,1.718,df1.shape[0])
    for index, filename in enumerate(args):
        df1, df2 = extract.extract_def(filename)
        ax.plot(strain, df2.iloc[:,35],label=labels[index],linewidth=2)
    ax.legend(fontsize=16)
    ax = set_x_props(ax, 'Strain')
    ax = set_y_props(ax, 'NGP')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
        'Parallel electric field',
        'Electric field = 1.2'))
    ax.text(0.40, 0.30, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
    ax.set_xlim(0,1.71)
    if save:
        plt.tight_layout()
        full_path = imagesavepath + 'ngp-'+ title + '.png'
        plt.savefig(full_path,dpi=300)
    
def msd_evolution(*args,labels,title,save=False):
    """Plot MSD for the given simnames"""
    fig=plt.figure(figsize=(8,8),dpi=150)
    ax=fig.add_subplot(1,1,1)
    df1, df2 = extract.extract_def(args[0])
    strain=np.linspace(0,1.718,df1.shape[0])
    for index, filename in enumerate(args):
        df1, df2 = extract.extract_def(filename)
        ax.plot(strain, df2.iloc[:,32],label=labels[index],linewidth=2)
    ax.legend(fontsize=16)
    ax = set_x_props(ax, 'Strain')
    ax = set_y_props(ax, 'MSD')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
        'Parallel electric field',
        'Electric field = 1.2'))
    ax.text(0.40, 0.60, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
    ax.set_xlim(0,1.71)
    if save:
        plt.tight_layout()
        full_path = imagesavepath + 'msd-'+ title + '.png'
        plt.savefig(full_path,dpi=300)
    
def temperature_evolution(*args,labels,title,save=False):
    """Plot MSD for the given simnames"""
    fig=plt.figure(figsize=(8,8),dpi=150)
    ax=fig.add_subplot(1,1,1)
    df1, df2 = extract.extract_def(args[0])
    strain=np.linspace(0,1.718,df1.shape[0])
    for index, filename in enumerate(args):
        df1, df2 = extract.extract_def(filename)
        ax.plot(strain, df1.iloc[:,7],label=labels[index],linewidth=2)
    ax.legend(fontsize=16)
    ax = set_x_props(ax, 'Strain')
    ax = set_y_props(ax, 'Temperature')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
        'Parallel electric field',
        'Electric field = 1.2'))
    ax.text(0.40, 0.70, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
    ax.set_xlim(0,1.71)
    if save:
        plt.tight_layout()
        full_path = imagesavepath + 'temperature-'+ title + '.png'
        plt.savefig(full_path,dpi=300)

def energy_all_evolution(*args,labels,title,save=False):
    """Plot energy for the given simnames"""
    energy=[9,12,13,14]
    ene_label=['Bonded-energy','Ecoul','Van-der-Waal','Total']
    df1, df2 = extract.extract_def(args[0])
    strain=np.linspace(0,1.718,df1.shape[0])
    label_fname=''
    for element in labels:
        label_fname+=element
    for i in range(4):
        fig=plt.figure(figsize=(8,8),dpi=150)
        ax=fig.add_subplot(1,1,1)
        for index, filename in enumerate(args):
            df1, df2 = extract.extract_def(filename)
            ene_temp=(df1.iloc[:,energy[i]] - df1.iloc[0,energy[i]]).values
            ax.plot(strain, ene_temp,label=labels[index],linewidth=2)
        ax.legend(fontsize=16)
        ax.set_title(title,fontsize=20)
        ax = set_x_props(ax, 'Strain')
        ax = set_y_props(ax, ene_label[i])
        ax.set_xlim(0,1.71)
        if save:
            plt.tight_layout()
            full_path = imagesavepath + 'v20/energy_during_deformation/32C_128DP_energy_' +str(ene_label[i])+label_fname+ '.png'
            plt.savefig(full_path,dpi=300)
    