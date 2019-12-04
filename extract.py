import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analib import fileIO
from scipy.integrate.quadrature import simps


def extract_def(simname,syntax='18f'):
    """
    This function reads the given LAMMPS simulation name (if present somewhere
    in the current directory or subdirectories) and returns a numpy array 
    containing the file's output (log, lammpstrj, def1, def2).
    
    *def* files are usually created by Raiter during deformations. They are 
    clutter-free and contain only numbers. This function will NOT work for log
    files. For log files, please check extract_log_thermo function.
    
    Args:
    simname (str): name of the *def* file
    syntax (str): the format specification of the requested def file. It can
                  be 'old' or 'veryold'. Default is '18f'
    
    Returns:
    def1 (dataframe): Pandas dataframe containing the output from the *def1*
                      file.
    def2 (dataframe): Pandas dataframe containing output from the *def2* file.
    """

    dump_wrapped, dump_unwrapped, dump_def1,dump_def2,\
        dump_def3, log_file=fileIO.retrieve_different_filetypes(simname)
    path=fileIO.findInSubdirectory(dump_def1)
    if syntax=='18f':
        column_names=['strain','pxx','pyy','pzz','lx','ly','lz','temp','epair',
                'ebond','eangle','edihed','ecoul','evdwl','etotal','pe','ke',
                'density']
    elif syntax=='old':
        column_names=[''] #FILL IN LATER
    def1=pd.read_csv(path,delim_whitespace=True,
                     skiprows=1,dtype=np.float64,
                     names=column_names,
                     index_col=False) #Read and skip the first line
    path=fileIO.findInSubdirectory(dump_def2) 
    if syntax=='18f':
        column_names=["strain","c_meansquarep[1]","c_meansquarep[2]",
                      "c_meansquarep[3]","c_meansquarep[4]","c_meansquaren[1]",
                      "c_meansquaren[2]","c_meansquaren[3]","c_meansquaren[4]",
                      "c_meansquarei[1]","c_meansquarei[2]","c_meansquarei[3]",
                      "c_meansquarei[4]","c_meansquareu[1]","c_meansquareu[2]",
                      "c_meansquareu[3]","c_meansquareu[4]","c_nongaussp[1]",
                      "c_nongaussp[2]","c_nongaussp[3]","c_nongaussn[1]",
                      "c_nongaussn[2]","c_nongaussn[3]","c_nongaussi[1]",
                      "c_nongaussi[2]","c_nongaussi[3]","c_nongaussu[1]",
                      "c_nongaussu[2]","c_nongaussu[3]","c_msdall[1]",
                      "c_msdall[2]","c_msdall[3]","c_msdall[4]","c_ngpall[1]",
                      "c_ngpall[2]","c_ngpall[3]"]
    def2=pd.read_csv(path,delim_whitespace=True,
                     skiprows=1,dtype=np.float64,
                     names=column_names,index_col=False)
    return def1,def2


def extract_log_thermo(simname):
    """Read and return the thermodynamic data from a LAMMMPS simulation file.
    Args:
    simname (str): name of the simulation.
    
    Returns:
    log_thermo (dict): Dictionary containing the number of runs in the current
                       file and corresponding pandas dataframe containing all 
                       thermodynamic output regarding that run file.
    """
    dump_wrapped, dump_unwrapped, dump_def1, dump_def2,\
        dump_def3, log_file=fileIO.retrieve_different_filetypes(simname)
    path=fileIO.findInSubdirectory(log_file)
    df3=pd.read_csv(path,sep='nevergonnahappen',
                    engine='python',index_col=False,
                    header=None)
    natoms=[]
    log_thermo={}
    run_num=0
    for index in df3.index:
        line=df3.iloc[index].str.split()
        try:
            if line[0][0]=='Per' and line[0][1]=='MPI':
                try:
                    float(df3.iloc[index+3].str.split()[0][0]) 
                        #To make sure we do not include "run 0" runs. 
                    col_names=df3.iloc[index+1].str.split()[0]
                    run_num+=1
                    #%%%%%%%%%%%%% FIND RUN and THERMO numbers %%%%%%%%%%%%%%%
                    run_index=index-1
                    while df3.iloc[run_index].str.split()[0][0]!='run':
                        run_index-=1
                    run = df3.iloc[run_index].str.split()[0][1]
                    thermo_index=run_index - 1
                    while df3.iloc[thermo_index].str.split()[0][0]!='thermo':
                        thermo_index-=1
                    thermo = df3.iloc[thermo_index].str.split()[0][1]
                    output_num=int(int(run)/int(thermo))
                    natoms.append(
                        df3.iloc[index+output_num+3].str.split()[0][-2])
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    #%%%%%%%%%%%%%%%% Read thermo data %%%%%%%%%%%%%%%%%%%%%%%
                    df3.iloc[index+2:index+output_num+3].to_csv(r'./test.csv',
                                                                index=False,
                                                                header=None) 
                                                                #write dummy 
                                                                # file
                    df2=pd.read_csv('./test.csv',delim_whitespace=True,
                                    names=col_names,dtype=np.float64) 
                                    #read dummy file
                    key='run_' + str(run_num) 
                        #save in the corresponding dictionary.
                    log_thermo[key]=df2
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                except ValueError:
                    pass
        except IndexError:
            pass
    return log_thermo

def extract_unwrapped(simname,format_spec =['id','mol','type','xu','yu','zu']  ):
    """Extract coordinates of all atoms from unwrapped trajectory files with 
    format 'id','mol','type','xu','yu','zu'
    Args:
    simname (str): name of the LAMMPS simulation.

    Returns:
    coord (dict): Dictionary with the number of timesteps as keys and 
                  coordinates of all atoms at the corresponding timestep as a pandas
                  dataframe.
    """

    dump_wrapped, dump_unwrapped, dump_def1,dump_def2,\
        dump_def3, log_file=fileIO.retrieve_different_filetypes(simname)
    path=fileIO.findInSubdirectory(dump_unwrapped)
    unwrap=pd.read_csv(path,header=None,index_col=False)
    index=0
    natoms_notfound=True
    while natoms_notfound:
        try:
            if unwrap.iloc[index].str.split()[0][1]=='NUMBER':
                natoms_notfound=False
            else:
                index+=1
        except IndexError:
            index+=1
    natoms=int(unwrap.iloc[index+1].str.split()[0][0])
    index=0
    timestep=0
    coord={}
    while index<len(unwrap):
        line=unwrap.iloc[index].str.split()
        try:
            if line[0][0]=='ITEM:' and line[0][1]=='ATOMS':
                length=len(unwrap.iloc[index+1].str.split()[0])
                df2=unwrap.iloc[index+1:index+natoms+1]
                df2=df2[0].str.split(' ', length-1,expand=True) 
                    #Split based on separator - expensive
                num2str = lambda x : float(x) 
                    #convert all elements from str to float
                df2 = df2.applymap(num2str) 
                    #apply num2str to every element - expensive
                df2.columns=format_spec
                    #add corresponding column labels
                df2=df2.sort_values(by=['id']) 
                    #sort based on atom id so that future operations are easy.
                key='timestep_' + str(timestep) 
                    #save in the corresponding dictionary.
                coord[key]= df2
                index=index + natoms
                timestep+=1
            else:
                index+=1
        except IndexError:
            index+=1
    return coord

def extract_wrapped(simname,format_spec = ['id','type','xs','ys','zs']):
    """Extract coordinates of all atoms from wrapped trajectory files with 
    format 'id','type','xs','ys','zs'
    Args:
    simname (str): name of the LAMMPS simulation.

    Returns:
    coord (dict): Dictionary with the number of timesteps as keys and 
                  coordinates of all atoms at the corresponding timestep as a pandas
                  dataframe.
    """
    dump_wrapped, dump_unwrapped, dump_def1,dump_def2,\
        dump_def3, log_file=fileIO.retrieve_different_filetypes(simname)
    path=fileIO.findInSubdirectory(dump_wrapped)
    unwrap=pd.read_csv(path,header=None,index_col=False)
    index=0
    natoms_notfound=True
    while natoms_notfound:
        try:
            if unwrap.iloc[index].str.split()[0][1]=='NUMBER':
                natoms_notfound=False
            else:
                index+=1
        except IndexError:
            index+=1
    natoms=int(unwrap.iloc[index+1].str.split()[0][0])
    natoms=int(unwrap.iloc[index+1].str.split()[0][0])
    index=0
    timestep=0
    coord={}
    while index<len(unwrap):
        line=unwrap.iloc[index].str.split()
        try:
            if line[0][0]=='ITEM:' and line[0][1]=='ATOMS':
                length=len(unwrap.iloc[index+1].str.split()[0])
                df2=unwrap.iloc[index+1:index+natoms+1]
                df2=df2[0].str.split(' ', length-1,expand=True) 
                    #Split based on separator - expensive
                num2str = lambda x : float(x) 
                    #convert all elements from str to float
                df2 = df2.applymap(num2str) 
                    #apply num2str to every element - expensive
                df2.columns=format_spec
                    #add corresponding column labels
                df2=df2.sort_values(by=['id']) 
                    #sort based on atom id so that future operations are easy.
                key='timestep_' + str(timestep) 
                    #save in the corresponding dictionary.
                coord[key]= df2
                index=index + natoms
                timestep+=1
            else:
                index+=1
        except IndexError:
            index+=1
    return coord

def read_boxsize(simname):
    """Reads the box size of the simulation from the unwrapped coordinates 
    trajectory.

    Args:
    simname (str): name of the LAMMPS simulation.
    
    Returns:
    prop_constant (np array): an array  containing the proportionality 
                              constant pertaining to the increase in size of
                              the box (along x, y and z) at every timestep.
    """
    dump_wrapped, dump_unwrapped, dump_def1, dump_def2,\
        dump_def3, log_file=fileIO.retrieve_different_filetypes(simname)
    path=fileIO.findInSubdirectory(dump_unwrapped)
    unwrap=pd.read_csv(path,header=None,index_col=False)
    index=0
    ################## READ LAST TIMESTEP ####################################
    last_timestep=173 #here for now. will change later.
    ##########################################################################
    timestep=1
    prev_index=-10004
    ref_size_x = (float(unwrap.iloc[prev_index + 10009].str.split()[0][1]) 
                  - float(unwrap.iloc[prev_index + 10009].str.split()[0][0]))
    ref_size_y = (float(unwrap.iloc[prev_index + 10010].str.split()[0][1]) 
                  - float(unwrap.iloc[prev_index + 10010].str.split()[0][0]))
    ref_size_z = (float(unwrap.iloc[prev_index + 10011].str.split()[0][1]) 
                  - float(unwrap.iloc[prev_index + 10011].str.split()[0][0]))
    prop_constant=[None]*last_timestep
    while timestep<last_timestep:
        curr_size_x = (float(unwrap.iloc[prev_index + 10009].str.split()[0][1])
                       - float(unwrap.iloc[prev_index + 10009].str.split()[0][0]))
        curr_size_y = (float(unwrap.iloc[prev_index + 10010].str.split()[0][1]) 
                       - float(unwrap.iloc[prev_index + 10010].str.split()[0][0]))
        curr_size_z = (float(unwrap.iloc[prev_index + 10011].str.split()[0][1]) 
                       - float(unwrap.iloc[prev_index + 10011].str.split()[0][0]))
        px=curr_size_x/ref_size_x
        py=curr_size_y/ref_size_y
        pz=curr_size_z/ref_size_z
        prop_constant[timestep-1]=[px,py,pz]
        prev_index+=10009
        timestep+=1
    return prop_constant

def find_pairs(coord):
    """Reads in a dictionary which contains timesteps as keys and atom 
    coordinates as values and returns a dictionary with timesteps as keys
    and the distances between positive and negative beads as values at that
    timestep.

    Args:
    coords (dict):  dictionary with the number of timesteps as keys and 
        coordinates of all atoms at the corresponding timestep as a pandas 
        dataframe.
    
    Returns:
    distances (dict): A dictionary with timesteps as keys and the distances
        between positive and negative beads (magnitude) as values at that timestep
        as a pandas dataframe. 
    distances_vec (dict): A dictionary with timesteps as keys and the distances
        between positive and negative beads (vector) as values at that timestep as 
        a pandas dataframe. 
    """
    distances={}
    dist_vec={}
    for key in coord:
        current_timestep=coord[key]
        type2=current_timestep[current_timestep['type']==2].iloc[:,3:6].values 
            # change depending on columns in data 3 to 6 or 2 to 5
        type3=current_timestep[current_timestep['type']==3].iloc[:,3:6].values 
            # change depending on columns in data 3 to 6 or 2 to 5
        current_dist_vec=[(type2[index_2] - type3[index_3]) 
                        for index_3 in range(type3.shape[0]) 
                        for index_2 in range(type2.shape[0])]
        current_distance=[np.linalg.norm(type2[index_2] - type3[index_3]) 
                        for index_3 in range(type3.shape[0]) 
                        for index_2 in range(type2.shape[0])]
        dist_vec[key] = np.asarray(current_dist_vec)
        distances[key] = np.asarray(current_distance)
    return distances,dist_vec

def radius_of_gyration(coords,coords_loc = [3,6]):
    """Compute the radius of gyration for each timestep stored in the distance
    dictionary
    
    Args:
    coords (dict):  dictionary with the number of timesteps as keys and 
        coordinates of all atoms at the corresponding timestep as a pandas 
        dataframe.
    coord_loc (list): Column start number and column end number for x,y and z
        coordinates in the coords dictionary. Default is 3 to 6 (for unwrapped)
    
    Returns:
    rog (list): Python list with radius of gyration at each timestep of the
        simulation (number of timesteps correspond to number of timesteps 
        present in coords).
    """
    rog = [None]*len(coords)
    index=0
    for key in coords:
        coord_curr=coords[key].iloc[:,coords_loc[0]:coords_loc[1]].values
        mass=[14]*len(coord_curr)
        xm = [(m*i, m*j, m*k) for (i, j, k), m in zip(coord_curr, mass)]
        tmass = sum(mass)
        rr = sum(mi*i + mj*j + mk*k 
                for (i, j, k), (mi, mj, mk) in zip(coord_curr, xm))
        mm = sum((sum(i) / tmass)**2 for i in zip(*xm))
        rg = np.sqrt(rr / tmass-mm)
        rog[index] = round(rg, 3)
        index+=1
    return rog


def radial_distribution_function(coordinates,box_l,coords_loc=[3,6]):
    """Compute the radial distribution function for given coordinates.

    Args:
    coordinates (array): Coordinates (x,y,z)
    nhis (int): Number of bins in histogram

    Returns:
    rdf ()
    """
    coords=coordinates.iloc[:,coords_loc[0]:coords_loc[1]].values #Convert to numpy array
    npart=np.size(coords,0) #Total number of particles
    """Initialize the histogram"""
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #CHANGE NUMBER OF BINS FROM HERE
    nhis=200 #Number of bins
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    delg=box_l/(2*nhis) #Compute size of one bin
    g=[None]*nhis #Initialize g(r)
    for index in range(nhis):
        g[index]=0 #make every element zero. Can be skipped if used 0 instead of None on Line 43.
    """Loop over pairs and determine the distribution of distances"""
    for partA in range(npart-1): #Don't loop over the last particle because we have two loop over the particles
        for partB in range(partA+1,npart): #Start from the next particle to avoid repetition of neighbor bins
            #Calculate the particle-particle distance
            dx = coords[partA][0] - coords[partB][0]
            dy = coords[partA][1] - coords[partB][1]
            dz = coords[partA][2] - coords[partB][2]
            distAB = [dx,dy,dz]
            r=np.linalg.norm(distAB) #Compute the magnitude of the distance
            if r<(box_l/2): #Check if distance is within cutoff (here half of box length)
                ig=int(r/delg) #Check which bin the particle belongs to 
                g[ig]=g[ig]+2 #Add two particles to that bin's index (because it's a pair)
    """Normalize the radial distribution function"""
    rho=npart/(box_l**3) #Number density
    for index in range(nhis): 
        r=delg*(index+1)
        area=2*np.pi*r*delg #Area betweeen bin i+1 and i
        g[index]=g[index]/npart #Divide your current count by the total number of particles in the system
        g[index]=g[index]/area #Divide by the area of the current bin size
        g[index]=g[index]/rho #Divide by the number density of an ideal gas
    r=np.arange(0,box_l/2,delg) #Create a numpy array with delg as distance
    """Plotting the radial distribution function"""
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #UNCOMMENT the following lines to view the rdf plot in python
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    plt.plot(r,g) #Plot rdf and set chart properties
    plt.xlabel('$r$')
    plt.ylabel('$g(r)$')
    plt.savefig('18k_divide_by_3_rdf.png',dpi=300)
    # plt.show()
    fwrite='18k_divide_by_3_rdf.txt' #Filename for writing r and g(r) values
    f=open(fwrite,'w')
    f.write('r \t\t\t g(r)\n')
    for index in range(len(g)): #Write r and g(r) values
        f.write('%s\t%s\n'%(r[index],g[index]))
    f.close()

def structure_factor(coordinates, rdf_fname,box_l,coords_loc=[3,6]):
    """
    Arguments:
    fname: Name of the file that contains the coordinates and the box lengths
    fname2: Name of the file that contains the r and g(r) data (computed using rdf.py)
    Variables:
    npart : number of particles in the system
    box_l : box length (assumed that box length along x and y is same)
    coords : numpy array containing the particle coordinates (x,y,z coordinates)
    r : r values for g(r)
    g(r) : radial distribution function values
    k : vector values
    km : magnitude of k vector 
    rho : number density of the particles
    """
    coords=coordinates.iloc[:,coords_loc[0]:coords_loc[1]].values
    npart=np.size(coords,0) #Find number of particles
    rdf = pd.read_csv(rdf_fname,sep='\t',header=None,skiprows=[0])
    r = rdf.iloc[:,0].values
    g = rdf.iloc[:,1].values
    """Create list of k values"""
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #SET DELTA K AND KMAX OVER HERE
    delk=(2*np.pi)/box_l #Set delta k
    kmax=int(8/delk) #Set maximum value of k around 10-11
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    kx=[]
    ky=[]
    kz=[]
    k=[]
    sk=[0]*kmax
    km=[0]*kmax
    for index in range(kmax): #Loop over kmax to compute k values
        kx=delk*index
        ky=delk*index
        kz=delk*index
        m=[kx,ky,kz]
        k.append(np.linalg.norm(m))
        km[index]=(np.sqrt(kx**2 + ky**2 + kz**2))
    """Compute integrand using g(r)"""
    rho=npart/(box_l**3) #Number density
    for value in range(1,kmax):
        t1=[0]*len(r) #Initialize array for term 1
        for particle in range(1,len(r)): #Loop over all the particles and numerically integrate using Simpson's rule
            #One line has been broken into three terms for human-readable form
            term1=r[particle]**2
            term2num=np.sin(km[value]*r[particle])
            term2den=km[value]*r[particle]
            term3=g[particle]
            t1[particle]=((term1*term2num*term3)/(term2den))
        integral=simps(t1) #integrate
        sk[value]=1+ 4*np.pi*rho*integral #Assign to S(k) value
    fwrite='18k_sk.txt' #Write the values 
    f=open(fwrite,'w')
    f.write('k values \t\t S(k) values\n')
    for index in range(len(k)):
        f.write('%s\t%s\n'%(k[index],sk[index]))
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #UNCOMMENT the following lines to view the S(k) plot in python
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    plt.plot(k,sk)
    plt.xlabel('$k$')
    plt.ylabel('$S(k)$')
    # plt.show()

def chain_orientation_parameter(coord,ex):
    orie_param=np.zeros(len(coord.keys())+ 1)
    n_applicable_atoms = 9979
    p2x = [0]*(n_applicable_atoms+1)
    for i,key in enumerate(coord):
        orient = 0
        outer_index = 0
        for chain in range(1,11):
            begin=(chain -1)*1000 + 2
            end = chain*1000
            curr_coordinates=coord[key]
            for index in range(begin,end,1):
                earlier_atom = curr_coordinates[curr_coordinates['id'] == index - 1].values[:,3:6]
                #ref_atom = curr_coordinates[curr_coordinates['id'] == index].values[:,3:6]
                later_atom = curr_coordinates[curr_coordinates['id'] == index + 1].values[:,3:6]
                ei = (later_atom - earlier_atom)/(np.linalg.norm(later_atom - earlier_atom))
                p2x[outer_index] = 1.5*((np.dot(ei,ex))**2) - 0.5
                outer_index+=1
    return np.sum(p2x)/n_applicable_atoms

def order_parameter_p2(coord):
    orie_param=np.zeros(len(coord.keys())+ 1)
    n_applicable_atoms = 9979
    p2x = [0]*(n_applicable_atoms*10000)
    for i,key in enumerate(coord):
        orient = 0
        outer_index = 0
        for chain in range(1,11):
            begin=(chain -1)*1000 + 2
            end = chain*1000
            curr_coordinates=coord[key]
            for index in range(begin,end-1,1):
                earlier_atom = curr_coordinates[curr_coordinates['id'] == index - 1].values[:,3:6]    
                later_atom = curr_coordinates[curr_coordinates['id'] == index + 1].values[:,3:6]
                ej = (later_atom - earlier_atom)/(np.linalg.norm(later_atom - earlier_atom))
                for index_inner in range(index+1,end,1):
                    earlier_atom = curr_coordinates[curr_coordinates['id'] == index_inner - 1].values[:,3:6]
                    #ref_atom = curr_coordinates[curr_coordinates['id'] == index].values[:,3:6]
                    later_atom = curr_coordinates[curr_coordinates['id'] == index_inner + 1].values[:,3:6]
                    ei = (later_atom - earlier_atom)/(np.linalg.norm(later_atom - earlier_atom))
                    p2x[outer_index] = 1.5*((np.dot(ei.T,ej))**2) - 0.5
                    outer_index+=1
    return np.sum(p2x)/outer_index

