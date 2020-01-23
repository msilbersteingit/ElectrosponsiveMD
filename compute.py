import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate.quadrature import simps

from analib import fileIO

def find_pairs(coord,coords_loc=[3,6]):
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
        type2=current_timestep[current_timestep['type']==2].iloc[:,
            coords_loc[0]:coords_loc[1]].values 
            # change depending on columns in data 3 to 6 or 2 to 5
        type3=current_timestep[current_timestep['type']==3].iloc[:,
            coords_loc[0]:coords_loc[1]].values 
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

def radius_of_gyration(coords,coord_loc = [3,6]):
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
        coord_curr=coords[key].iloc[:,coord_loc[0]:coord_loc[1]].values
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

def radial_distribution_function(coordinates,box_l,simname,coords_loc=[3,6],
    nhis=200,save=False):
    """Compute the radial distribution function for given coordinates.

    Args:
    coordinates (array): Coordinates (x,y,z)
    nhis (int): Number of bins in histogram

    Returns:
    rdf ()
    """
    coords=coordinates.iloc[:,coords_loc[0]:coords_loc[1]].values #Convert 
        #to numpy array
    npart=np.size(coords,0) #Total number of particles
    """Initialize the histogram"""

    delg=box_l/(2*nhis) #Compute size of one bin
    g=[None]*nhis #Initialize g(r)
    for index in range(nhis):
        g[index]=0 #make every element zero. Can be skipped if used 0 instead
        # of None on Line 43.
    """Loop over pairs and determine the distribution of distances"""
    for partA in range(npart-1): #Don't loop over the last particle because 
        #we have two loop over the particles
        for partB in range(partA+1,npart): #Start from the next particle to 
            #avoid repetition of neighbor bins
            #Calculate the particle-particle distance
            dx = coords[partA][0] - coords[partB][0]
            dy = coords[partA][1] - coords[partB][1]
            dz = coords[partA][2] - coords[partB][2]
            distAB = [dx,dy,dz]
            r=np.linalg.norm(distAB) #Compute the magnitude of the distance
            if r<(box_l/2): #Check if distance is within cutoff (here half
                # of box length)
                ig=int(r/delg) #Check which bin the particle belongs to 
                g[ig]=g[ig]+2 #Add two particles to that bin's index 
                #(because it's a pair)
    """Normalize the radial distribution function"""
    rho=npart/(box_l**3) #Number density
    for index in range(nhis): 
        r=delg*(index+1)
        volume=4*np.pi*r*r*delg #Area betweeen bin i+1 and i
        g[index]=g[index]/npart #Divide your current count by the total
        # number
            # of particles in the system
        g[index]=g[index]/volume #Divide by the volume of the current bin
        g[index]=g[index]/rho #Divide by the number density of an ideal gas
    r=np.arange(0,box_l/2,delg) #Create a numpy array with delg as distance
    fwrite=simname + '_rdf.txt' #Filename for writing r and g(r) values
    f=open(fwrite,'w')
    f.write('r \t\t\t g(r)\n')
    for index in range(len(g)): #Write r and g(r) values
        f.write('%s\t%s\n'%(r[index],g[index]))
    f.close()
    return g, r

def structure_factor(coordinates, rdf_fname,box_l,simname,coords_loc=[3,6]):
    """
    Arguments:
    fname: Name of the file that contains the coordinates and the box lengths
    fname2: Name of the file that contains the r and g(r) data (computed 
    using rdf.py)

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
        for particle in range(1,len(r)): #Loop over all the particles 
            #and numerically integrate using Simpson's rule
            #One line has been broken into three terms for human-readable form
            term1=r[particle]**2
            term2num=np.sin(km[value]*r[particle])
            term2den=km[value]*r[particle]
            term3=g[particle]
            t1[particle]=((term1*term2num*term3)/(term2den))
        integral=simps(t1) #integrate
        sk[value]=1+ 4*np.pi*rho*integral #Assign to S(k) value
    fwrite=simname+'structure_factor.txt' #Write the values 
    f=open(fwrite,'w')
    f.write('k values \t\t S(k) values\n')
    for index in range(len(k)):
        f.write('%s\t%s\n'%(k[index],sk[index]))
    f.close()
    return k,sk

def chain_orientation_parameter(curr_coordinates,ex,nc,dp,coord_loc=[3,6]):
    """Compute the chain orientation parameter for one given timestep.
    
    coord (pandas dataframe): Current coordinates for ONE timestep. Do not 
    provide the coord output from extract.extract_unwrapped function.
    
    Returns:
    cop (float): Chain orientation parameter for the particular timestep 
    provided."""
    n_applicable_atoms = nc*dp - nc*2
    p2x = [0]*(n_applicable_atoms)
    orient = 0
    outer_index = 0
    for chain in range(1,nc+1):
        begin=(chain -1)*dp + 2
        end = chain*dp
        for index in range(begin,end,1):
            earlier_atom = curr_coordinates[curr_coordinates['id']
                == index - 1].values[:,coord_loc[0]:coord_loc[1]]
            #ref_atom = curr_coordinates[curr_coordinates['id'] == index].values[:,3:6]
            later_atom = curr_coordinates[curr_coordinates['id'] 
            == index + 1].values[:,coord_loc[0]:coord_loc[1]]
            ei = (later_atom - earlier_atom)/(np.linalg.norm(later_atom - earlier_atom))
            p2x[outer_index] = 1.5*((np.dot(ei,ex))**2) - 0.5
            outer_index+=1
    return np.sum(p2x)/n_applicable_atoms

def chain_entanglement_parameter(coord,nc,dp,coord_loc=[3,6]):
    """
    Find chain entanglement parameter based on bond data and coordinates of 
    all atoms.
    """
    ent_param = [0]*(len(coord.keys()))
    n_applicable_atoms=nc*dp - nc*20
    for i,key in enumerate(coord):
        entang=0
        entang = chain_entang_helper(coord[key],entang,nc,dp,coord_loc)
        ent_param[i]= entang/n_applicable_atoms
    return ent_param

def chain_entang_helper(curr_coordinates,entang,nc,dp,coord_loc):
    for key in range(1,nc+1):
        begin=(key -1)*dp + 11
        end = key*dp - 10 
        for index in range(begin,end,1):
            earlier_atom = curr_coordinates[curr_coordinates['id'] == index - 10].values[:,coord_loc[0]:coord_loc[1]]
            ref_atom = curr_coordinates[curr_coordinates['id'] == index].values[:,coord_loc[0]:coord_loc[1]]
            later_atom = curr_coordinates[curr_coordinates['id'] == index + 10 ].values[:,coord_loc[0]:coord_loc[1]]
            v1 = later_atom - ref_atom
            v1 = v1.reshape((3,))
            v2 = earlier_atom - ref_atom
            v2 = v2.reshape((3,))
            theta = np.arccos((np.dot(v1,v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)))
            if theta <1.570796:
                entang+=1
    return entang

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

