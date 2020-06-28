import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate.quadrature import simps
from scipy import stats
import random

from analib import fileIO
from analib import extract

def periodic_distance(x,y,z,x2,y2,z2,sidehx,sidehy,sidehz,bs):
    """Computes the periodic distance between given two beads.

    Args:
    x, y, z, x2, y2 z2 (float - all): Coordinates of the beads
    under consideration
    sidehx, sidehy, sidehz (float - all): half of size of the box 
    along the three cartesian coordinates.
    bs (list): size of the box along the three cartesian
    coordinates. 

    Returns:
    dist_mag (float): distance between the two beads.
    """
    dx = x2 - x
    dy = y2 - y
    dz = z2 - z
    if (dx < -sidehx):   dx = dx + bs[0]
    if (dx > sidehx):    dx = dx - bs[0]
    if (dy < -sidehy):   dy = dy + bs[1]
    if (dy > sidehy):    dy = dy - bs[1]
    if (dz < -sidehz):   dz = dz + bs[2]
    if (dz > sidehz):    dz = dz - bs[2]
    distAB = [dx,dy,dz]
    dist_mag = np.linalg.norm(distAB)
    return dist_mag
    
def cluster_analysis(simname,cutoff,coords_loc=[3,6]):
    """Read a LAMMPS trajectory (extract_unwrapped) based on the given filename 
    (first timestep/snapshot only) and return number of clusters and size
    of clusters at the first snapshot. Very slow -- might take couple of hours
    depending on size of the system.
    Args:
    simname(str): Name of the simulation file
    cutoff (float): Minimum distance between a member of a cluster and at least
    one another member of the cluster.

    Returns:
    cluster (dict): A dictionary that contains clusters as keys and list of 
    atomIDs in that specific cluster as values.
    """
    coord, bs=coord,bs=extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    current_timestep = coord['timestep_0']
    cluster_set = {}
    cluster_index=0
    type2=current_timestep[current_timestep['type']==2].values 
    type3=current_timestep[current_timestep['type']==3].values 
    sidehx = bs[0]/2
    sidehy = bs[1]/2
    sidehz = bs[2]/2
    for pindex, pcharge in enumerate(type2):
        for nindex, ncharge in enumerate(type3):
            distsn = periodic_distance(pcharge[3],pcharge[4],pcharge[5],
                              ncharge[3],ncharge[4],ncharge[5],
                              sidehx,sidehy,sidehz,bs)
            if distsn<cutoff:
                cluster_not_found=True
                for cluster in cluster_set:
                    if cluster_not_found:
                        if pcharge[0] in cluster_set[cluster]:
                            if ncharge[0] in cluster_set[cluster]:
                                pass
                            else:
                                cluster_set[cluster].append(ncharge[0])
                            cluster_not_found=False
                        elif ncharge[0] in cluster_set[cluster]:
                            if pcharge[0] in cluster_set[cluster]:
                                pass
                            else:
                                cluster_set[cluster].append(pcharge[0])
                            cluster_not_found=False
                        else:
                            index=0
                            while index<len(cluster_set[cluster]) and cluster_not_found:
                                current_atom = current_timestep[current_timestep['id']==cluster_set[cluster][index]].iloc[:,:].values[0]
                                dists2 = periodic_distance(pcharge[3],pcharge[4],pcharge[5],
                                                        current_atom[3],current_atom[4],current_atom[5],
                                                        sidehx,sidehy,sidehz,bs)
                                if dists2<cutoff:
                                    cluster_set[cluster].append(pcharge[0])
                                    if ncharge[0] in cluster_set[cluster]:
                                        print('repeat dists2')
                                    cluster_set[cluster].append(ncharge[0])
                                    cluster_not_found=False
                                if not cluster_not_found:
                                    dists3 = periodic_distance(ncharge[3],ncharge[4],ncharge[5],
                                                        current_atom[3],current_atom[4],current_atom[5],
                                                        sidehx,sidehy,sidehz,bs)
                                    if dists3<cutoff:
                                        cluster_set[cluster].append(pcharge[0])
                                        if ncharge[0] in cluster_set[cluster]:
                                            print('repeat dists3',ncharge[0])
                                        else:
                                            cluster_set[cluster].append(ncharge[0])
                                        cluster_not_found=False
                                index+=1
                if cluster_not_found:
                    cluster_set[cluster_index]=[pcharge[0]]
                    if ncharge[0] in cluster_set[cluster_index]:
                        print('repeat cnf')
                    cluster_set[cluster_index].append(ncharge[0])
                    cluster_index+=1
    return cluster_set

def cluster_analysis_array(simname,cutoff,coords_loc=[3,6]):
    """Read a LAMMPS trajectory (extract_unwrapped) based on the given filename 
    (first timestep/snapshot only) and return number of clusters and size
    of clusters at the first snapshot. Relatively fast (compared to 
    cluster_analysis). Instead of loops, random indices are chosen to find and merge
    clusters. Method increases the speed significantly -- errors are possible but 
    not likely.
    
    Args:
    simname(str): Name of the simulation file
    cutoff (float): Minimum distance between a member of a cluster and at least
    one another member of the cluster.

    Returns:
    cluster (dict): A dictionary that contains clusters as keys and list of 
    atomIDs in that specific cluster as values. 
    """
    coord, bs=coord,bs=extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    current_timestep = coord['timestep_0']
    del coord
    cluster_set = {}
    cluster_index=0
    type2=current_timestep[current_timestep['type']==2].values 
    type3=current_timestep[current_timestep['type']==3].values 
    sidehx = bs[0]/2
    sidehy = bs[1]/2
    sidehz = bs[2]/2
    charge_distances=np.zeros((type2.shape[0],type3.shape[0]))
    map_patomids={}
    map_natomids={}
    for pindex, pcharge in enumerate(type2):
        map_patomids[pindex] = pcharge[0]
        for nindex, ncharge in enumerate(type3):
            map_natomids[nindex] = ncharge[0]
            charge_distances[pindex,nindex] = periodic_distance(pcharge[3],pcharge[4],pcharge[5],
                              ncharge[3],ncharge[4],ncharge[5],
                              sidehx,sidehy,sidehz,bs)
    index_distances={}
    cluster={}
    for index,row in enumerate(charge_distances):
        index_distances[index] = np.where(row<cutoff)
    
    for key in index_distances:
        cluster[key]=[]
        for element in index_distances[key][0]:
            cluster[key].append(map_patomids[key])
            cluster[key].append(map_natomids[element])
            for inner_key in index_distances:
                for inner_element in index_distances[inner_key][0]:
                    if element==inner_element and key!=inner_key:
                        cluster[key].append(map_patomids[inner_key])
                        cluster[key].append(map_natomids[inner_element])
    
    for key in cluster:
        cluster[key]=np.unique(cluster[key])
    # for key in cluster:
    #     #cluster[key]=np.unique(cluster[key])
    #     for inner_key in cluster:
    #         cluster[key]=np.unique(cluster[key])
    #         ln=min(len(cluster[key]),len(cluster[inner_key]))
    #         if np.any(cluster[key][:ln]==cluster[inner_key][:ln]) and key!=inner_key:
    #             cluster[key]=np.append(cluster[key],cluster[inner_key])
    random_cals=0
    cluster_keys_set=list(range(0,len(cluster)))
    not_steady_set=True
    while not_steady_set:
        dict_key=random.choice(cluster_keys_set)
        try:
            element_index=random.randrange(0,len(cluster[dict_key]))
        except ValueError:
            pass
        dict_key2=random.choice(cluster_keys_set)
        try:
            element_index2=random.randrange(0,len(cluster[dict_key2]))
        except:
            pass
        try:
            if cluster[dict_key][element_index]==cluster[dict_key2][element_index2] and dict_key!=dict_key2:
                cluster[dict_key]=np.append(cluster[dict_key],cluster[dict_key2])
                del cluster[dict_key2]
                cluster_keys_set.remove(dict_key2)
                last_change=random_cals
            random_cals+=1
            try:
                if random_cals - last_change > 15000000:
                    print('last change at ',last_change,'total rcals',random_cals)
                    not_steady_set=False
            except:
                pass
        except:
            pass
    for key in cluster:
        cluster[key]=np.unique(cluster[key])
    cluster_copy=cluster.copy()
    mean_size=0
    
    # for index in range(len(cluster)):
    #     for inner_index in range(index+1,len(cluster)):
    #         if cluster[index][0]==cluster[inner_index][0] and index!=inner_index:
    #             try:
    #                 print('del')
    #                 del cluster_copy[inner_index]
    #             except:
    #                 pass
    
    for key in cluster_copy:
        mean_size+=len(cluster_copy[key])
    print(mean_size/len(cluster_copy))
    return cluster_copy

def find_pairs(simname,coords_loc=[3,6]):
    """Reads a filename and  returns a dictionary with timesteps as keys
    and the distances between positive and negative beads as values at that
    timestep. Uniqueness in pairs is not maintained.

    Args:
    simname (str): filename of the simulation
    
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
    coord=extract.extract_unwrapped(simname)
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

def find_pairs_unique(simname,coords_loc=[3,6]):
    matched_pair={}
    coord,bs=extract.extract_unwrapped(simname,first_only=True,boxsize_whole=True)
    for key in coord:
        current_timestep=coord[key]
        type2=current_timestep[current_timestep['type']==2].iloc[:,
            coords_loc[0]:coords_loc[1]].values 
        type3=current_timestep[current_timestep['type']==3].iloc[:,
            coords_loc[0]:coords_loc[1]].values 
        distsn=np.zeros((type3.shape[0],1))
        distsp=np.zeros((type2.shape[0],1))
        matched_pair[key]=np.zeros((type2.shape[0],3))
        repeat_checkn = []
        nrepeat=0
        matched=0
        unmatched=0
        box_l=bs[key]
        sidehx=box_l[0]/2
        sidehy=box_l[1]/2
        sidehz=box_l[2]/2
        for pindex,pcharge in enumerate(type2):
            for nindex,ncharge in enumerate(type3):
                dx = pcharge[0] - ncharge[0]
                dy = pcharge[1] - ncharge[1]
                dz = pcharge[2] - ncharge[2]
                if (dx < -sidehx):   dx = dx + box_l[0]
                if (dx > sidehx):    dx = dx - box_l[0]
                if (dy < -sidehy):   dy = dy + box_l[1]
                if (dy > sidehy):    dy = dy - box_l[1]
                if (dz < -sidehz):   dz = dz + box_l[2]
                if (dz > sidehz):    dz = dz - box_l[2]
                distAB = [dx,dy,dz]
                distsn[nindex] = np.linalg.norm(distAB)
            closest_ncharge_id = np.argmin(distsn)
            for positiveindex,positivecharge in enumerate(type2):
                dx = positivecharge[0] - type3[closest_ncharge_id][0]
                dy = positivecharge[1] - type3[closest_ncharge_id][1]
                dz = positivecharge[2] - type3[closest_ncharge_id][2]
                if (dx < -sidehx):   dx = dx + box_l[0]
                if (dx > sidehx):    dx = dx - box_l[0]
                if (dy < -sidehy):   dy = dy + box_l[1]
                if (dy > sidehy):    dy = dy - box_l[1]
                if (dz < -sidehz):   dz = dz + box_l[2]
                if (dz > sidehz):    dz = dz - box_l[2]
                distAB = [dx,dy,dz]
                distsp[positiveindex] = np.linalg.norm(distAB)
            closest_pcharge_id = np.argmin(distsp)
            if closest_pcharge_id == pindex:
                if closest_ncharge_id in repeat_checkn:
                    nrepeat=+1
                    unmatched=+1
                else:
                    repeat_checkn.append(closest_ncharge_id)
                    matched+=1
                    dx = pcharge[0] - type3[closest_ncharge_id][0]
                    dy = pcharge[1] - type3[closest_ncharge_id][1]
                    dz = pcharge[2] - type3[closest_ncharge_id][2]
                    if (dx < -sidehx):   dx = dx + box_l[0]
                    if (dx > sidehx):    dx = dx - box_l[0]
                    if (dy < -sidehy):   dy = dy + box_l[1]
                    if (dy > sidehy):    dy = dy - box_l[1]
                    if (dz < -sidehz):   dz = dz + box_l[2]
                    if (dz > sidehz):    dz = dz - box_l[2]
                    distAB = [dx,dy,dz]
                    positivechargeid=current_timestep[current_timestep['xu']==pcharge[0]]
                    positivechargeid=positivechargeid[positivechargeid['yu']==pcharge[1]]
                    positivechargeid=positivechargeid[positivechargeid['zu']==pcharge[2]] 
                    negativechargeid=current_timestep[current_timestep['xu']==type3[closest_ncharge_id][0]] 
                    negativechargeid=negativechargeid[negativechargeid['yu']==type3[closest_ncharge_id][1]]
                    negativechargeid=negativechargeid[negativechargeid['zu']==type3[closest_ncharge_id][2]] 
                    matched_pair[key][pindex] = [int(positivechargeid['id']), int(negativechargeid['id']), np.linalg.norm(distAB)]
            else:
                unmatched+=1
        matched_pair[key] =matched_pair[key][~np.all(matched_pair[key] == 0, axis=1)]
    print((matched/(matched+unmatched))*100)
    return matched_pair
    
def find_decorrelation_pairs(simname,cutoff,coords_loc=[3,6]):
    coord,bs=extract.extract_unwrapped(simname,boxsize_whole=True)
    correlation_fn_whole=np.zeros((len(coord.keys()),1))
    outer_index = 0
    for key in coord:
        current_timestep=coord[key]
        type2=current_timestep[current_timestep['type']==2].iloc[:,
            coords_loc[0]:coords_loc[1]].values 
        type3=current_timestep[current_timestep['type']==3].iloc[:,
            coords_loc[0]:coords_loc[1]].values 
        distsn=np.zeros((type3.shape[0],1))
        correlation_fn=np.zeros((type3.shape[0],1))
        distsp=np.zeros((type2.shape[0],1))
        box_l=bs[key]
        sidehx=box_l[0]/2
        sidehy=box_l[1]/2
        sidehz=box_l[2]/2
        for nindex,ncharge in enumerate(type3):
            for pindex,pcharge in enumerate(type2):
                dx = pcharge[0] - ncharge[0]
                dy = pcharge[1] - ncharge[1]
                dz = pcharge[2] - ncharge[2]
                if (dx < -sidehx):   dx = dx + box_l[0]
                if (dx > sidehx):    dx = dx - box_l[0]
                if (dy < -sidehy):   dy = dy + box_l[1]
                if (dy > sidehy):    dy = dy - box_l[1]
                if (dz < -sidehz):   dz = dz + box_l[2]
                if (dz > sidehz):    dz = dz - box_l[2]
                distAB = [dx,dy,dz]
                distsp[pindex] = np.linalg.norm(distAB)
            if np.asarray(np.where(distsp == 0)).shape[1]>0:
                print('something wrong')
            if np.asarray(np.where(distsp<cutoff)).shape[1] > 0:
                correlation_fn[nindex]=1
        correlation_fn_whole[outer_index]=np.sum(correlation_fn)
        outer_index+=1  
    return correlation_fn_whole

def find_decorrelation_pairs_distances(simname,coords_loc=[3,6]):
    coord,bs=extract.extract_unwrapped(simname,boxsize_whole=True)
    all_distances={}
    for key in coord:
        current_timestep=coord[key]
        type2=current_timestep[current_timestep['type']==2].iloc[:,
            coords_loc[0]:coords_loc[1]].values 
        type3=current_timestep[current_timestep['type']==3].iloc[:,
            coords_loc[0]:coords_loc[1]].values 
        distsn=np.zeros((type3.shape[0],1))
        correlation_fn=np.zeros((type3.shape[0],1))
        distsp=np.zeros((type2.shape[0],1))
        box_l=bs[key]
        sidehx=box_l[0]/2
        sidehy=box_l[1]/2
        sidehz=box_l[2]/2
        distance_from_this_ncharge={}
        for nindex,ncharge in enumerate(type3):
            for pindex,pcharge in enumerate(type2):
                dx = pcharge[0] - ncharge[0]
                dy = pcharge[1] - ncharge[1]
                dz = pcharge[2] - ncharge[2]
                if (dx < -sidehx):   dx = dx + box_l[0]
                if (dx > sidehx):    dx = dx - box_l[0]
                if (dy < -sidehy):   dy = dy + box_l[1]
                if (dy > sidehy):    dy = dy - box_l[1]
                if (dz < -sidehz):   dz = dz + box_l[2]
                if (dz > sidehz):    dz = dz - box_l[2]
                distAB = [dx,dy,dz]
                distsp[pindex] = np.linalg.norm(distAB)
            distance_from_this_ncharge[nindex] = distsp
        all_distances[key] = distance_from_this_ncharge
    return all_distances

def find_charged_pairs_center(coord,coords_loc=[3,6]):
    """Reads in a dictionary which contains timesteps as keys and atom 
    coordinates as values and returns a dictionary with timesteps as keys
    and positions of centers of positive and negative beads.

    Args:
    coords (dict):  dictionary with the number of timesteps as keys and 
        coordinates of all atoms at the corresponding timestep as a pandas 
        dataframe.
    
    Returns:
    c_pos (array): positions of centers of charged pairs
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
        distsn=np.zeros((type3.shape[0],1))
        distsp=np.zeros((type2.shape[0],1))
        charge_center=np.zeros((type2.shape[0],3))
        repeat_checkn = []
        nrepeat=0
        matched=0
        unmatched=0
        for pindex,pcharge  in enumerate(type2):
            for nindex,ncharge in enumerate(type3):
                distsn[nindex] = np.linalg.norm(pcharge - ncharge)
            closest_ncharge_id = np.argmin(distsn)
            for positiveindex,positivecharge in enumerate(type2):
                distsp[positiveindex] = np.linalg.norm(positivecharge - type3[closest_ncharge_id])
            closest_pcharge_id = np.argmin(distsp)
            if closest_pcharge_id == pindex:
                matched+=1
                charge_center[pindex] = (type3[closest_ncharge_id] + pcharge)/2
            else:
                check = np.min(distsn) - np.min(distsp)
                if check<0.5:
                    charge_center[pindex] = (type3[closest_ncharge_id] + pcharge)/2
                else:
                    unmatched+=1

            if closest_ncharge_id in repeat_checkn:
                nrepeat+=1
            else:
                repeat_checkn.append(closest_ncharge_id)
    charge_center = charge_center[~np.all(charge_center == 0, axis=1)]
    print(charge_center.shape)
    f=open('VMD-file.lammpstrj','w')
    f.write('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n%s\nITEM: BOX BOUNDS pp pp pp\n'%(charge_center.shape[0]))
    f.write('0.0000000000000000e+00 2.5000000000000000e+01\n0.0000000000000000e+00 2.5000000000000000e+01\n0.0000000000000000e+00 2.5000000000000000e+01\n')
    f.write('ITEM: ATOMS id xu yu zu \n')
    for index, pos in enumerate(charge_center):
        f.write('%s\t%s\t%s\t%s\n'%(index, pos[0],pos[1],pos[2]))
    f.close()
    return charge_center
    
def radius_of_gyration(coords,mass,coord_loc = [3,6]):
    """Compute the rooot mean square radius of gyration for each timestep stored in the distance
    dictionary (for each chain)
    
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
        rg=[]
        for chain in list(coords[key].mol.unique()):
            coord_curr=coords[key][coords[key]['mol']==chain].iloc[:,coord_loc[0]:coord_loc[1]].values            
            mass_list = [mass]*len(coord_curr)
            xm = [(m*i, m*j, m*k) for (i, j, k), m in zip(coord_curr, mass_list)]
            tmass = sum(mass_list)
            rr = sum(mi*i + mj*j + mk*k 
                    for (i, j, k), (mi, mj, mk) in zip(coord_curr, xm))
            mm = sum((sum(i) / tmass)**2 for i in zip(*xm))
            rg.append(np.sqrt(rr / tmass-mm))
        rog[index] = round(np.mean(rg),3)
        index+=1
    return rog

def radius_of_gyration_squared(coords,mass,coord_loc = [3,6]):
    """Compute the mean radius of gyration (squared) for each timestep stored in the distance
    dictionary (for each chain)
    
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
    rog=[None]*len(coords)
    index=0
    for key in coords:
        rg=[]
        for chain in list(coords[key].mol.unique()):
            coord_curr=coords[key][coords[key]['mol']==chain].iloc[:,coord_loc[0]:coord_loc[1]].values            
            com = np.mean(coord_curr,axis=0)
            dist_from_com=(np.linalg.norm(coord_curr - com))**2
            numer=mass*(np.sum(dist_from_com))
            denom=mass*len(coord_curr)
            rg.append(numer/denom)            
        rog[index] = round(np.mean(rg),3)
        index+=1
    return rog

def end_to_end_distance(coords,coords_loc=[3,6],sidechain=False):
    """Compute the average end to end distance of the system.
    Args:
    coords (dict):  dictionary with the number of timesteps as keys and 
        coordinates of all atoms at the corresponding timestep as a pandas 
        dataframe.
    coord_loc (list): Column start number and column end number for x,y and z
        coordinates in the coords dictionary. Default is 3 to 6 (for unwrapped)

    Returns:
    rend (list): Python list with with mean end to end distance at
     each timestep.
    """
    rend_list = [None]*len(coords)
    index=0
    for key in coords:
        rend=[]
        for chain in list(coords[key].mol.unique()):
            coord_curr=coords[key][coords[key]['mol']==chain]
            id_start = (len(coord_curr))*(chain-1) + 1
            if sidechain:
                id_end = (chain-1)*len(coord_curr) + 89
            else:
                id_end = (chain)*(len(coord_curr))
            r_start = coord_curr[coord_curr['id']==id_start].iloc[:,coords_loc[0]:coords_loc[1]].values
            r_end = coord_curr[coord_curr['id']==id_end].iloc[:,coords_loc[0]:coords_loc[1]].values
            curr_dist = np.linalg.norm(r_end-r_start)
            rend.append(curr_dist)
        rend_list[index] = round(np.mean(rend),3)
        index+=1
    return rend_list

def end_to_end_distance_2(coords,coords_loc=[3,6],sidechain=False):
    """Compute the average end to end distance of the system.
    Args:
    coords (dict):  dictionary with the number of timesteps as keys and 
        coordinates of all atoms at the corresponding timestep as a pandas 
        dataframe.
    coord_loc (list): Column start number and column end number for x,y and z
        coordinates in the coords dictionary. Default is 3 to 6 (for unwrapped)

    Returns:
    rend (list): Python list with with mean end to end distance at
     each timestep.
    """
    rend_list = [None]*len(coords)
    index=0
    index_chain=0
    for key in coords:
        rend=[]
        for chain in list(coords[key].mol.unique()):
            coord_curr=coords[key][coords[key]['mol']==chain]
            id_start = (len(coord_curr))*(chain-1) + 1
            if sidechain:
                id_end = (chain-1)*len(coord_curr) + 89
            else:
                id_end = (chain)*(len(coord_curr))
            r_start = coord_curr[coord_curr['id']==id_start].iloc[:,coords_loc[0]:coords_loc[1]].values
            r_end = coord_curr[coord_curr['id']==id_end].iloc[:,coords_loc[0]:coords_loc[1]].values
            print(coord_curr[coord_curr['id']==id_start].iloc[:,6:9].values[0],coord_curr[coord_curr['id']==id_end].iloc[:,6:9].values[0])
            if np.all(coord_curr[coord_curr['id']==id_start].iloc[:,6:9].values[0] == coord_curr[coord_curr['id']==id_end].iloc[:,6:9].values[0]):
                print('true')
                curr_dist = np.linalg.norm(r_end-r_start)
                rend.append(curr_dist)
                index_chain+=1
            else:
                print('false')
            
        rend_list[index] = round(np.mean(rend),3)
        index+=1
    print(index_chain)
    return rend_list

def end_to_end_distance_squared(coords,coords_loc=[3,6],sidechain=False):
    """Compute the average end to end distance of the system.
    Args:
    coords (dict):  dictionary with the number of timesteps as keys and 
        coordinates of all atoms at the corresponding timestep as a pandas 
        dataframe.
    coord_loc (list): Column start number and column end number for x,y and z
        coordinates in the coords dictionary. Default is 3 to 6 (for unwrapped)

    Returns:
    rend (list): Python list with with mean end to end distance at
     each timestep.
    """
    rend_list = [None]*len(coords)
    gauss_list={}
    index=0
    for key in coords:
        rend=[]
        gauss_list[key]=[]
        for chain in list(coords[key].mol.unique()):
            coord_curr=coords[key][coords[key]['mol']==chain]
            id_start = (len(coord_curr))*(chain-1) + 1
            if sidechain:
                id_end = (chain-1)*len(coord_curr) + 89
            else:
                id_end = (chain)*(len(coord_curr))
            r_start = coord_curr[coord_curr['id']==id_start].iloc[:,coords_loc[0]:coords_loc[1]].values
            r_end = coord_curr[coord_curr['id']==id_end].iloc[:,coords_loc[0]:coords_loc[1]].values
            curr_dist = np.linalg.norm(r_end-r_start)
            rend.append(curr_dist**2)
            gauss_list[key].append(curr_dist)
        rend_list[index] = round(np.mean(rend),3)
        index+=1
    return rend_list, gauss_list

def radial_distribution_function(coordinates,box_l,simname,coords_loc=[3,6],
    nhis=200,save=False):
    """Compute the radial distribution function for given coordinates.

    Args:
    coordinates (array): Coordinates (x,y,z)
    nhis (int): Number of bins in histogram

    Returns:
    rdf ()
    """
    coords=coordinates[coordinates['type']!=1].iloc[:,coords_loc[0]:coords_loc[1]].values #Convert 
        #to numpy array
    npart=np.size(coords,0) #Total number of particles
    """Initialize the histogram"""
    delg=box_l[0]/(2*nhis) #Compute size of one bin
    sidehx=box_l[0]/2
    sidehy=box_l[1]/2
    sidehz=box_l[2]/2
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
            if (dx < -sidehx):   dx = dx + box_l[0]
            if (dx > sidehx):    dx = dx - box_l[0]
            if (dy < -sidehy):   dy = dy + box_l[1]
            if (dy > sidehy):    dy = dy - box_l[1]
            if (dz < -sidehz):   dz = dz + box_l[2]
            if (dz > sidehz):    dz = dz - box_l[2]
            distAB = [dx,dy,dz]
            r=np.linalg.norm(distAB) #Compute the magnitude of the distance
            if r<(box_l[0]/2): #Check if distance is within cutoff (here half
                # of box length)
                ig=int(r/delg) #Check which bin the particle belongs to 
                g[ig]=g[ig]+2 #Add two particles to that bin's index 
                #(because it's a pair)
    """Normalize the radial distribution function"""
    rho=npart/(box_l[0]**3) #Number density
    for index in range(nhis): 
        r=delg*(index+1)
        #volume=4*np.pi*r*r*delg #Volume betweeen bin i+1 and i
        v1 = (4/3)*np.pi*(r**3)
        v2= (4/3)*np.pi*((r+delg)**3)
        volume=v2-v1
        g[index]=g[index]/npart #Divide your current count by the total
        # number
            # of particles in the system
        g[index]=g[index]/volume #Divide by the volume of the current bin
        g[index]=g[index]/rho #Divide by the number density of an ideal gas
    r=np.arange(0,box_l[0]/2,delg) #Create a numpy array with delg as distance
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
    coords : pandas array containing the particle coordinates (x,y,z coordinates)
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

def wrapper_chain_orientation_parameter(simname, nc, dp, 
                                        coord_loc =[3,6], dir='x',
                                        sc=False):
    """The chain_orientation_parameter function takes a pandas dataframe as
    input. To wrap the function higher, I have made this new function that takes
    in the name of the simulation and returns a chain orientation parameter
    list for the given simulation.

    Args:
    simname (str): name of the simulation
    ex (list): the unit vector for the direction
    nc (int): number of chains in the system
    dp (int): degree of polymerization of the system
    
    Returns:
    cop_list (list): list containing the chain orientation parameter for each
    timestep of the given simulation.
    """
    coord = extract.extract_unwrapped(simname)
    cop_list_x =[]
    cop_list_y =[]
    cop_list_z =[]
    if dir=='x':
        for key in coord:
            cop_list_x.append(chain_orientation_parameter(coord[key],[1,0,0],nc,dp,sidechain=sc))
        return cop_list_x
    if dir=='y':
        for key in coord:
            cop_list_y.append(chain_orientation_parameter(coord[key],[0,1,0],nc,dp,sidechain=sc))
        return cop_list_y
    if dir=='z':
        for key in coord:
            cop_list_z.append(chain_orientation_parameter(coord[key],[0,0,1],nc,dp,sidechain=sc))
        return cop_list_z

def chain_orientation_parameter(curr_coordinates,ex,nc,dp,sidechain=False,
                                coord_loc=[3,6]):
    """Compute the chain orientation parameter for one given timestep.
    
    coord (pandas dataframe): Current coordinates for ONE timestep. Do not 
    provide the coord output from extract.extract_unwrapped function.
    ex (list): the unit vector for the direction
    nc (int): number of chains in the system
    dp (int): degree of polymerization of the system
    
    Returns:
    cop (float): Chain orientation parameter for the particular timestep 
    provided."""
    n_applicable_atoms = nc*dp - nc*2
    p2x = [0]*(n_applicable_atoms)
    orient = 0
    outer_index = 0
    for chain in range(1,nc+1):
        begin=(chain -1)*dp + 2
        if sidechain:
            end=(chain-1)*dp + 89
        else:
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

def one_timestep_parameters(simname,nc,dp,coord_loc=[3,6],
sidechain=False):
    """Compute chain orientation parameter at one timestep"""
    coord = extract.extract_unwrapped(simname,first_only=True)
    x = chain_orientation_parameter(coord['timestep_0'],[1,0,0],nc,dp,sidechain=sidechain)
    y = chain_orientation_parameter(coord['timestep_0'],[0,1,0],nc,dp,sidechain=sidechain)
    z = chain_orientation_parameter(coord['timestep_0'],[0,0,1],nc,dp,sidechain=sidechain)
    cep = chain_entanglement_parameter(coord,nc,dp,coord_loc,sidechain=sidechain)
    print('x = %.3f \ny = %.3f \nz = %.3f'%(x,y,z))

def one_timestep_set_of_parameters(simname,nc,dp,mass,coord_loc=[3,6],
sidechain=False):
    """Compute chain orientation parameter at one timestep"""
    coord = extract.extract_unwrapped(simname,first_only=True)
    #x = chain_orientation_parameter(coord['timestep_0'],[1,0,0],nc,dp,sidechain=sidechain)
    #y = chain_orientation_parameter(coord['timestep_0'],[0,1,0],nc,dp,sidechain=sidechain)
    #z = chain_orientation_parameter(coord['timestep_0'],[0,0,1],nc,dp,sidechain=sidechain)
    cep = chain_entanglement_parameter(coord,nc,dp,coord_loc,sidechain=sidechain)
    rog = radius_of_gyration(coord, mass)
    rog_sq = radius_of_gyration_squared(coord, mass)
    rend = end_to_end_distance(coord,sidechain=sidechain)
    rend_sq,gauss = end_to_end_distance_squared(coord,sidechain=sidechain)
    ratio = rend_sq[0]/rog_sq[0]
    print('cep = %.3f \nms-rog = %.3f \nms-rend = %.3f \nratio = %.3f \nrog = %.3f \nrms = %.3f'%(cep[0],rog_sq[0],rend_sq[0],ratio,rog[0],rend[0]))

def chain_entanglement_parameter(coord,nc,dp,coord_loc=[3,6],sidechain=False):
    """
    Find chain entanglement parameter based on bond data and coordinates of 
    all atoms.
    """
    ent_param = [0]*(len(coord.keys()))
    n_applicable_atoms=nc*dp - nc*20
    for i,key in enumerate(coord):
        entang=0
        entang = chain_entang_helper(coord[key],entang,nc,dp,coord_loc,sidechain=sidechain)
        ent_param[i]= entang/n_applicable_atoms
    return ent_param

def chain_entang_helper(curr_coordinates,entang,nc,dp,coord_loc,sidechain=False):
    for key in range(1,nc+1):
        begin=(key -1)*dp + 11
        if sidechain:
            end = (key - 1)*dp + 89 - 10
        else:
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

def msd(coord,r_0,coord_loc=[3,6]):
    """Find the mean-squared displacement of the polymer at every timestep.

    Args:
    coords (dict):  dictionary with the number of timesteps as keys and 
        coordinates of all atoms at the corresponding timestep as a pandas 
        dataframe.
    r_0 (numpy array): array containing the coordinates of the atoms/beads in 
    the first timestep.
    coord_loc (list): columns which contain the coordinates.

    Returns:
    msd_list (list): List containing msd at every given timestep
    """

    msd_list=[]
    for key in coord:
        r_curr=coord[key]
        r_curr.sort_values(by=['id'],inplace=True)
        r_curr=r_curr.iloc[:,coord_loc[0]:coord_loc[1]].values
        dist=r_curr - r_0
        dist=np.square(dist)
        dist=np.sum(dist,axis=1)
        msd_list.append(np.average(dist))
    return msd_list

def msd_molecular_com(coord,r_0_com,coord_loc=[3,6]):
    """Find the mean-squared displacement of the molecular center of mass
    of the chain.
        Args:
    coords (dict):  dictionary with the number of timesteps as keys and 
        coordinates of all atoms at the corresponding timestep as a pandas 
        dataframe.
    r_0 (numpy array): array containing the coordinates of the center of the 
    atoms/beads in the first timestep.
    coord_loc (list): columns which contain the coordinates.

    Returns:
    msd_list (list): List containing msd at every given timestep
    """
    msd_list=[]
    for key in coord:
        r_curr=coord[key]
        total_chains=sorted(r_curr.mol.unique())
        r_curr_com=np.empty((len(total_chains),3),dtype=None)
        for chain in total_chains:
            curr_chain=r_curr[r_curr['mol']==chain].iloc[:,coord_loc[0]:coord_loc[1]].values
            curr_chain_com=np.mean(curr_chain,axis=0)
            r_curr_com[int(chain-1),:]=curr_chain_com
        dist=r_curr_com-r_0_com
        dist=np.square(dist)
        dist=np.sum(dist,axis=1)
        msd_list.append(np.average(dist))
    return msd_list

def msd_com_inner(coord,r_0,r_0_com,coord_loc=[3,6]):
    """g2(t)
    https://onlinelibrary.wiley.com/doi/abs/10.1002/polb.23175
    """ 
    msd_list=[]
    for key in coord:
        r_curr=coord[key]
        total_chains=sorted(r_curr.mol.unique())
        r_curr_com=np.empty((0,3))
        for chain in total_chains:
            curr_chain=r_curr[r_curr['mol']==chain].iloc[:,coord_loc[0]:coord_loc[1]].values
            dp=len(r_curr[r_curr['mol']==chain].iloc[:,coord_loc[0]].values)
            curr_chain_com=np.mean(curr_chain,axis=0)
            curr_chain_com_tile=np.tile(curr_chain_com,(dp,1))
            r_curr_com=np.append(r_curr_com,curr_chain_com_tile,axis=0)
        r_curr=coord[key]
        r_curr.sort_values(by=['id'],inplace=True)
        r_curr=r_curr.iloc[:,coord_loc[0]:coord_loc[1]].values
        dist=r_curr - r_0
        dist=dist-r_curr_com 
        dist=np.square(dist)
        dist=np.sum(dist,axis=1)
        msd_list.append(np.average(dist))
    return msd_list

def bonded_beads_distance(simname,coord_loc=[3,6],first_only=False,save=False):
    """Find the mean distance between bonded beads during deformation and return mean
    bonded distance at a specific timestep. This is the generic version for calculating
    bonded beads mean distance (backbone polymer versions)

    Args:
    simname (str): Name of the simulation file that needs to be analyzed.
    coord_loc (list): Column index for the coordinates (x and z, respectively) 
    in the trajectory file.
    
    Returns:
    mean_dist (float): Mean distance between bonded beads
    """
    if first_only:
        df, boxsize =extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    else:
        df, boxsize=extract.extract_unwrapped(simname,last_only=True,boxsize=True)
    coord=df['timestep_0']
    del df
    coord.sort_values(by=['id'],inplace=True)
    chains = coord['mol'].unique()
    diff_array=np.array([])
    for chain in chains:
        curr_chain_coords=coord[coord['mol']==chain].values[:,coord_loc[0]:coord_loc[1]]
        one_row_shifted=curr_chain_coords[1:,:]
        curr_chain_coords= np.delete(curr_chain_coords,-1,0)
        curr_diff_array=curr_chain_coords - one_row_shifted
        sidehalfx = boxsize[0]/2
        sidehalfy = boxsize[1]/2
        sidehalfz = boxsize[2]/2
        for index_c in range(len(curr_diff_array)):
            if (curr_diff_array[index_c][0] < -sidehalfx): 
                curr_diff_array[index_c][0] = curr_diff_array[index_c][0] + boxsize[0]
            if (curr_diff_array[index_c][0] > sidehalfx):
                curr_diff_array[index_c][0] = curr_diff_array[index_c][0] - boxsize[0]
            if (curr_diff_array[index_c][1] < -sidehalfy): 
                curr_diff_array[index_c][1] = curr_diff_array[index_c][1] + boxsize[1]
            if (curr_diff_array[index_c][1] > sidehalfy):
                curr_diff_array[index_c][1] = curr_diff_array[index_c][1] - boxsize[1]
            if (curr_diff_array[index_c][2] < -sidehalfz): 
                curr_diff_array[index_c][2] = curr_diff_array[index_c][2] + boxsize[2]
            if (curr_diff_array[index_c][2] > sidehalfz):
                curr_diff_array[index_c][2] = curr_diff_array[index_c][2] - boxsize[2]
        curr_diff_array=np.linalg.norm((curr_diff_array),axis=1)
        diff_array=np.append(diff_array,curr_diff_array)
    plt.hist(curr_diff_array)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Bond length',fontsize=20,fontfamily='Serif')
    if save:
        plt.tight_layout()
        plt.savefig(simname +'_bond_length.png',dpi=300)
    return np.mean(diff_array)

def bonded_beads_distance_whole(simname,coord_loc=[3,6],save=False):
    """Find the mean distance between bonded beads during deformation and return
    a list containing the mean bonded distance at each timestep. The only difference
    between this and bonded_beads_distance is that this function computes the mean
    bonded distance for ALL timesteps whereas the previous one computes it at a specific
    snapshot.

    Args:
    simname (str): Name of the simulation file that needs to be analyzed.
    coord_loc (list): Column index for the coordinates (x and z, respectively) 
    in the trajectory file.
    
    Returns:
    mean_dist (float): Mean distance between bonded beads
    """
    bonded_beads_distance_list=[]
    df, boxsize_whole =extract.extract_unwrapped(simname,boxsize_whole=True)
    for key in df:
        coord=df[key]
        coord.sort_values(by=['id'],inplace=True)
        chains = coord['mol'].unique()
        diff_array=np.array([])
        for chain in chains:
            curr_chain_coords=coord[coord['mol']==chain].values[:,coord_loc[0]:coord_loc[1]]
            one_row_shifted=curr_chain_coords[1:,:]
            curr_chain_coords= np.delete(curr_chain_coords,-1,0)
            curr_diff_array=curr_chain_coords - one_row_shifted
            boxsize = boxsize_whole[key]
            sidehalfx = boxsize[0]/2
            sidehalfy = boxsize[1]/2
            sidehalfz = boxsize[2]/2
            for index_c in range(len(curr_diff_array)):
                if (curr_diff_array[index_c][0] < -sidehalfx): 
                    curr_diff_array[index_c][0] = curr_diff_array[index_c][0] + boxsize[0]
                if (curr_diff_array[index_c][0] > sidehalfx):
                    curr_diff_array[index_c][0] = curr_diff_array[index_c][0] - boxsize[0]
                if (curr_diff_array[index_c][1] < -sidehalfy): 
                    curr_diff_array[index_c][1] = curr_diff_array[index_c][1] + boxsize[1]
                if (curr_diff_array[index_c][1] > sidehalfy):
                    curr_diff_array[index_c][1] = curr_diff_array[index_c][1] - boxsize[1]
                if (curr_diff_array[index_c][2] < -sidehalfz): 
                    curr_diff_array[index_c][2] = curr_diff_array[index_c][2] + boxsize[2]
                if (curr_diff_array[index_c][2] > sidehalfz):
                    curr_diff_array[index_c][2] = curr_diff_array[index_c][2] - boxsize[2]
            curr_diff_array=np.linalg.norm((curr_diff_array),axis=1)
            diff_array=np.append(diff_array,curr_diff_array)
        bonded_beads_distance_list.append(np.mean(diff_array))
    return bonded_beads_distance_list

def bonded_beads_distance_whole_backbone(simname,coord_loc=[3,6],save=False):
    """Find the mean distance between bonded beads during deformation and return
    a list containing the mean bonded distance at each timestep. This is a special
    version for polymers containing sidechain. It only works for 3 bead sidechain 
    versions with 128 DP. It calculates the mean distance of the beads that are present
    on the backbone and ignores the sidechain beads. 

    Args:
    simname (str): Name of the simulation file that needs to be analyzed.
    coord_loc (list): Column index for the coordinates (x and z, respectively) 
    in the trajectory file.
    
    Returns:
    mean_dist (float): Mean distance between bonded beads
    """
    bonded_beads_distance_list=[]
    df, boxsize_whole =extract.extract_unwrapped(simname,boxsize_whole=True)
    for key in df:
        coord=df[key]
        coord.sort_values(by=['id'],inplace=True)
        chains = coord['mol'].unique()
        diff_array=np.array([])
        for chain in chains:
            curr_chain_coords=coord[coord['mol']==chain].values[:,coord_loc[0]:coord_loc[1]]
            one_row_shifted=curr_chain_coords[1:,:]
            curr_chain_coords= np.delete(curr_chain_coords,-1,0)
            curr_diff_array=curr_chain_coords - one_row_shifted
            curr_diff_array = curr_diff_array[:88,:]
            boxsize = boxsize_whole[key]
            sidehalfx = boxsize[0]/2
            sidehalfy = boxsize[1]/2
            sidehalfz = boxsize[2]/2
            for index_c in range(len(curr_diff_array)):
                if (curr_diff_array[index_c][0] < -sidehalfx): 
                    curr_diff_array[index_c][0] = curr_diff_array[index_c][0] + boxsize[0]
                if (curr_diff_array[index_c][0] > sidehalfx):
                    curr_diff_array[index_c][0] = curr_diff_array[index_c][0] - boxsize[0]
                if (curr_diff_array[index_c][1] < -sidehalfy): 
                    curr_diff_array[index_c][1] = curr_diff_array[index_c][1] + boxsize[1]
                if (curr_diff_array[index_c][1] > sidehalfy):
                    curr_diff_array[index_c][1] = curr_diff_array[index_c][1] - boxsize[1]
                if (curr_diff_array[index_c][2] < -sidehalfz): 
                    curr_diff_array[index_c][2] = curr_diff_array[index_c][2] + boxsize[2]
                if (curr_diff_array[index_c][2] > sidehalfz):
                    curr_diff_array[index_c][2] = curr_diff_array[index_c][2] - boxsize[2]
            curr_diff_array=np.linalg.norm((curr_diff_array),axis=1)
            diff_array=np.append(diff_array,curr_diff_array)
        bonded_beads_distance_list.append(np.mean(diff_array))
    return bonded_beads_distance_list

def bonded_beads_distance_whole_sidechain(simname,coord_loc=[3,6],save=False):
    """Find the mean distance between bonded beads during deformation and return
    a list containing the mean bonded distance at each timestep. This version 
    calculates the sidechain mean bonded distance and ignores the backbone beads.
    It only works for 128DP 3 bead sidechain model.

    Args:
    simname (str): Name of the simulation file that needs to be analyzed.
    coord_loc (list): Column index for the coordinates (x and z, respectively) 
    in the trajectory file.
    
    Returns:
    mean_dist (float): Mean distance between bonded beads
    """
    bonded_beads_distance_list=[]
    df, boxsize_whole =extract.extract_unwrapped(simname,boxsize_whole=True)
    for key in df:
        coord=df[key]
        coord.sort_values(by=['id'],inplace=True)
        chains = coord['mol'].unique()
        diff_array=np.array([])
        new_diff_array=np.array([None,None,None])
        for chain in chains:
            new_diff_array=np.zeros((26,3))
            curr_chain_coords=coord[coord['mol']==chain].values[:,coord_loc[0]:coord_loc[1]]
            one_row_shifted=curr_chain_coords[1:,:]
            curr_chain_coords= np.delete(curr_chain_coords,-1,0)
            curr_diff_array=curr_chain_coords - one_row_shifted
            curr_diff_array = curr_diff_array[89:,:]
            inner_index_diff=0
            for index in range(curr_diff_array.shape[0]):
                if (index+1)%3!=0:
                    try:
                        new_diff_array[inner_index_diff]=curr_diff_array[index,:]
                        inner_index_diff+=1
                    except IndexError:
                        pass
            curr_diff_array=new_diff_array
            boxsize = boxsize_whole[key]
            sidehalfx = boxsize[0]/2
            sidehalfy = boxsize[1]/2
            sidehalfz = boxsize[2]/2
            for index_c in range(len(curr_diff_array)):
                if (curr_diff_array[index_c][0] < -sidehalfx): 
                    curr_diff_array[index_c][0] = curr_diff_array[index_c][0] + boxsize[0]
                if (curr_diff_array[index_c][0] > sidehalfx):
                    curr_diff_array[index_c][0] = curr_diff_array[index_c][0] - boxsize[0]
                if (curr_diff_array[index_c][1] < -sidehalfy): 
                    curr_diff_array[index_c][1] = curr_diff_array[index_c][1] + boxsize[1]
                if (curr_diff_array[index_c][1] > sidehalfy):
                    curr_diff_array[index_c][1] = curr_diff_array[index_c][1] - boxsize[1]
                if (curr_diff_array[index_c][2] < -sidehalfz): 
                    curr_diff_array[index_c][2] = curr_diff_array[index_c][2] + boxsize[2]
                if (curr_diff_array[index_c][2] > sidehalfz):
                    curr_diff_array[index_c][2] = curr_diff_array[index_c][2] - boxsize[2]
            
            curr_diff_array=np.linalg.norm((curr_diff_array),axis=1)
            diff_array=np.append(diff_array,curr_diff_array)
        bonded_beads_distance_list.append(np.mean(diff_array))
    return bonded_beads_distance_list

def charged_beads_mean_distance(simname,cutoff,coords_loc=[3,6]):
    coord,bs=extract.extract_unwrapped(simname,boxsize_whole=True)
    correlation_fn_whole=np.zeros((len(coord.keys()),1))
    outer_index = 0
    for key in coord:
        current_timestep=coord[key]
        type2=current_timestep[current_timestep['type']==2].iloc[:,
            coords_loc[0]:coords_loc[1]].values 
        type3=current_timestep[current_timestep['type']==3].iloc[:,
            coords_loc[0]:coords_loc[1]].values 
        distsn=np.zeros((type3.shape[0],1))
        correlation_fn=[]
        box_l=bs[key]
        sidehx=box_l[0]/2
        sidehy=box_l[1]/2
        sidehz=box_l[2]/2
        for nindex,ncharge in enumerate(type3):
            distsp=[]
            for pindex,pcharge in enumerate(type2):
                dx = pcharge[0] - ncharge[0]
                dy = pcharge[1] - ncharge[1]
                dz = pcharge[2] - ncharge[2]
                if (dx < -sidehx):   dx = dx + box_l[0]
                if (dx > sidehx):    dx = dx - box_l[0]
                if (dy < -sidehy):   dy = dy + box_l[1]
                if (dy > sidehy):    dy = dy - box_l[1]
                if (dz < -sidehz):   dz = dz + box_l[2]
                if (dz > sidehz):    dz = dz - box_l[2]
                distAB = [dx,dy,dz]
                distsp.append(np.linalg.norm(distAB))
            distsp=np.asarray(distsp)
            if np.asarray(np.where(distsp<cutoff)).shape[1]>0:
                distsp=np.take(distsp,np.where(distsp<cutoff))
                correlation_fn.append(np.mean(distsp))
            else:
                pass
        correlation_fn_whole[outer_index]=np.mean(correlation_fn)
        outer_index+=1
    return correlation_fn_whole

def return_youngs_modulus(simname):
    """Find young's modulus (0.03 strain)
    """
    df1,df2=extract.extract_def(simname)
    df3,df4=extract.extract_def('CG_256C_128DP_v20_deform_lj_kg_uk_8_x')
    df1=df1.values
    df2=df2.values
    df3=df3.values
    deformation_along = np.argmax(np.array([np.std(df1[:,1]),
                                    np.std(df1[:,2]),
                                    np.std(df1[:,3])]))
    strain=df3[:,0]
    #for index, element in enumerate(strain):
    #    print(index,element)
    till=196
    slope, intercept, r_value, p_value, std_err = stats.linregress(strain[:till],df1[:till,deformation_along+1])
    return slope

def tangent_modulus(simname):
    """Compute tangent modulus between 0.05 and 0.08 strain
    
    Args:
    simname (str): filename of the simulation

    Returns:
    slope (float): tangent modulus
    """
    df1,df2=extract.extract_def(simname)
    df3,df4=extract.extract_def('CG_256C_128DP_v20_deform_lj_kg_uk_3_x')
    df1=df1.values
    df2=df2.values
    df3=df3.values
    deformation_along = np.argmax(np.array([np.std(df1[:,1]),
                                    np.std(df1[:,2]),
                                    np.std(df1[:,3])]))
    strain=df3[:,0]
    #for index, element in enumerate(strain):
    #    print(index,element)
    begin=30
    till=61
    slope, intercept, r_value, p_value, std_err = stats.linregress(strain[begin:till],df1[begin:till,deformation_along+1])
    return slope

def tangent_modulus_with_error(*args):
    """Compute the tangent modulus for each of the given files (they 
    should be repeats of the same simulation) and return the mean, 
    standard deviation and standard error.

    Args:
    *args (comma separated str): Filename of repeats of the same
    simulation.

    Returns:
    tm_mean (float), tm_standard_deviation (float), 
    tm_standard error (float)
    tm - tangent modulus
    """
    tm=[]
    for simname in args:
        tm.append(tangent_modulus(simname))
    tm_mean = np.mean(np.asarray(tm))
    tm_std = np.std(np.asarray(tm))
    tm_se = stats.sem(np.asarray(tm))
    return tm_mean, tm_std, tm_se

def youngs_modulus_with_error(*args):
    ym=[]
    for simname in args:
        ym.append(return_youngs_modulus(simname))
    ym_mean = np.mean(np.asarray(ym))
    ym_std = np.std(np.asarray(ym))
    ym_se = stats.sem(np.asarray(ym))
    return ym_mean, ym_std, ym_se

def poisson_ratio(simname):
    """Compute poisson's ratio between ____
    """
    df1,df2=extract.extract_def(simname)
    df3,df4=extract.extract_def('CG_256C_128DP_v20_deform_lj_kg_uk_3_x')
    df1=df1.values
    df2=df2.values
    df3=df3.values
    deformation_along = np.argmax(np.array([np.std(df1[:,1]),
                                    np.std(df1[:,2]),
                                    np.std(df1[:,3])]))
    strain=df3[:,0]
    begin=0
    till=8
    if (deformation_along + 1) == 1:
        transverse_1=5
        transverse_2=6
    elif (deformation_along + 1) == 2:
        transverse_1=4
        transverse_2=6
    else:
        transverse_1=4
        transverse_2=5
    transverse_1_list=df1[begin:till,transverse_1]
    transverse_2_list=df1[begin:till,transverse_2]
    transverse_1_list=(transverse_1_list - transverse_1_list[0])/transverse_1_list[0]
    transverse_2_list=(transverse_2_list - transverse_2_list[0])/transverse_2_list[0]
    strain_list=strain[begin:till]
    poisson_1=-(transverse_1_list/strain_list)
    poisson_2=-(transverse_2_list/strain_list)
    return poisson_1[7], poisson_2[7]

def poisson_ratio_with_error(*args):
    pr=[]
    for simname in args:
        pr.append(poisson_ratio(simname))
    pr_mean = np.mean(np.asarray(pr))
    pr_std = np.std(np.asarray(pr))
    pr_se = stats.sem(np.asarray(pr))
    return pr_mean, pr_std, pr_se

def yield_point(simname):
    df1,df2=extract.extract_def(simname)
    df3,df4=extract.extract_def('CG_256C_128DP_v20_deform_lj_kg_uk_3_x')
    df1=df1.values
    df2=df2.values
    df3=df3.values
    deformation_along = np.argmax(np.array([np.std(df1[:,1]),
                                    np.std(df1[:,2]),
                                    np.std(df1[:,3])]))
    strain=df3[:,0]
    #for index, element in enumerate(strain):
    #    print(index,element)
    till=19
    slope, intercept, r_value, p_value, std_err = stats.linregress(strain[:till],df1[:till,deformation_along+1])

def return_mean_energies(*args):
    """This function returns the mean energies (zero start) of repeats for the same
    simulation. Simulation filenames of all the repeats need to be passed
    as arguments.

    *args (str): Filenames of repeats of the same simulation.

    Returns:
    ebond (numpy array): Mean bonded energy
    ecoul (numpy array): Mean coulombic energy
    evdwl (numpy array): Mean van der Waals energy
    etotal (numpy array): Mean total energy
    """

    df1v20_x,  df2v20_x  = extract.extract_def(args[0])
    df1v20_y,  df2v20_y  = extract.extract_def(args[1])
    df1v20_z,  df2v20_z  = extract.extract_def(args[2])
    df1v20_x2, df2v20_x2 = extract.extract_def(args[3])
    df1v20_y2, df2v20_y2 = extract.extract_def(args[4])
    df1v20_z2, df2v20_z2 = extract.extract_def(args[5])
    df1v20_x3, df2v20_x3 = extract.extract_def(args[6])
    df1v20_y3, df2v20_y3 = extract.extract_def(args[7])
    df1v20_z3, df2v20_z3 = extract.extract_def(args[8])
    df1v20_x4, df2v20_x4 = extract.extract_def(args[9])
    df1v20_y4, df2v20_y4 = extract.extract_def(args[10])
    df1v20_z4, df2v20_z4 = extract.extract_def(args[11])

    ebondx = (df1v20_x.iloc[:,9] - df1v20_x.iloc[0,9]).values
    ecoulx = (df1v20_x.iloc[:,12] - df1v20_x.iloc[0,12]).values
    evdwlx = (df1v20_x.iloc[:,13] - df1v20_x.iloc[0,13]).values
    etotalx = (df1v20_x.iloc[:,14] - df1v20_x.iloc[0,14]).values

    ebondy = (df1v20_y.iloc[:,9] - df1v20_y.iloc[0,9]).values
    ecouly= (df1v20_y.iloc[:,12] - df1v20_y.iloc[0,12]).values
    evdwly= (df1v20_y.iloc[:,13] - df1v20_y.iloc[0,13]).values
    etotaly = (df1v20_y.iloc[:,14] - df1v20_y.iloc[0,14]).values

    ebondz = (df1v20_z.iloc[:,9] - df1v20_z.iloc[0,9]).values
    ecoulz= (df1v20_z.iloc[:,12] - df1v20_z.iloc[0,12]).values
    evdwlz= (df1v20_z.iloc[:,13] - df1v20_z.iloc[0,13]).values
    etotalz = (df1v20_z.iloc[:,14] - df1v20_z.iloc[0,14]).values

    ebondx2 = (df1v20_x2.iloc[:,9] - df1v20_x2.iloc[0,9]).values
    ecoulx2 = (df1v20_x2.iloc[:,12] - df1v20_x2.iloc[0,12]).values
    evdwlx2 = (df1v20_x2.iloc[:,13] - df1v20_x2.iloc[0,13]).values
    etotalx2 = (df1v20_x2.iloc[:,14] - df1v20_x2.iloc[0,14]).values

    ebondy2 = (df1v20_y2.iloc[:,9] - df1v20_y2.iloc[0,9]).values
    ecouly2= (df1v20_y2.iloc[:,12] - df1v20_y2.iloc[0,12]).values
    evdwly2= (df1v20_y2.iloc[:,13] - df1v20_y2.iloc[0,13]).values
    etotaly2 = (df1v20_y2.iloc[:,14] - df1v20_y2.iloc[0,14]).values

    ebondz2 = (df1v20_z2.iloc[:,9] - df1v20_z2.iloc[0,9]).values
    ecoulz2= (df1v20_z2.iloc[:,12] - df1v20_z2.iloc[0,12]).values
    evdwlz2= (df1v20_z2.iloc[:,13] - df1v20_z2.iloc[0,13]).values
    etotalz2 = (df1v20_z2.iloc[:,14] - df1v20_z2.iloc[0,14]).values
    
    ebondx3 = (df1v20_x3.iloc[:,9] - df1v20_x3.iloc[0,9]).values
    ecoulx3 = (df1v20_x3.iloc[:,12] - df1v20_x3.iloc[0,12]).values
    evdwlx3 = (df1v20_x3.iloc[:,13] - df1v20_x3.iloc[0,13]).values
    etotalx3 = (df1v20_x3.iloc[:,14] - df1v20_x3.iloc[0,14]).values

    ebondy3 = (df1v20_y3.iloc[:,9] - df1v20_y3.iloc[0,9]).values
    ecouly3= (df1v20_y3.iloc[:,12] - df1v20_y3.iloc[0,12]).values
    evdwly3= (df1v20_y3.iloc[:,13] - df1v20_y3.iloc[0,13]).values
    etotaly3 = (df1v20_y3.iloc[:,14] - df1v20_y3.iloc[0,14]).values

    ebondz3 = (df1v20_z3.iloc[:,9] - df1v20_z3.iloc[0,9]).values
    ecoulz3= (df1v20_z3.iloc[:,12] - df1v20_z3.iloc[0,12]).values
    evdwlz3= (df1v20_z3.iloc[:,13] - df1v20_z3.iloc[0,13]).values
    etotalz3 = (df1v20_z3.iloc[:,14] - df1v20_z3.iloc[0,14]).values

    ebondx4 = (df1v20_x4.iloc[:,9] - df1v20_x4.iloc[0,9]).values
    ecoulx4 = (df1v20_x4.iloc[:,12] - df1v20_x4.iloc[0,12]).values
    evdwlx4 = (df1v20_x4.iloc[:,13] - df1v20_x4.iloc[0,13]).values
    etotalx4 = (df1v20_x4.iloc[:,14] - df1v20_x4.iloc[0,14]).values

    ebondy4 = (df1v20_y4.iloc[:,9] - df1v20_y4.iloc[0,9]).values
    ecouly4 = (df1v20_y4.iloc[:,12] - df1v20_y4.iloc[0,12]).values
    evdwly4 = (df1v20_y4.iloc[:,13] - df1v20_y4.iloc[0,13]).values
    etotaly4 = (df1v20_y4.iloc[:,14] - df1v20_y4.iloc[0,14]).values

    ebondz4 = (df1v20_z4.iloc[:,9] - df1v20_z4.iloc[0,9]).values
    ecoulz4 = (df1v20_z4.iloc[:,12] - df1v20_z4.iloc[0,12]).values
    evdwlz4 = (df1v20_z4.iloc[:,13] - df1v20_z4.iloc[0,13]).values
    etotalz4 = (df1v20_z4.iloc[:,14] - df1v20_z4.iloc[0,14]).values

    ebond_temp = np.zeros((ebondx.shape[0],12))
    ebond_temp[:,0]=ebondx
    ebond_temp[:,1]=ebondy
    ebond_temp[:,2]=ebondz
    ebond_temp[:,3]=ebondx2
    ebond_temp[:,4]=ebondy2
    ebond_temp[:,5]=ebondz2
    ebond_temp[:,6]=ebondx3
    ebond_temp[:,7]=ebondy3
    ebond_temp[:,8]=ebondz3
    ebond_temp[:,9]=ebondx4
    ebond_temp[:,10]=ebondy4
    ebond_temp[:,11]=ebondz4
    ebond_temp = np.mean(ebond_temp, axis = 1)

    ecoul_temp = np.zeros((ecoulx.shape[0],12))
    ecoul_temp[:,0]=ecoulx
    ecoul_temp[:,1]=ecouly
    ecoul_temp[:,2]=ecoulz
    ecoul_temp[:,3]=ecoulx2
    ecoul_temp[:,4]=ecouly2
    ecoul_temp[:,5]=ecoulz2
    ecoul_temp[:,6]=ecoulx3
    ecoul_temp[:,7]=ecouly3
    ecoul_temp[:,8]=ecoulz3
    ecoul_temp[:,9]=ecoulx4
    ecoul_temp[:,10]=ecouly4
    ecoul_temp[:,11]=ecoulz4
    ecoul_temp = np.mean(ecoul_temp, axis = 1)

    evdwl_temp = np.zeros((evdwlx.shape[0],12))
    evdwl_temp[:,0]=evdwlx
    evdwl_temp[:,1]=evdwly
    evdwl_temp[:,2]=evdwlz
    evdwl_temp[:,3]=evdwlx2
    evdwl_temp[:,4]=evdwly2
    evdwl_temp[:,5]=evdwlz2
    evdwl_temp[:,6]=evdwlx3
    evdwl_temp[:,7]=evdwly3
    evdwl_temp[:,8]=evdwlz3
    evdwl_temp[:,9]=evdwlx4
    evdwl_temp[:,10]=evdwly4
    evdwl_temp[:,11]=evdwlz4
    evdwl_temp = np.mean(evdwl_temp, axis = 1)

    etotal_temp = np.zeros((etotalx.shape[0],12))
    etotal_temp[:,0]=etotalx
    etotal_temp[:,1]=etotaly
    etotal_temp[:,2]=etotalz
    etotal_temp[:,3]=etotalx2
    etotal_temp[:,4]=etotaly2
    etotal_temp[:,5]=etotalz2
    etotal_temp[:,6]=etotalx3
    etotal_temp[:,7]=etotaly3
    etotal_temp[:,8]=etotalz3
    etotal_temp[:,9]=etotalx4
    etotal_temp[:,10]=etotaly4
    etotal_temp[:,11]=etotalz4
    etotal_temp = np.mean(etotal_temp, axis = 1)

    return ebond_temp, ecoul_temp, evdwl_temp, etotal_temp
    
def return_mean_abs_energies(*args):
    """This function returns the mean energies (absolute) of repeats for the same
    simulation. Simulation filenames of all the repeats need to be passed
    as arguments.

    *args (str): Filenames of repeats of the same simulation.

    Returns:
    ebond (numpy array): Mean bonded energy
    ecoul (numpy array): Mean coulombic energy
    evdwl (numpy array): Mean van der Waals energy
    etotal (numpy array): Mean total energy
    """
    df1v20_x,  df2v20_x  = extract.extract_def(args[0])
    df1v20_y,  df2v20_y  = extract.extract_def(args[1])
    df1v20_z,  df2v20_z  = extract.extract_def(args[2])
    df1v20_x2, df2v20_x2 = extract.extract_def(args[3])
    df1v20_y2, df2v20_y2 = extract.extract_def(args[4])
    df1v20_z2, df2v20_z2 = extract.extract_def(args[5])
    df1v20_x3, df2v20_x3 = extract.extract_def(args[6])
    df1v20_y3, df2v20_y3 = extract.extract_def(args[7])
    df1v20_z3, df2v20_z3 = extract.extract_def(args[8])
    df1v20_x4, df2v20_x4 = extract.extract_def(args[9])
    df1v20_y4, df2v20_y4 = extract.extract_def(args[10])
    df1v20_z4, df2v20_z4 = extract.extract_def(args[11])

    ebondx = df1v20_x.iloc[:,9].values
    ecoulx = df1v20_x.iloc[:,12].values 
    evdwlx = df1v20_x.iloc[:,13].values 
    etotalx = df1v20_x.iloc[:,14].values

    ebondy = df1v20_y.iloc[:,9].values   
    ecouly= df1v20_y.iloc[:,12].values   
    evdwly= df1v20_y.iloc[:,13].values   
    etotaly =df1v20_y.iloc[:,14].values 

    ebondz = df1v20_z.iloc[:,9].values   
    ecoulz= df1v20_z.iloc[:,12].values   
    evdwlz= df1v20_z.iloc[:,13].values   
    etotalz = df1v20_z.iloc[:,14].values 

    ebondx2 = df1v20_x2.iloc[:,9].values   
    ecoulx2 = df1v20_x2.iloc[:,12].values  
    evdwlx2 = df1v20_x2.iloc[:,13].values  
    etotalx2 = df1v20_x2.iloc[:,14].values   

    ebondy2 =df1v20_y2.iloc[:,9].values   
    ecouly2= df1v20_y2.iloc[:,12].values   
    evdwly2= df1v20_y2.iloc[:,13].values   
    etotaly2 = df1v20_y2.iloc[:,14].values   

    ebondz2 = df1v20_z2.iloc[:,9].values   
    ecoulz2= df1v20_z2.iloc[:,12].values   
    evdwlz2= df1v20_z2.iloc[:,13].values   
    etotalz2 = df1v20_z2.iloc[:,14].values 
    
    ebondx3 = df1v20_x3.iloc[:,9]
    ecoulx3 = df1v20_x3.iloc[:,12] 
    evdwlx3 = df1v20_x3.iloc[:,13] 
    etotalx3 = df1v20_x3.iloc[:,14]

    ebondy3 = df1v20_y3.iloc[:,9].values
    ecouly3= df1v20_y3.iloc[:,12].values
    evdwly3= df1v20_y3.iloc[:,13].values
    etotaly3 = df1v20_y3.iloc[:,14].values

    ebondz3 = df1v20_z3.iloc[:,9].values
    ecoulz3= df1v20_z3.iloc[:,12].values
    evdwlz3= df1v20_z3.iloc[:,13].values
    etotalz3 = df1v20_z3.iloc[:,14].values

    ebondx4 = df1v20_x4.iloc[:,9].values
    ecoulx4 = df1v20_x4.iloc[:,12].values
    evdwlx4 = df1v20_x4.iloc[:,13].values
    etotalx4 = df1v20_x4.iloc[:,14].values

    ebondy4 = df1v20_y4.iloc[:,9].values
    ecouly4 = df1v20_y4.iloc[:,12].values
    evdwly4 = df1v20_y4.iloc[:,13].values
    etotaly4 = df1v20_y4.iloc[:,14].values

    ebondz4 = df1v20_z4.iloc[:,9].values
    ecoulz4 = df1v20_z4.iloc[:,12].values
    evdwlz4 = df1v20_z4.iloc[:,13].values
    etotalz4 = df1v20_z4.iloc[:,14].values

    ebond_temp = np.zeros((ebondx.shape[0],12))
    ebond_temp[:,0]=ebondx
    ebond_temp[:,1]=ebondy
    ebond_temp[:,2]=ebondz
    ebond_temp[:,3]=ebondx2
    ebond_temp[:,4]=ebondy2
    ebond_temp[:,5]=ebondz2
    ebond_temp[:,6]=ebondx3
    ebond_temp[:,7]=ebondy3
    ebond_temp[:,8]=ebondz3
    ebond_temp[:,9]=ebondx4
    ebond_temp[:,10]=ebondy4
    ebond_temp[:,11]=ebondz4
    ebond_temp = np.mean(ebond_temp, axis = 1)

    ecoul_temp = np.zeros((ecoulx.shape[0],12))
    ecoul_temp[:,0]=ecoulx
    ecoul_temp[:,1]=ecouly
    ecoul_temp[:,2]=ecoulz
    ecoul_temp[:,3]=ecoulx2
    ecoul_temp[:,4]=ecouly2
    ecoul_temp[:,5]=ecoulz2
    ecoul_temp[:,6]=ecoulx3
    ecoul_temp[:,7]=ecouly3
    ecoul_temp[:,8]=ecoulz3
    ecoul_temp[:,9]=ecoulx4
    ecoul_temp[:,10]=ecouly4
    ecoul_temp[:,11]=ecoulz4
    ecoul_temp = np.mean(ecoul_temp, axis = 1)

    evdwl_temp = np.zeros((evdwlx.shape[0],12))
    evdwl_temp[:,0]=evdwlx
    evdwl_temp[:,1]=evdwly
    evdwl_temp[:,2]=evdwlz
    evdwl_temp[:,3]=evdwlx2
    evdwl_temp[:,4]=evdwly2
    evdwl_temp[:,5]=evdwlz2
    evdwl_temp[:,6]=evdwlx3
    evdwl_temp[:,7]=evdwly3
    evdwl_temp[:,8]=evdwlz3
    evdwl_temp[:,9]=evdwlx4
    evdwl_temp[:,10]=evdwly4
    evdwl_temp[:,11]=evdwlz4
    evdwl_temp = np.mean(evdwl_temp, axis = 1)

    etotal_temp = np.zeros((etotalx.shape[0],12))
    etotal_temp[:,0]=etotalx
    etotal_temp[:,1]=etotaly
    etotal_temp[:,2]=etotalz
    etotal_temp[:,3]=etotalx2
    etotal_temp[:,4]=etotaly2
    etotal_temp[:,5]=etotalz2
    etotal_temp[:,6]=etotalx3
    etotal_temp[:,7]=etotaly3
    etotal_temp[:,8]=etotalz3
    etotal_temp[:,9]=etotalx4
    etotal_temp[:,10]=etotaly4
    etotal_temp[:,11]=etotalz4
    etotal_temp = np.mean(etotal_temp, axis = 1)

    return ebond_temp, ecoul_temp, evdwl_temp, etotal_temp
