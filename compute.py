import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate.quadrature import simps
from scipy import stats
import random
from shapely.geometry import Point, LineString

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

def periodic_distance_return_new_coords(x,y,z,x2,y2,z2,sidehx,sidehy,sidehz,bs):
    """Find out new coordinates based on minimum image convention.
    """
    dx = x2 - x
    dy = y2 - y
    dz = z2 - z
    if (dx < -sidehx):   x = x - bs[0]
    if (dx > sidehx):    x = x + bs[0]
    if (dy < -sidehy):   y = y - bs[1]
    if (dy > sidehy):    y = y + bs[1]
    if (dz < -sidehz):   z = z - bs[2]
    if (dz > sidehz):    z = z + bs[2]
    return [x, y, z]
    
def get_new_coords(curr_coordinates, bs, box_images):
    """Get new coordinates from current coordinates based on box size
    and box_image.
    """
    for index, element in enumerate(curr_coordinates):
        if box_images[index]==1:
            curr_coordinates[index] = curr_coordinates[index]  + bs[index]
        elif box_images[index]==-1:
            curr_coordinates[index] = curr_coordinates[index] - bs[index]
    return curr_coordinates

def get_new_chain_coords(coord_curr,bs,coords_loc=[3,6]):
    coord_curr_copy=coord_curr.copy()
    sidehx=bs[0]/2
    sidehy=bs[1]/2
    sidehz=bs[2]/2
    for index,element in enumerate(coord_curr.iterrows()):
        if index!=127:
            bead_1=coord_curr_copy.iloc[index,coords_loc[0]:coords_loc[1]].values
            bead_2=coord_curr_copy.iloc[index+1,coords_loc[0]:coords_loc[1]].values
            dist_beads=np.linalg.norm(bead_2-bead_1)
            if dist_beads>1.5:
                new_bead_2=periodic_distance_return_new_coords(bead_2[0],bead_2[1],bead_2[2],
                    bead_1[0],bead_1[1],bead_1[2],
                    sidehx,sidehy,sidehz,bs)
                coord_curr_copy.iat[index+1,3]=new_bead_2[0]
                coord_curr_copy.iat[index+1,4]=new_bead_2[1]
                coord_curr_copy.iat[index+1,5]=new_bead_2[2]
    return coord_curr_copy

def get_bonds_atomids(topologyfilename):
    path=fileIO.findInSubdirectory(topologyfilename)
    df = pd.read_csv(path,header=None,index_col=False)
    for index in range(len(df)):
        if df.iloc[index].str.split()[0][0]=='Bonds':
            df2 = df.iloc[index +1:index+32512+1]
    df2 = df2[0].str.split('\t',4,expand=True)
    num2str = lambda x : float(x) 
    df2 = df2.applymap(num2str)
    df2.columns=['bondid','bondtype','atom1id','atom2id']
    return df2

def get_new_chain_coords_forsidechainmodel(coord_curr,bs,bondids,coords_loc=[3,6]):
    coord_curr_copy=coord_curr.copy()
    sidehx=bs[0]/2
    sidehy=bs[1]/2
    sidehz=bs[2]/2
    for index,element in enumerate(coord_curr.iterrows()):
        if index<88:
            bead_1=coord_curr_copy.iloc[index,coords_loc[0]:coords_loc[1]].values
            bead_2=coord_curr_copy.iloc[index+1,coords_loc[0]:coords_loc[1]].values
            dist_beads=np.linalg.norm(bead_2-bead_1)
            if dist_beads>1.5:
                new_bead_2=periodic_distance_return_new_coords(bead_2[0],bead_2[1],bead_2[2],
                    bead_1[0],bead_1[1],bead_1[2],
                    sidehx,sidehy,sidehz,bs)
                coord_curr_copy.iat[index+1,3]=new_bead_2[0]
                coord_curr_copy.iat[index+1,4]=new_bead_2[1]
                coord_curr_copy.iat[index+1,5]=new_bead_2[2]
        elif index>88:
            if (index+1)%3==0:
                sidechainbondindex = np.where(bondids['atom2id']==index+1)[0][0]
                sidechainbondsite = int(bondids.iloc[sidechainbondindex][2])
                bead_1=coord_curr_copy.iloc[sidechainbondsite,coords_loc[0]:coords_loc[1]].values
                bead_2=coord_curr_copy.iloc[index,coords_loc[0]:coords_loc[1]].values
                if np.linalg.norm(bead_2 -bead_1)>1.5:
                    new_bead_2=periodic_distance_return_new_coords(bead_2[0],bead_2[1],bead_2[2],
                    bead_1[0],bead_1[1],bead_1[2],
                    sidehx,sidehy,sidehz,bs)
                    coord_curr_copy.iat[index,3]=new_bead_2[0]
                    coord_curr_copy.iat[index,4]=new_bead_2[1]
                    coord_curr_copy.iat[index,5]=new_bead_2[2]
            else:
                bead_1=coord_curr_copy.iloc[index-1,coords_loc[0]:coords_loc[1]].values
                bead_2=coord_curr_copy.iloc[index,coords_loc[0]:coords_loc[1]].values
                dist_beads=np.linalg.norm(bead_2-bead_1)
                if dist_beads>1.5:
                    new_bead_2=periodic_distance_return_new_coords(bead_2[0],bead_2[1],bead_2[2],
                    bead_1[0],bead_1[1],bead_1[2],
                    sidehx,sidehy,sidehz,bs)
                    coord_curr_copy.iat[index,3]=new_bead_2[0]
                    coord_curr_copy.iat[index,4]=new_bead_2[1]
                    coord_curr_copy.iat[index,5]=new_bead_2[2]
    return coord_curr_copy

def persistence_length_exp(simname,nc,coord_loc=[3,6]):
    """Compute the persistence length of the first snapshot (from the
    given trajectory). There are many ways to compute persistence length.
    Here, a microscopic definition is used.
    """     
    coords, bs = extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    rms_bond_length=bonded_beads_distance(simname,root_mean_square=True)
    outer_index=0
    all_chains=list(coords['timestep_0'].mol.unique())
    curr_cos=np.zeros((nc,63))
    key='timestep_0'
    for chain in all_chains:
        persis_length_list =[]
        coord_curr=coords[key][coords[key]['mol']==chain]
        dp = len(coord_curr)
        id_start=(dp)*(chain-1)+1
        id_end = (chain)*(dp)
        coord_curr_copy = get_new_chain_coords(coord_curr, bs)
        H=id_start + int(dp/2) # half
        r_n = coord_curr_copy[coord_curr_copy['id']==H].iloc[:,coord_loc[0]:coord_loc[1]].values
        r_m = coord_curr_copy[coord_curr_copy['id']==H + 1].iloc[:,coord_loc[0]:coord_loc[1]].values
        vec_b_nm = r_m - r_n
        vec_b_nm=vec_b_nm[0]
        sum_iter=int(dp/2)-1
        for index in range(sum_iter):
            left_index = H - index
            right_index = H + index
            r_flt = coord_curr_copy[coord_curr_copy['id']==left_index].iloc[:,coord_loc[0]:coord_loc[1]].values # first left term
            r_slt = coord_curr_copy[coord_curr_copy['id']==left_index+1].iloc[:,coord_loc[0]:coord_loc[1]].values # second left term
            b_left = r_slt - r_flt
            r_frt = coord_curr_copy[coord_curr_copy['id']==right_index].iloc[:,coord_loc[0]:coord_loc[1]].values # first right term
            r_srt = coord_curr_copy[coord_curr_copy['id']==right_index+1].iloc[:,coord_loc[0]:coord_loc[1]].values # second right term
            b_right = r_srt - r_frt
            b_left=b_left[0]
            b_right=b_right[0]
            cos_sim_left=np.dot(vec_b_nm,b_left)/(np.linalg.norm(vec_b_nm)*np.linalg.norm(b_left))
            cos_sim_right=np.dot(vec_b_nm,b_right)/(np.linalg.norm(vec_b_nm)*np.linalg.norm(b_right))
            curr_cos[int(chain)-1,index]=(cos_sim_left+cos_sim_right)/2
    lp=np.mean(curr_cos,axis=0)
    lp_list=np.zeros((63,1))
    for index,element in enumerate(lp):
        print(element)
        lp_list[index]=(-index*rms_bond_length)/np.log(element)
    return lp,lp_list

def persistence_length(simname,topologyfilename,coord_loc=[3,6],sidechain=False):
    """Compute the persistence length of the first snapshot (from the
    given trajectory). There are many ways to compute persistence length.
    Here, a microscopic definition is used.
    """     
    coords, bs = extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    if sidechain:
        rms_bond_length = bonded_beads_distance_whole_backbone(simname,root_mean_square=True)
        bondids = get_bonds_atomids(topologyfilename)
    else:
        rms_bond_length=bonded_beads_distance(simname,root_mean_square=True)
    outer_index=0
    all_chains=list(coords['timestep_0'].mol.unique())
    nc = len(all_chains)
    persis_length_outer_list=[None]*nc
    for key in coords:
        for chain in all_chains:
            persis_length_list =[]
            coord_curr=coords[key][coords[key]['mol']==chain]
            dp = len(coord_curr)
            id_start=(dp)*(chain-1)+1
            if sidechain:
                id_end = (chain-1)*dp + 89
                coord_curr_copy=get_new_chain_coords_forsidechainmodel(coord_curr,bs,bondids)
                dp = 89
            else:
                id_end = (chain)*(dp)
                coord_curr_copy = get_new_chain_coords(coord_curr, bs)
            H=id_start + int(dp/2) # half
            r_n = coord_curr_copy[coord_curr_copy['id']==H].iloc[:,coord_loc[0]:coord_loc[1]].values
            r_m = coord_curr_copy[coord_curr_copy['id']==H + 1].iloc[:,coord_loc[0]:coord_loc[1]].values
            vec_b_nm = r_m - r_n
            vec_b_nm=vec_b_nm[0]
            sum_iter=int(dp/2)-1
            for index in range(sum_iter):
                left_index = H - index
                right_index = H + index
                r_flt = coord_curr_copy[coord_curr_copy['id']==left_index].iloc[:,coord_loc[0]:coord_loc[1]].values # first left term
                r_slt = coord_curr_copy[coord_curr_copy['id']==left_index+1].iloc[:,coord_loc[0]:coord_loc[1]].values # second left term
                b_left = r_slt - r_flt
                r_frt = coord_curr_copy[coord_curr_copy['id']==right_index].iloc[:,coord_loc[0]:coord_loc[1]].values # first right term
                r_srt = coord_curr_copy[coord_curr_copy['id']==right_index+1].iloc[:,coord_loc[0]:coord_loc[1]].values # second right term
                b_right = r_srt - r_frt
                b_left=b_left[0]
                b_right=b_right[0]
                curr_plength=np.dot(vec_b_nm,b_left)+np.dot(vec_b_nm,b_right)
                persis_length_list.append(curr_plength)
            #r_end = coord_curr_copy[coord_curr_copy['id']==id_end].iloc[:,coord_loc[0]:coord_loc[1]].values
            #vec_ee = r_end - r_1
            #vec_ee=vec_ee[0]
            #persis_length_outer_list[outer_index]=np.dot(vec_b_nm,vec_ee)/rms_bond_length
            #cos_sim = np.dot(vec_b_nm, vec_ee)/(np.linalg.norm(vec_b_nm)*np.linalg.norm(vec_ee))
            #print(cos_sim*180/np.pi)
            persis_length_outer_list[outer_index]=(np.sum(persis_length_list))/(2*rms_bond_length)
            outer_index+=1
    lp=np.mean(persis_length_outer_list)
    return lp, persis_length_outer_list

def hydro_helper(dp,id_start,coord_curr_copy,coord_loc):
    hydro_radius=0
    for outer_index in range(dp):
        for inner_index in range(dp):
            if outer_index!=inner_index:
                id_r1=id_start + outer_index
                id_r2=id_start + inner_index
                r_1 = coord_curr_copy[coord_curr_copy['id']==id_r1].iloc[:,coord_loc[0]:coord_loc[1]].values
                r_2 = coord_curr_copy[coord_curr_copy['id']==id_r2].iloc[:,coord_loc[0]:coord_loc[1]].values
                r_12 = r_2 - r_1
                r_12=np.linalg.norm(r_12[0])
                if r_12!=0:
                    r_inverse12=np.power(r_12,(-1))
                hydro_radius+=r_inverse12
    return hydro_radius

def hydrodynamic_radius(simname,topologyfilename,coord_loc=[3,6],sidechain=False):
    coords, bs = extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    outermost_index=0
    all_chains=list(coords['timestep_0'].mol.unique())
    if sidechain:
        bondids = get_bonds_atomids(topologyfilename)
    hydro_outer_list=[0]*256
    key='timestep_0'
    for chain in all_chains:
        coord_curr=coords[key][coords[key]['mol']==chain]
        dp = len(coord_curr)
        id_start=(dp)*(chain-1)+1
        id_end = (chain)*(dp)
        if sidechain:
            coord_curr_copy=get_new_chain_coords_forsidechainmodel(coord_curr,bs,bondids)
        else:
            coord_curr_copy = get_new_chain_coords(coord_curr, bs)
        hydro_radius=hydro_helper(dp,id_start,coord_curr_copy,coord_loc)
        hydro_outer_list[outermost_index]=np.power((hydro_radius/(dp**2)),-1)
        outermost_index+=1
    hydro=np.mean(hydro_outer_list)
    return hydro, hydro_outer_list

def cluster_analysis(simname,cutoff,coords_loc=[3,6]):
    """Read a LAMMPS trajectory (extract_unwrapped) based on the given filename 
    (first timestep/snapshot only) and returns the number of clusters and size
    of clusters at the first snapshot. Very slow -- might take couple of hours
    depending on size of the system.
    Args:
    simname(str): Name of the simulation file
    cutoff (float): Minimum distance between a member of a cluster and at least
    one another member of the cluster.

    Returns:
    cluster (dict): A dictionary that contains clusters as keys and list of 
    atomIDs in that specific cluster as values.

    Reference: No reference.
    """
    coord, bs=extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    current_timestep = coord['timestep_0']
    cluster_set = {}
    cluster_index=0
    type2=current_timestep[current_timestep['type']==2].values 
    type3=current_timestep[current_timestep['type']==3].values 
    sidehx = bs[0]/2
    sidehy = bs[1]/2
    sidehz = bs[2]/2
    ## Find distances between positive and negative charges
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

    Reference: No reference.
    """
    coord, bs=extract.extract_unwrapped(simname,first_only=True,boxsize=True)
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
    ### Find distances between positive and negative charges.
    for pindex, pcharge in enumerate(type2):
        map_patomids[pindex] = pcharge[0]
        for nindex, ncharge in enumerate(type3):
            map_natomids[nindex] = ncharge[0]
            charge_distances[pindex,nindex] = periodic_distance(pcharge[3],pcharge[4],pcharge[5],
                              ncharge[3],ncharge[4],ncharge[5],
                              sidehx,sidehy,sidehz,bs)
    index_distances={}
    cluster={}
    ### Find the indices where the distances are less than cutoff.
    for index,row in enumerate(charge_distances):
        index_distances[index] = np.where(row<cutoff)
    ### Make clusters for the indices and atomIDs to those clusters
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
    ### Remove duplicate atomids in each row and sort them.
    for key in cluster:
        cluster[key]=np.unique(cluster[key])
    random_cals=0
    ### Make random calls to two indices of the clusters and check if they have
    ### any atomid in common. If they do, merge the clusters and delete one of them.
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
    ### Remove duplicates in same row and sort
    for key in cluster:
        cluster[key]=np.unique(cluster[key])
    ### Find the mean size of the cluster and print it on the console.
    cluster_copy=cluster.copy()
    mean_size=0
    for key in cluster_copy:
        mean_size+=len(cluster_copy[key])
    print(mean_size/len(cluster_copy))
    return cluster_copy

def prolateness_parameter(simname,topologyfilename,coords_loc=[3,6],sidechain=False):
    """Compute the Prolateness parameter and Asphericity of a given polymer. Eigenvalues of the
    gyration polymer are used to compute the shape parameter.

    Args:
    simname (str): Name of the simulation

    Returns:
    A_s (list): List containing asphericity parameter of every chain.
    P_s (list): List containing prolateness parameter of every chain.

    Reference: 
    1. Arkın, H., & Janke, W. (2013). Gyration tensor based analysis of the shapes of 
    polymer chains in an attractive spherical cage. The Journal of chemical 
    physics, 138(5), 054904. 
    https://aip.scitation.org/doi/abs/10.1063/1.4788616?journalCode=jcp
    2. Ma, B., Nguyen, T. D., & Olvera de la Cruz, M. (2019). Control of Ionic Mobility
    via Charge Size Asymmetry in Random Ionomers. Nano letters, 20(1), 43-49.
    https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.9b02743
    """
    coord, bs=extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    all_chains=list(coord['timestep_0'].mol.unique())
    coord=coord['timestep_0']
    sidehx = bs[0]/2
    sidehy = bs[1]/2
    sidehz = bs[2]/2
    all_gyration_tensors={}
    P_s={}
    A_s={}
    if sidechain:
        bondids = get_bonds_atomids(topologyfilename)
    #for index, key in enumerate(coord):
    for index,chain in enumerate(all_chains):
        coord_curr=coord[coord['mol']==chain]
        dp = len(coord_curr)
        if sidechain:
            coord_curr_copy=get_new_chain_coords_forsidechainmodel(coord_curr,bs,bondids)
        else:
            coord_curr_copy = get_new_chain_coords(coord_curr, bs)
        all_gyration_tensors[index]=np.zeros((3,3))
        coord_curr_copy=coord_curr_copy.iloc[:,coords_loc[0]:coords_loc[1]].values
        xcom,ycom,zcom=np.mean(coord_curr_copy,axis=0) #Find the center of mass of the cluster
        ### Find the gyration tensor of the cluster and compute its eigenvalues and use the eigenvalues
        ### to compute the asphericity parameter. 
        for element in coord_curr_copy:
            dx = element[0] - xcom
            dy = element[1] - ycom
            dz = element[2] - zcom
            all_gyration_tensors[index][0,0]+=(dx)**2
            all_gyration_tensors[index][0,1]+=(dx)*(dy)
            all_gyration_tensors[index][0,2]+=(dx)*(dz)
            all_gyration_tensors[index][1,0]+=(dx)*(dy)
            all_gyration_tensors[index][1,1]+=(dy)**2
            all_gyration_tensors[index][1,2]+=(dy)*(dz)
            all_gyration_tensors[index][2,0]+=(dx)*(dz)
            all_gyration_tensors[index][2,1]+=(dy)*(dz)
            all_gyration_tensors[index][2,2]+=(dz)**2
        all_gyration_tensors[index]=all_gyration_tensors[index]/128
        w, v = np.linalg.eig(all_gyration_tensors[index])
        l_bar=(w[0]+w[1]+w[2])/3
        P_s[index]= ((w[0]-l_bar)*(w[1]-l_bar)*(w[2]-l_bar))/(l_bar**3)
        A_s[index]= ((w[0] - w[1])**2 + (w[1]-w[2])**2 + (w[2]-w[0])**2)/(2*(w[0]**2 +w[1]**2+w[2]**2))
    return list(P_s.values()),list(A_s.values())

def asphericity_parameter(cluster,simname,coords_loc=[3,6]):
    """Compute the asphericity parameter of a given cluster. Eigenvalues of the
    gyration tensor are used to compute the shape parameter.

    Args:
    cluster (dict): A dictionary containing isolated clusters (atomids) as values
    simname (str): Name of the simulation

    Returns:
    A_s (list): List containing asphericity parameter of each of the clusters passed
    through the cluster dictionary

    Reference: 
    1. Arkın, H., & Janke, W. (2013). Gyration tensor based analysis of the shapes of 
    polymer chains in an attractive spherical cage. The Journal of chemical 
    physics, 138(5), 054904. 
    https://aip.scitation.org/doi/abs/10.1063/1.4788616?journalCode=jcp
    2. Ma, B., Nguyen, T. D., & Olvera de la Cruz, M. (2019). Control of Ionic Mobility
    via Charge Size Asymmetry in Random Ionomers. Nano letters, 20(1), 43-49.
    https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.9b02743
    """
    coord, bs=extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    coord=coord['timestep_0']
    sidehx = bs[0]/2
    sidehy = bs[1]/2
    sidehz = bs[2]/2
    all_gyration_tensors={}
    A_s=[]
    for index, key in enumerate(cluster):
        if len(cluster[key])>2:
            all_gyration_tensors[index]=np.zeros((3,3))
            curr_matrix=np.zeros((len(cluster[key]),3))
            ### Find x, y, z coordinates of beads based on ids present in the cluster
            for inner_index, atomid in enumerate(cluster[key]):
                curr_matrix[inner_index] = coord[coord['id']==atomid].iloc[:,coords_loc[0]:coords_loc[1]].values
            cluster_size = len(curr_matrix)
            reference_bead = curr_matrix[0]
            for outer_element in range(cluster_size):
                dist = np.linalg.norm(reference_bead - curr_matrix[outer_element])
                if dist>sidehx:
                    new_bead_outer = periodic_distance_return_new_coords(curr_matrix[outer_element][0],curr_matrix[outer_element][1],curr_matrix[outer_element][2],
                    reference_bead[0],reference_bead[1],reference_bead[2],sidehx,sidehy,sidehz,bs)
                    curr_matrix[outer_element][0]=new_bead_outer[0]
                    curr_matrix[outer_element][1]=new_bead_outer[1]
                    curr_matrix[outer_element][2]=new_bead_outer[2]
            xcom,ycom,zcom=np.mean(curr_matrix,axis=0) #Find the center of mass of the cluster
            ### Find the gyration tensor of the cluster and compute its eigenvalues and use the eigenvalues
            ### to compute the asphericity parameter. 
            for element in curr_matrix:
                dx = element[0] - xcom
                dy = element[1] - ycom
                dz = element[2] - zcom
                all_gyration_tensors[index][0,0]+=(dx)**2
                all_gyration_tensors[index][0,1]+=(dx)*(dy)
                all_gyration_tensors[index][0,2]+=(dx)*(dz)
                all_gyration_tensors[index][1,0]+=(dx)*(dy)
                all_gyration_tensors[index][1,1]+=(dy)**2
                all_gyration_tensors[index][1,2]+=(dy)*(dz)
                all_gyration_tensors[index][2,0]+=(dx)*(dz)
                all_gyration_tensors[index][2,1]+=(dy)*(dz)
                all_gyration_tensors[index][2,2]+=(dz)**2
            all_gyration_tensors[index]=all_gyration_tensors[index]/len(cluster[key])
            w, v = np.linalg.eig(all_gyration_tensors[index])
            A_s.append(((w[0] - w[1])**2 + (w[1]-w[2])**2 + (w[2]-w[0])**2)/(2*(w[0]**2 +w[1]**2+w[2]**2)))
    return A_s

def eigenvalue_based_rog(simname,coord_loc=[3,6]):
    """Compute radius of gyration based on eigenvalue of gyration tensor. 
    """
    coord, bs=extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    coord=coord['timestep_0']
    sidehx = bs[0]/2
    sidehy = bs[1]/2
    sidehz = bs[2]/2
    all_gyration_tensors={}
    rog={}
    for index, chain in enumerate(list(coord.mol.unique())):
        all_gyration_tensors[index]=np.zeros((3,3))
        curr_matrix=coord[coord['mol']==chain].iloc[:,coord_loc[0]:coord_loc[1]].values            
        xcom,ycom,zcom=np.mean(curr_matrix,axis=0) #Find the center of mass of the cluster
        ### Find the gyration tensor of the cluster and compute its eigenvalues and use the eigenvalues
        ### to compute the asphericity parameter. 
        for element in curr_matrix:
            dx = element[0] - xcom
            dy = element[1] - ycom
            dz = element[2] - zcom
            if (dx < -sidehx):   dx = dx + bs[0]
            if (dx > sidehx):    dx = dx - bs[0]
            if (dy < -sidehy):   dy = dy + bs[1]
            if (dy > sidehy):    dy = dy - bs[1]
            if (dz < -sidehz):   dz = dz + bs[2]
            if (dz > sidehz):    dz = dz - bs[2]
            all_gyration_tensors[index][0,0]+=(dx)**2
            all_gyration_tensors[index][0,1]+=(dx)*(dy)
            all_gyration_tensors[index][0,2]+=(dx)*(dz)
            all_gyration_tensors[index][1,0]+=(dx)*(dy)
            all_gyration_tensors[index][1,1]+=(dy)**2
            all_gyration_tensors[index][1,2]+=(dy)*(dz)
            all_gyration_tensors[index][2,0]+=(dx)*(dz)
            all_gyration_tensors[index][2,1]+=(dy)*(dz)
            all_gyration_tensors[index][2,2]+=(dz)**2
        all_gyration_tensors[index]=all_gyration_tensors[index]/128
        w, v = np.linalg.eig(all_gyration_tensors[index])
        rog[index]=(w[0]+w[1]+w[2])**(0.5)
    return list(rog.values())

def find_pairs(simname,coords_loc=[3,6]):
    """Reads a filename and  returns a dictionary with timesteps as keys
    and the distances between positive and negative beads as values at that
    timestep. Uniqueness in pairs is not maintained (This is the old script
    that was the first step for computing the tracked, cumulative and hypothetical
    charge pairs).

    Args:
    simname (str): filename of the simulation
    
    Returns:
    distances (dict): A dictionary with timesteps as keys and the distances
        between positive and negative beads (magnitude) as values at that timestep
        as a pandas dataframe. 
    distances_vec (dict): A dictionary with timesteps as keys and the distances
        between positive and negative beads (vector) as values at that timestep as 
        a pandas dataframe. 

    Reference: No reference.
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
    """Compute the distance between positive and negative charges. Makes
    sure that uniqueness of pairs is maintained. If a bead/atom is chosen as
    part of a pair, then it is subsequently removed from future searches for
    oppositely charged beads. A pair is deemed a pair if the nearest neighbour
    condition is satisfies for both of the charges. Can be made faster with 
    an array version.

    Args:
    simname (str): Filename of the simulation
    coords_loc (list - default): Column indices of x coordinates and z coordinates.

    Returns:
    matched_pair (dict): Dictionary containing timesteps as keys and ids of charged
    pairs and the distance between those pairs as values. 

    Reference: Kind of similar to lifetime correlation function, but not the same.
    1. Ma, B., Nguyen, T. D., & Olvera de la Cruz, M. (2019). Control of Ionic 
    Mobility via Charge Size Asymmetry in Random Ionomers. Nano letters, 20(1), 43-49.
    https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.9b02743
    """
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

def find_decorrelation_pairs_new(current_timestep,box_l,cutoff,coords_loc=[3,6]):
    """Find unique pairs at the given timestep.
    """
    type2=current_timestep[current_timestep['type']==2].iloc[:,
        coords_loc[0]:coords_loc[1]].values 
    type3=current_timestep[current_timestep['type']==3].iloc[:,
        coords_loc[0]:coords_loc[1]].values 
    distsn=np.zeros((type3.shape[0],1))
    distsp=np.zeros((type2.shape[0],1))
    matched_pair=np.zeros((type2.shape[0],3))
    repeat_checkn = []
    pairs_under_cutoff=0
    nrepeat=0
    matched=0
    unmatched=0
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
                if np.linalg.norm(distAB)<cutoff:
                    pairs_under_cutoff+=1
                matched_pair[pindex] = [int(positivechargeid['id']), int(negativechargeid['id']), np.linalg.norm(distAB)]

        else:
            unmatched+=1
    matched_pair =matched_pair[~np.all(matched_pair == 0, axis=1)]
    print((matched/(matched+unmatched))*100)
    return pairs_under_cutoff

def wrapper_decorrelation_pairs_new(simname,cutoff):
    """
    """
    coord, bs = extract.extract_unwrapped(simname, boxsize_whole=True)
    pairs_under_cutoff=[None]*len(coord)
    index = 0
    for key in coord:
        pairs_under_cutoff[index] = find_decorrelation_pairs_new(coord[key],bs[key],cutoff)
        index+=1
    return pairs_under_cutoff

def find_decorrelation_pairs(simname,cutoff,coords_loc=[3,6]):
    """Find out how many charges have an oppositely charged bead within 
    the given cutoff. I am checking how many negative beads have positive
    beads within the given cutoff. It doesn't matter how many oppositely 
    charged beads are present within the given cutoff. It's a step function.
    Either a bead is present, or it's not present.

    Args: 
    simname (str): Name of the simulation
    cutoff (float): Cutoff to be considered for qualifying beads
    
    Returns:
    correlation_fn_whole (list): List containing correlation pairs.
    Reference: https://pubs.acs.org/doi/10.1021/acs.nanolett.9b02743?goto=supporting-info
    """
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
                distsp[pindex] = periodic_distance(pcharge[0],pcharge[1],pcharge[2],
                                                   ncharge[0],ncharge[1],ncharge[2],
                                                   sidehx,sidehy,sidehz,box_l)
            if np.asarray(np.where(distsp == 0)).shape[1]>0:
                print('something wrong')
            if np.asarray(np.where(distsp<cutoff)).shape[1] > 0:
                correlation_fn[nindex]=1
        correlation_fn_whole[outer_index]=np.sum(correlation_fn)
        outer_index+=1  
    return correlation_fn_whole

def find_decorrelation_pairs_distances(simname,coords_loc=[3,6]):
    """Find the distance between positive and negative beads at 
    every snapshot. Do not use. Bad function. 

    Args: 
    simname (str): Name of the simulation.
    
    Returns:
    all_distances (dict): A dictionary with timesteps as keys that contains
    another dictionary which has indices of negative beads as keys that has 
    a list of distances of all positive beads from that negative bead as the
    value of the dictionary.
    """
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
                distsp[pindex] = periodic_distance(pcharge[0],pcharge[1],pcharge[2],
                                                   ncharge[0],ncharge[1],ncharge[2],
                                                   sidehx,sidehy,sidehz,box_l)
            distance_from_this_ncharge[nindex] = distsp
        all_distances[key] = distance_from_this_ncharge
    return all_distances

def find_charged_pairs_center(coord,coords_loc=[3,6]):
    """Reads in a dictionary which contains timesteps as keys and atom 
    coordinates as values and returns a dictionary with timesteps as keys
    and positions of centers of positive and negative beads. Writes a
    file that can be directly read in VMD to view the centers of charged
    pairs.

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

def radius_of_gyration_squared(coords,mass,bs,topologyfilename,sidechain=False,
hist=False,coord_loc = [3,6]):
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
    sidehx = bs[0]/2
    sidehy = bs[1]/2
    sidehz = bs[2]/2
    if sidechain:
        bondids = get_bonds_atomids(topologyfilename)
    all_chains = list(coords['timestep_0'].mol.unique())
    for key in coords:
        rg=[]
        for chain in all_chains:
            coord_curr=coords[key][coords[key]['mol']==chain]
            if sidechain:
                coord_curr=get_new_chain_coords_forsidechainmodel(coord_curr,bs,bondids)
            else:
                coord_curr=get_new_chain_coords(coord_curr,bs)
            coord_curr=coord_curr.iloc[:,coord_loc[0]:coord_loc[1]].values
            com = np.mean(coord_curr,axis=0)
            dist_from_com=(np.linalg.norm(coord_curr - com))**2
            numer=mass*(np.sum(dist_from_com))
            denom=mass*len(coord_curr)
            rg.append(numer/denom)            
        if hist:
            rog[index] = rg
        else:
            rog[index] = np.sqrt(np.mean(rg))
        index+=1
    return rog

def wrapper_radius_of_gyration_squared(simname,topologyfilename,sidechain=False,mass=1,hist=False):
    """^^
    """
    coord, bs=extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    rog=radius_of_gyration_squared(coord, mass,bs,topologyfilename,sidechain=sidechain,hist=hist)
    return rog

def individual_bead_rend(simname,coords_loc=[3,6],sidechain=False):
    """The end_to_end_distance_squared function is not returning correct
    end to end distance. The reason seems to be the wrapping of the beads
    back into the box (the problem remains with unwrapped coordinates as well).
    This function first finds out the unwrapped coordinate of each bead and then
    computes the end to end distance.
    """
    coords, bs = extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    rend_list=[None]*len(coords)
    sidehx=bs[0]/2
    sidehy=bs[1]/2
    sidehz=bs[2]/2
    all_chains=list(coords['timestep_0'].mol.unique())
    index_outer=0
    for key in coords:
        rend=[]
        for chain in all_chains:
            coord_curr=coords[key][coords[key]['mol']==chain]
            id_start=(len(coord_curr))*(chain-1)+1
            if sidechain:
                id_end = (chain-1)*len(coord_curr) + 89
            else:
                id_end = (chain)*(len(coord_curr))
            r_start=coord_curr[coord_curr['id']==id_start].iloc[:,coords_loc[0]:coords_loc[1]].values[0]
            coord_curr_copy=get_new_chain_coords(coord_curr,bs)
            r_end_copy=coord_curr_copy[coord_curr_copy['id']==id_end].iloc[:,coords_loc[0]:coords_loc[1]].values[0]
            rend.append(np.linalg.norm(r_end_copy-r_start))
        rend_list[index_outer]=rend
        index_outer+=1
    return rend_list

def get_bond_vectors(coord_curr_copy):
    """Based on the given timestep coordinates, find out the bond vectors.
    """
    bond_vectors = np.zeros((len(coord_curr_copy)-1,3))
    for index, element in enumerate(coord_curr_copy.iterrows()):
        if index!=127:
            r_1=coord_curr_copy.iloc[index,3:6].values
            r_2=coord_curr_copy.iloc[index+1,3:6].values
            bond_vectors[index,:]=r_2-r_1
    return bond_vectors

def mean_squared_end_to_end(simname,coords_loc=[3,6]):
    """This function is another way to compute the end to end vector. The method
    is validated. The answers are identical to individual_bead_rend function.
    """
    coords, bs = extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    rend_list=[None]*len(coords)
    all_chains=list(coords['timestep_0'].mol.unique())
    rend_inner_list=[None]*len(all_chains)
    index_outer=0
    for key in coords:
        for chain in all_chains:
            rend=[]
            coord_curr=coords[key][coords[key]['mol']==chain]
            coord_curr_copy=get_new_chain_coords(coord_curr,bs)
            coord_curr_copy=get_bond_vectors(coord_curr_copy)
            for outer_id in range(len(coord_curr_copy)):
                for inner_id in range(len(coord_curr_copy)):
                    rend.append(np.dot(coord_curr_copy[outer_id],coord_curr_copy[inner_id]))
            rend_inner_list[int(chain-1)]=np.sum(rend)
    return rend_inner_list
            
def end_to_end_distance_squared(coords,bs,coords_loc=[3,6],sidechain=False,hist=False):
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
    sidehx = bs[0]/2
    sidehy = bs[1]/2
    sidehz = bs[2]/2
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
            r_start = coord_curr[coord_curr['id']==id_start].iloc[:,coords_loc[0]:coords_loc[1]].values[0]
            r_end = coord_curr[coord_curr['id']==id_end].iloc[:,coords_loc[0]:coords_loc[1]].values[0]
            curr_dist= periodic_distance(r_end[0],r_end[1],r_end[2],
                 r_start[0],r_start[1],r_start[2],
                 sidehx,sidehy,sidehz,bs)
            #curr_dist = np.linalg.norm(r_end-r_start)
            rend.append(curr_dist**2)
            gauss_list[key].append(curr_dist)
        if hist:
            rend_list[index]=rend
        else:
            rend_list[index] = np.sqrt(round(np.mean(rend),3))
        index+=1
    return rend_list, gauss_list

def wrapper_end_squared(simname,hist=False,coords_loc=[3,6]):
    coord, bs=extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    rend=end_to_end_distance_squared(coord, bs,hist=hist)
    return rend

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

def radial_distribution_function_all(coordinates,box_l,simname,coords_loc=[3,6],
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
    fwrite=simname + '_all_beads_rdf.txt' #Filename for writing r and g(r) values
    f=open(fwrite,'w')
    f.write('r \t\t\t g(r)\n')
    for index in range(len(g)): #Write r and g(r) values
        f.write('%s\t%s\n'%(r[index],g[index]))
    f.close()
    return g, r

def radial_distribution_function_pairs(coordinates,box_l,simname,coords_loc=[3,6],
    nhis=200,save=False):
    """Compute the radial distribution function for given coordinates.

    Args:
    coordinates (array): Coordinates (x,y,z)
    nhis (int): Number of bins in histogram

    Returns:
    rdf ()
    """
    coords_p=coordinates[coordinates['type']==2].iloc[:,coords_loc[0]:coords_loc[1]].values #Convert 
        #to numpy array
    coords_n=coordinates[coordinates['type']==3].iloc[:,coords_loc[0]:coords_loc[1]].values 
    npart_p=np.size(coords_p,0) #Total number of particles
    npart_n=np.size(coords_n,0)
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
    for partA in range(npart_p): #Don't loop over the last particle because 
        #we have two loop over the particles
        for partB in range(npart_n): #Start from the next particle to 
            #avoid repetition of neighbor bins
            #Calculate the particle-particle distance
            dx = coords_p[partA][0] - coords_n[partB][0]
            dy = coords_p[partA][1] - coords_n[partB][1]
            dz = coords_p[partA][2] - coords_n[partB][2]
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
                g[ig]=g[ig]+1 #Add two particles to that bin's index 
                #(because it's a pair)
    """Normalize the radial distribution function"""
    npart=npart_p
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
    fwrite=simname + '_pairs_rdf.txt' #Filename for writing r and g(r) values
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
    delk=(2*np.pi)/10000 #Set delta k
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
        integral=np.abs(simps(t1)) #integrate
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

def get_entanglement_length(simname,coords_loc=[3,6],sidechain=False):
    """Compute the entanglement length using the primitive path analysis. All intramolecular
    excluded volume interactions are turned off and intermolecular excluded volume interactions
    are retained. Maximum FENE bond is set at 1.2 \sigma and temperature is cooled to
    0.001 \epsilon/\k_b T while keeping the ends of the polymers fixed (crude summary. check
    reference for complete details)
    
    Args:
    simname (str): Name of the simulation file
    coord_loc (list): Column number of x, y and z coordinates in the lammps trajectory file.

    Reference:
    1. Everaers, R., Sukumaran, S. K., Grest, G. S., Svaneborg, C., Sivasubramanian, A., & Kremer, K. (2004). Rheology and microscopic topology of entangled polymeric liquids. Science, 303(5659), 823-826.
    <link> : https://science.sciencemag.org/content/303/5659/823
    2. Hoy, R. S., & Robbins, M. O. (2005). Effect of equilibration on primitive path analyses of entangled polymers. Physical Review E, 72(6), 061802.
    <link> : https://journals.aps.org/pre/abstract/10.1103/PhysRevE.72.061802
    3. Sukumaran, S. K., Grest, G. S., Kremer, K., & Everaers, R. (2005). Identifying the primitive path mesh in entangled polymer liquids. Journal of Polymer Science Part B: Polymer Physics, 43(8), 917-933.
    <link> : https://onlinelibrary.wiley.com/doi/10.1002/polb.20384
    """
    coord, bs = extract.extract_unwrapped(simname, last_only=True,boxsize=True)
    coordinates = coord['timestep_0']
    rendsq = []
    nc = len(coordinates.mol.unique())
    diff_array=np.array([])
    for chain in range(1,nc+1):
        curr_coordinates=coordinates[coordinates['mol']==chain]
        coord_curr = get_new_chain_coords(curr_coordinates, bs)
        id_start=(len(coord_curr))*(chain-1)+1
        if sidechain:
            id_end = (chain-1)*len(coord_curr) + 89            
        else:
            id_end = (chain)*(len(coord_curr))
        r_start=coord_curr[coord_curr['id']==id_start].iloc[:,coords_loc[0]:coords_loc[1]].values[0]
        r_end=coord_curr[coord_curr['id']==id_end].iloc[:,coords_loc[0]:coords_loc[1]].values[0]
        rendsq.append(np.linalg.norm(r_end-r_start)**2)
        coord_curr = coord_curr.values[:,coords_loc[0]:coords_loc[1]]
        one_row_shifted=coord_curr[1:,:]
        coord_curr = np.delete(coord_curr,-1,0)
        curr_diff_array=coord_curr - one_row_shifted
        if sidechain:
            curr_diff_array = curr_diff_array[:88,:]
        curr_diff_array=np.linalg.norm((curr_diff_array),axis=1)
        diff_array=np.append(diff_array,curr_diff_array)
    b_pp = np.mean(diff_array)
    r_ee = np.mean(rendsq)
    n_e = (r_ee)/((len(coord_curr) - 1)*(b_pp**2))
    return n_e, b_pp, r_ee

def chain_orientation_parameter(curr_coordinates_all,ex,nc,dp,bs,topologyfilename,sidechain=False,
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
    if sidechain:
        bondids = get_bonds_atomids(topologyfilename)
    p2x = [0]*(n_applicable_atoms)
    orient = 0
    outer_index = 0
    for chain in range(1,nc+1):
        curr_coordinates=curr_coordinates_all[curr_coordinates_all['mol']==chain]
        begin=(chain -1)*dp + 2
        if sidechain:
            curr_coordinates=get_new_chain_coords_forsidechainmodel(curr_coordinates,bs,bondids)
            end=(chain-1)*dp + 89
        else:
            curr_coordinates = get_new_chain_coords(curr_coordinates, bs)
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

def get_params(simname,topologyfilename,ppa_simname,nc=256,dp=128,sidechain=False,coord_loc=[3,6],hydro=False):
    coord,bs = extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    rend_v20_3x=rend_32c_v20_3x=individual_bead_rend(simname,sidechain=sidechain)
    rog_v20_3x_com=wrapper_radius_of_gyration_squared(simname,topologyfilename,sidechain=sidechain,hist=True)
    p, a = prolateness_parameter(simname,topologyfilename)
    per_mean, per_list=persistence_length(simname,topologyfilename,sidechain=sidechain)
    cep=0
    cep = chain_entang_helper(coord['timestep_0'],cep,nc,dp,bs,coord_loc,topologyfilename,sidechain=sidechain)
    copx = chain_orientation_parameter(coord['timestep_0'],[1,0,0],nc,dp,bs,topologyfilename,sidechain=sidechain)
    n_e, bpp, r_ee = get_entanglement_length(ppa_simname,sidechain=sidechain)
    if hydro:
        print('computing hydro -- will take a long time')
        rh_mean,rh_list=compute.hydrodynamic_radius(simname)
    print('\n\nMean-squared ROG %.2f'%np.mean(rog_v20_3x_com[0]))
    if hydro:
        print('Hydrodynamic radius %.2f'%rh_mean)
        print('Ratio of hydrodynamic radius to RMS radius of gyration %.2f'%((rh_mean)/(np.sqrt(np.mean(rog_v20_3x_com[0])))))
    print('Mean squared end to end end to end distance %.2f'%np.mean(np.power(rend_v20_3x,2)))
    print('Ratio rend to rg %.2f'%(np.mean(np.power(rend_v20_3x,2))/np.mean(rog_v20_3x_com[0])))
    print('Asphericity %.3f'%np.mean(a))
    print('Prolateness %.3f'%np.mean(p))
    print('Persistence length %.3f'%per_mean)
    print('Chain orientation parameter %.3f'%copx)
    print('Entanglement length %.2f'%n_e)
    print('Chain entanglement parameter %.3f'%(cep/(nc*dp - nc*20)))
    print('\n\nEnd to end distance %.2f'%np.mean(rend_v20_3x))
    print('RMS end to end distance %.2f'%np.sqrt(np.mean(np.power(rend_v20_3x,2))))
    print('RMS ROG %.2f'%np.sqrt(np.mean(rog_v20_3x_com[0])))
    

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

def chain_entang_helper(curr_coordinates_all,entang,nc,dp,bs,coord_loc,topologyfilename,sidechain=False):
    if sidechain:
        bondids = get_bonds_atomids(topologyfilename)
    for key in range(1,nc+1):
        curr_coordinates=curr_coordinates_all[curr_coordinates_all['mol']==key]
        begin=(key -1)*dp + 11
        if sidechain:
            curr_coordinates=get_new_chain_coords_forsidechainmodel(curr_coordinates,bs,bondids)
            end = (key - 1)*dp + 89 - 10
        else:
            end = key*dp - 10 
            curr_coordinates= get_new_chain_coords(curr_coordinates, bs)
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

def bonded_beads_distance(simname,coord_loc=[3,6],first_only=False,save=False,
hist=False,root_mean_square=False):
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
    if hist:
        plt.hist(curr_diff_array)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Bond length',fontsize=20,fontfamily='Serif')
    if save:
        plt.tight_layout()
        plt.savefig(simname +'_bond_length.png',dpi=300)
    if root_mean_square:
        return np.sqrt(np.mean(np.power(diff_array,2)))
    else:
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

def bonded_beads_distance_whole_backbone(simname,first_only=False,coord_loc=[3,6],save=False,root_mean_square=False):
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
    if first_only:
        df, bs =extract.extract_unwrapped(simname,first_only=True,boxsize=True)
    else:
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
            if first_only:
                boxsize = bs
            else:
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
    if root_mean_square:
        return np.sqrt(np.mean(np.power(diff_array,2)))
    else:
        return np.mean(diff_array)

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
    df3,df4=extract.extract_def('CG_256C_128DP_v20_deform_lj_kg_uk_2_x')
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

def tangent_modulus(simname,strain):
    """Compute tangent modulus between 0.1 to 0.4 strain
    
    Args:
    simname (str): filename of the simulation

    Returns:
    slope (float): tangent modulus
    """
    df1,df2=extract.extract_def(simname)
    df1=df1.values
    df2=df2.values
    deformation_along = np.argmax(np.array([np.std(df1[:,1]),
                                    np.std(df1[:,2]),
                                    np.std(df1[:,3])]))
    #for index, element in enumerate(strain):
    #    print(index,element)
    begin=611
    till=2446
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
    df3,df4=extract.extract_def('CG_256C_128DP_v20_deform_lj_kg_uk_3_x')
    df3=df3.values
    strain=df3[:,0]
    for simname in args:
        tm.append(tangent_modulus(simname,strain))
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
    df3,df4=extract.extract_def('CG_256C_128DP_v20_deform_lj_kg_uk_2_x')
    df1=df1.values
    df2=df2.values
    df3=df3.values
    deformation_along = np.argmax(np.array([np.std(df1[:,1]),
                                    np.std(df1[:,2]),
                                    np.std(df1[:,3])]))
    strain=df3[:,0]
    begin=0
    #till=305
    till=303
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
    slope, intercept, r_value, p_value, std_err = stats.linregress(strain_list,transverse_1_list)
    poisson_1=-(transverse_1_list/strain_list)
    poisson_2=-(transverse_2_list/strain_list)
    print(-slope)
    return poisson_1[300], poisson_2[300]

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

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    #plt.plot(x_vals, y_vals, '--',color='C1')
    return x_vals,y_vals

def intersectionpoint(slope,intercept,slope2,intercept2):
    A, B =abline(slope2,intercept2)
    C, D =abline(slope,intercept)
    M = [A[0],B[0]]
    N = [A[1],B[1]]
    O = [C[0],D[0]]
    P = [C[1],D[1]]
    line1 = LineString([M, N])
    line2 = LineString([O, P])
    int_pt = line1.intersection(line2)
    point_of_intersection = int_pt.x, int_pt.y
    return int_pt.x

def intersectionpoint2(xvals,yvals,xvals2,yvals2):
    line1 = LineString(np.column_stack((xvals, yvals)))
    line2 = LineString(np.column_stack((xvals2, yvals2)))
    int_pt = line1.intersection(line2)
    point_of_intersection = int_pt.x, int_pt.y
    return int_pt.x,int_pt.y

def drawline(int_pt):
    y_vals=np.array([0,1,2,3,4])
    a=[int_pt]
    x_vals=np.tile(a,len(y_vals))
    return x_vals, y_vals

def yield_stress(simname,strain):
    plt.figure(figsize=(8,8))
    df1,df2=extract.extract_def(simname)
    df1=df1.values
    df2=df2.values
    deformation_along = np.argmax(np.array([np.std(df1[:,1]),
                                    np.std(df1[:,2]),
                                    np.std(df1[:,3])]))
    curr_stress=df1[:,deformation_along+1]
    plt.figure(figsize=(8,8))
    plt.plot(strain,curr_stress)
    begin=0 # 0.00 strain
    till=196 #0.0322 strain
    slope, intercept, r_value, p_value, std_err = stats.linregress(strain[begin:till],
                                                                    curr_stress[begin:till])
    begin2=1000 #0.163 strain
    till2=2500 #0.408 strain
    slope2, intercept2, r_value, p_value, std_err = stats.linregress(strain[begin2:till2],
                                                                    curr_stress[begin2:till2])
    int_pt=intersectionpoint(slope,intercept,slope2,intercept2)
    x_vals, y_vals=drawline(int_pt)
    intrx,intry=intersectionpoint2(x_vals,y_vals,strain, curr_stress)
    if intrx == int_pt:
        plt.close()
        return intry
    else:
        print('error')

def yield_stress_with_error(*args):
    ys=[]
    df3,df4=extract.extract_def('CG_256C_128DP_v20_deform_lj_kg_uk_2_x')
    df3=df3.values
    strain=df3[:,0]
    for simname in args:
        ys.append(yield_stress(simname,strain))
    ys_mean = np.mean(np.asarray(ys))
    ys_std = np.std(np.asarray(ys))
    ys_se = stats.sem(np.asarray(ys))
    return ys_mean, ys_std, ys_se