
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:07:20 2016

Generates a simple, representative single polymer chain model.

@author: suwonbae

Modified on Fri Jan 18 by @prathameshraiter

Can generate a model of multiple chains 

Instead of a straight line polymer chain, gives out a random walk configuration for the polymer chain.

Model is based on a polymethacrylate functionalized with trimethylammonium (cation) and carboxylate

This file gives different molecule ID for each chain. Cations are half of the chains, and anions are 
rest half of the chains. 

For example, for 100 chains - 50 cations, and 50 anions. 

"""

from math import *
import random
import numpy as np

length_actual= 0.96 # average bond length
length = 1
r_equilibrium = 0.9
mass=1


monomernum=int(input('What is the degree of polymerization of the chain? '))
numberofchains = int(input('How many chains are there in the system? '))
density = float(input('What is the approximate desired initial density? '))
charge_percent=int(input('What is the approximate desired charged monomer percentage? '))
charge=int(input('What is the absolute charge value? '))

charge_divisor=int(100/charge_percent)
############ ONLY for 128DP version ########################
backbone_beads= 89
num_of_charges_per_chain = 13
## 89 + (13x3) = 128
## approximately 10 percent charge

############################################################

numberofchains2 = numberofchains
atomnum=monomernum
bondnum=atomnum-1
anglenum=atomnum-2
dihedralnum=atomnum-3

masses=[]
atoms=[]
bonds=[]
angles=[]
dihedrals=[]
xstart = ystart = zstart =0 
xend = yend = zend = ceil(((monomernum*numberofchains*mass)/density)**(1/3))
print('Box size has been set as {}'.format(xend))


atomtypes=3
bondtypes=1
angletypes=1
dihedraltypes=1
j = 1
masses.append([1, mass])
masses.append([2, mass])
masses.append([3, mass])
count = m = 0 
molID = l = 1
starta = 0
bondID = 1
atomID = 1
coords=[]
r_new = np.array([[xstart,ystart,zstart],[xend,yend,zend]])
H, edges = np.histogramdd(r_new, bins = (ceil((xend - xstart)/2), ceil((yend-ystart)/2), ceil((zend - zstart)/2)))
all_bins={}
print('edges:',edges[0])
def apply_add(bin_value):
    if bin_value==(len(edges[0])-1):
        return 1
    else:
        binned = bin_value+1
        return binned

def apply_sub(bin_value):
    if bin_value==1:
        binned = (len(edges[0]) -1)
        return binned
    else:
        binned = bin_value -1
        return binned

def decide_charge(charge, molID):
    if molID <= numberofchains/2:
        return charge
    else:
        return -charge

def decide_type(charge):
    if charge > 0:
        return 2
    else:
        return 3 

while numberofchains2!=0:
    starting_point_far = False
    index = 0
    while not starting_point_far:
        #print('new guess for starting point')
        xx_temp, yy_temp, zz_temp=random.uniform(xstart,xend), random.uniform(ystart,yend), random.uniform(zstart,zend)
        while not starting_point_far and index < len(coords):
            dist = sqrt( ((coords[index][0]-xx_temp)**2)+((coords[index][1]-yy_temp)**2) +((coords[index][2]-zz_temp)**2))
            if dist < r_equilibrium:
                #print('wrong guess')
                break
            index +=1
        else:
            starting_point_far = True
            #print('correct guess',[xx_temp,yy_temp,zz_temp])
    xx = xx_temp
    yy = yy_temp
    zz = zz_temp
    curr_bin = np.digitize(np.array([xx,yy,zz]),edges[0])
    if str(curr_bin) not in all_bins:
        all_bins[str(curr_bin)]=[]
    all_bins[str(curr_bin)].append([xx,yy,zz])
    '--This part is responsible for the multiple chain--'
    k = 0
    actual_moves = 0 
    #for i in range(starta, starta+atomnum):
    while actual_moves<backbone_beads:
        '--This part creates one chain--'
        i = starta + actual_moves
        phi = np.random.uniform(0,np.pi*2)
        costheta = np.random.uniform(-1,1)
        theta = np.arccos( costheta )
        #### FIND A RANDOM POINT ON A SPHERE OF RADIUS EQUAL TO LENGTH #####
        xtemp = xx + length*(sin(theta)*cos(phi))
        ytemp = yy + length*(sin(theta)*sin(phi))
        ztemp = zz + length*(cos(theta))
        wrap_ref_x=xtemp
        wrap_ref_y=ytemp
        wrap_ref_z=ztemp
        set_curr_point_wrapped=False
        if xtemp > xend:
            xtemp = xtemp - xend
            #print('periodicx',xtemp, wrap_ref_x)
            set_curr_point_wrapped=True
        elif xtemp < xstart:
            xtemp = xend + xtemp
            #print('periodicx',xtemp, wrap_ref_x)
            set_curr_point_wrapped=True
        if ytemp > yend:
            ytemp = ytemp - yend
            #print('periodicy',ytemp, wrap_ref_y)
            set_curr_point_wrapped=True
        elif ytemp < ystart:
            ytemp = yend + ytemp
            #print('periodicy',ytemp, wrap_ref_y)
            set_curr_point_wrapped=True
        if ztemp > zend:
            ztemp = ztemp - zend
            #print('periodicz',ztemp, wrap_ref_z)
            set_curr_point_wrapped=True
        elif ztemp < zstart:
            ztemp = zend + ztemp
            #print('periodicz',ztemp, wrap_ref_z)
            set_curr_point_wrapped=True
        curr_bin = np.digitize(np.array([xtemp,ytemp,ztemp]),edges[0])
        if str(curr_bin) not in all_bins:
            all_bins[str(curr_bin)]=[]
        all_bins[str(curr_bin)].append([xtemp,ytemp,ztemp])
        ############
            #all values are betweeen 1 and last bin
        bins_to_test =[
        [apply_add(curr_bin[0]),curr_bin[1],curr_bin[2]],
        [apply_add(curr_bin[0]),apply_add(curr_bin[1]),curr_bin[2]],
        [apply_add(curr_bin[0]),apply_add(curr_bin[1]),apply_add(curr_bin[2])],
        [apply_add(curr_bin[0]),apply_add(curr_bin[1]),apply_sub(curr_bin[2])],
        [apply_add(curr_bin[0]),curr_bin[1],apply_add(curr_bin[2])],
        [apply_add(curr_bin[0]),curr_bin[1],apply_sub(curr_bin[2])],
        [apply_add(curr_bin[0]),apply_sub(curr_bin[1]),curr_bin[2]],
        [apply_add(curr_bin[0]),apply_sub(curr_bin[1]),apply_add(curr_bin[2])],
        [apply_add(curr_bin[0]),apply_sub(curr_bin[1]),apply_sub(curr_bin[2])],
        [curr_bin[0],apply_add(curr_bin[1]),curr_bin[2]],
        [curr_bin[0],apply_add(curr_bin[1]),apply_add(curr_bin[2])],
        [curr_bin[0],apply_add(curr_bin[1]),apply_sub(curr_bin[2])],
        [curr_bin[0],curr_bin[1],apply_add(curr_bin[2])],
        [curr_bin[0],curr_bin[1],apply_sub(curr_bin[2])],
        [curr_bin[0],apply_sub(curr_bin[1]),curr_bin[2]],
        [curr_bin[0],apply_sub(curr_bin[1]),apply_add(curr_bin[2])],
        [curr_bin[0],apply_sub(curr_bin[1]),apply_sub(curr_bin[2])],
        [apply_sub(curr_bin[0]),curr_bin[1],curr_bin[2]],
        [apply_sub(curr_bin[0]),apply_add(curr_bin[1]),curr_bin[2]],
        [apply_sub(curr_bin[0]),apply_add(curr_bin[1]),apply_add(curr_bin[2])],
        [apply_sub(curr_bin[0]),apply_add(curr_bin[1]),apply_sub(curr_bin[2])],
        [apply_sub(curr_bin[0]),curr_bin[1],apply_add(curr_bin[2])],
        [apply_sub(curr_bin[0]),curr_bin[1],apply_sub(curr_bin[2])],
        [apply_sub(curr_bin[0]),apply_sub(curr_bin[1]),curr_bin[2]],
        [apply_sub(curr_bin[0]),apply_sub(curr_bin[1]),apply_add(curr_bin[2])],
        [apply_sub(curr_bin[0]),apply_sub(curr_bin[1]),apply_sub(curr_bin[2])]
        ]
        all_atoms_far = True
        index=0
        for bin_ids in bins_to_test:
            dict_key_id=str(np.array(bin_ids))
            try:
                while all_atoms_far and index < len(all_bins[dict_key_id]):
                    dist = sqrt(((all_bins[dict_key_id][index][0]-xtemp)**2)+((all_bins[dict_key_id][index][1]-ytemp)**2) +((all_bins[dict_key_id][index][2]-ztemp)**2))
                    #print(dist,'distance from wrapped point')
                    if set_curr_point_wrapped:
                        dist = sqrt(((all_bins[dict_key_id][index][0]-wrap_ref_x)**2)+((all_bins[dict_key_id][index][1]-wrap_ref_y)**2) +((all_bins[dict_key_id][index][2]-wrap_ref_z)**2))
                        #print('distance from unwrapped point',dist)
                    if dist < r_equilibrium:
                        #print('all atoms not far')
                        all_atoms_far = False
                    index+=1
            except KeyError:
                pass
        if not all_atoms_far:
            all_bins[str(curr_bin)].pop()
        if all_atoms_far:
            xx = xtemp
            yy = ytemp
            zz = ztemp
            atoms.append([atomID, molID, 1, 0, xx, yy, zz])
            coords.append([xx,yy,zz])
            k+=1
            ## atomID = i+1
            if actual_moves==0:
                current_bead= atomID - 1
                final_bead = atomID + 5
            if actual_moves!=backbone_beads-1:
                bonds.append([bondID,1,i+1, i+2])
                bondID+=1
            if ((100*(i+1))/(numberofchains*monomernum))%5==0:
                print('{0:.0f} % work done'.format((i/(numberofchains*monomernum))*100))
            actual_moves+=1
            atomID+=1
    ### sidebead is bead, num_of_charges_per_chain is chargenum
    for sidebead in range(num_of_charges_per_chain):
        choice_of_bead=random.randint(current_bead,final_bead)
        chosen_bead_ID=atoms[choice_of_bead][0]
        chosen_bead_xx=atoms[choice_of_bead][4]
        chosen_bead_yy=atoms[choice_of_bead][5]
        chosen_bead_zz=atoms[choice_of_bead][6]
        first_bead_not_found=True
        second_bead_not_found=True
        third_bead_not_found=True
        
        while first_bead_not_found:    
            phi = np.random.uniform(0,np.pi*2)
            costheta = np.random.uniform(-1,1)
            theta = np.arccos( costheta )
            #### FIND A RANDOM POINT ON A SPHERE OF RADIUS EQUAL TO LENGTH #####
            xtemp = chosen_bead_xx + length*(sin(theta)*cos(phi))
            ytemp = chosen_bead_yy + length*(sin(theta)*sin(phi))
            ztemp = chosen_bead_zz + length*(cos(theta))
            wrap_ref_x=xtemp
            wrap_ref_y=ytemp
            wrap_ref_z=ztemp
            set_curr_point_wrapped=False
            if xtemp > xend:
                xtemp = xtemp - xend
                #print('periodicx',xtemp, wrap_ref_x)
                set_curr_point_wrapped=True
            elif xtemp < xstart:
                xtemp = xend + xtemp
                #print('periodicx',xtemp, wrap_ref_x)
                set_curr_point_wrapped=True
            if ytemp > yend:
                ytemp = ytemp - yend
                #print('periodicy',ytemp, wrap_ref_y)
                set_curr_point_wrapped=True
            elif ytemp < ystart:
                ytemp = yend + ytemp
                #print('periodicy',ytemp, wrap_ref_y)
                set_curr_point_wrapped=True
            if ztemp > zend:
                ztemp = ztemp - zend
                #print('periodicz',ztemp, wrap_ref_z)
                set_curr_point_wrapped=True
            elif ztemp < zstart:
                ztemp = zend + ztemp
                #print('periodicz',ztemp, wrap_ref_z)
                set_curr_point_wrapped=True
            curr_bin = np.digitize(np.array([xtemp,ytemp,ztemp]),edges[0])
            if str(curr_bin) not in all_bins:
                all_bins[str(curr_bin)]=[]
            all_bins[str(curr_bin)].append([xtemp,ytemp,ztemp])
            bins_to_test =[
            [apply_add(curr_bin[0]),curr_bin[1],curr_bin[2]],
            [apply_add(curr_bin[0]),apply_add(curr_bin[1]),curr_bin[2]],
            [apply_add(curr_bin[0]),apply_add(curr_bin[1]),apply_add(curr_bin[2])],
            [apply_add(curr_bin[0]),apply_add(curr_bin[1]),apply_sub(curr_bin[2])],
            [apply_add(curr_bin[0]),curr_bin[1],apply_add(curr_bin[2])],
            [apply_add(curr_bin[0]),curr_bin[1],apply_sub(curr_bin[2])],
            [apply_add(curr_bin[0]),apply_sub(curr_bin[1]),curr_bin[2]],
            [apply_add(curr_bin[0]),apply_sub(curr_bin[1]),apply_add(curr_bin[2])],
            [apply_add(curr_bin[0]),apply_sub(curr_bin[1]),apply_sub(curr_bin[2])],
            [curr_bin[0],apply_add(curr_bin[1]),curr_bin[2]],
            [curr_bin[0],apply_add(curr_bin[1]),apply_add(curr_bin[2])],
            [curr_bin[0],apply_add(curr_bin[1]),apply_sub(curr_bin[2])],
            [curr_bin[0],curr_bin[1],apply_add(curr_bin[2])],
            [curr_bin[0],curr_bin[1],apply_sub(curr_bin[2])],
            [curr_bin[0],apply_sub(curr_bin[1]),curr_bin[2]],
            [curr_bin[0],apply_sub(curr_bin[1]),apply_add(curr_bin[2])],
            [curr_bin[0],apply_sub(curr_bin[1]),apply_sub(curr_bin[2])],
            [apply_sub(curr_bin[0]),curr_bin[1],curr_bin[2]],
            [apply_sub(curr_bin[0]),apply_add(curr_bin[1]),curr_bin[2]],
            [apply_sub(curr_bin[0]),apply_add(curr_bin[1]),apply_add(curr_bin[2])],
            [apply_sub(curr_bin[0]),apply_add(curr_bin[1]),apply_sub(curr_bin[2])],
            [apply_sub(curr_bin[0]),curr_bin[1],apply_add(curr_bin[2])],
            [apply_sub(curr_bin[0]),curr_bin[1],apply_sub(curr_bin[2])],
            [apply_sub(curr_bin[0]),apply_sub(curr_bin[1]),curr_bin[2]],
            [apply_sub(curr_bin[0]),apply_sub(curr_bin[1]),apply_add(curr_bin[2])],
            [apply_sub(curr_bin[0]),apply_sub(curr_bin[1]),apply_sub(curr_bin[2])]
            ]
            all_atoms_far = True
            index=0
            for bin_ids in bins_to_test:
                dict_key_id=str(np.array(bin_ids))
                try:
                    while all_atoms_far and index < len(all_bins[dict_key_id]):
                        dist = sqrt(((all_bins[dict_key_id][index][0]-xtemp)**2)+((all_bins[dict_key_id][index][1]-ytemp)**2) +((all_bins[dict_key_id][index][2]-ztemp)**2))
                        #print(dist,'distance from wrapped point')
                        if set_curr_point_wrapped:
                            dist = sqrt(((all_bins[dict_key_id][index][0]-wrap_ref_x)**2)+((all_bins[dict_key_id][index][1]-wrap_ref_y)**2) +((all_bins[dict_key_id][index][2]-wrap_ref_z)**2))
                            #print('distance from unwrapped point',dist)
                        if dist < r_equilibrium:
                            #print('all atoms not far')
                            all_atoms_far = False
                        index+=1
                except KeyError:
                    pass
            if not all_atoms_far:
                all_bins[str(curr_bin)].pop()
            if all_atoms_far:
                new_first_bead_xx = xtemp
                new_first_bead_yy = ytemp
                new_first_bead_zz = ztemp
                coords.append([new_first_bead_xx , new_first_bead_yy, new_first_bead_zz])
                first_bead_not_found = False
        
        while second_bead_not_found:    
            phi = np.random.uniform(0,np.pi*2)
            costheta = np.random.uniform(-1,1)
            theta = np.arccos( costheta )
            #### FIND A RANDOM POINT ON A SPHERE OF RADIUS EQUAL TO LENGTH #####
            xtemp = new_first_bead_xx + length*(sin(theta)*cos(phi))
            ytemp = new_first_bead_yy + length*(sin(theta)*sin(phi))
            ztemp = new_first_bead_zz + length*(cos(theta))
            wrap_ref_x=xtemp
            wrap_ref_y=ytemp
            wrap_ref_z=ztemp
            set_curr_point_wrapped=False
            if xtemp > xend:
                xtemp = xtemp - xend
                #print('periodicx',xtemp, wrap_ref_x)
                set_curr_point_wrapped=True
            elif xtemp < xstart:
                xtemp = xend + xtemp
                #print('periodicx',xtemp, wrap_ref_x)
                set_curr_point_wrapped=True
            if ytemp > yend:
                ytemp = ytemp - yend
                #print('periodicy',ytemp, wrap_ref_y)
                set_curr_point_wrapped=True
            elif ytemp < ystart:
                ytemp = yend + ytemp
                #print('periodicy',ytemp, wrap_ref_y)
                set_curr_point_wrapped=True
            if ztemp > zend:
                ztemp = ztemp - zend
                #print('periodicz',ztemp, wrap_ref_z)
                set_curr_point_wrapped=True
            elif ztemp < zstart:
                ztemp = zend + ztemp
                #print('periodicz',ztemp, wrap_ref_z)
                set_curr_point_wrapped=True
            curr_bin = np.digitize(np.array([xtemp,ytemp,ztemp]),edges[0])
            if str(curr_bin) not in all_bins:
                all_bins[str(curr_bin)]=[]
            all_bins[str(curr_bin)].append([xtemp,ytemp,ztemp])
            bins_to_test =[
            [apply_add(curr_bin[0]),curr_bin[1],curr_bin[2]],
            [apply_add(curr_bin[0]),apply_add(curr_bin[1]),curr_bin[2]],
            [apply_add(curr_bin[0]),apply_add(curr_bin[1]),apply_add(curr_bin[2])],
            [apply_add(curr_bin[0]),apply_add(curr_bin[1]),apply_sub(curr_bin[2])],
            [apply_add(curr_bin[0]),curr_bin[1],apply_add(curr_bin[2])],
            [apply_add(curr_bin[0]),curr_bin[1],apply_sub(curr_bin[2])],
            [apply_add(curr_bin[0]),apply_sub(curr_bin[1]),curr_bin[2]],
            [apply_add(curr_bin[0]),apply_sub(curr_bin[1]),apply_add(curr_bin[2])],
            [apply_add(curr_bin[0]),apply_sub(curr_bin[1]),apply_sub(curr_bin[2])],
            [curr_bin[0],apply_add(curr_bin[1]),curr_bin[2]],
            [curr_bin[0],apply_add(curr_bin[1]),apply_add(curr_bin[2])],
            [curr_bin[0],apply_add(curr_bin[1]),apply_sub(curr_bin[2])],
            [curr_bin[0],curr_bin[1],apply_add(curr_bin[2])],
            [curr_bin[0],curr_bin[1],apply_sub(curr_bin[2])],
            [curr_bin[0],apply_sub(curr_bin[1]),curr_bin[2]],
            [curr_bin[0],apply_sub(curr_bin[1]),apply_add(curr_bin[2])],
            [curr_bin[0],apply_sub(curr_bin[1]),apply_sub(curr_bin[2])],
            [apply_sub(curr_bin[0]),curr_bin[1],curr_bin[2]],
            [apply_sub(curr_bin[0]),apply_add(curr_bin[1]),curr_bin[2]],
            [apply_sub(curr_bin[0]),apply_add(curr_bin[1]),apply_add(curr_bin[2])],
            [apply_sub(curr_bin[0]),apply_add(curr_bin[1]),apply_sub(curr_bin[2])],
            [apply_sub(curr_bin[0]),curr_bin[1],apply_add(curr_bin[2])],
            [apply_sub(curr_bin[0]),curr_bin[1],apply_sub(curr_bin[2])],
            [apply_sub(curr_bin[0]),apply_sub(curr_bin[1]),curr_bin[2]],
            [apply_sub(curr_bin[0]),apply_sub(curr_bin[1]),apply_add(curr_bin[2])],
            [apply_sub(curr_bin[0]),apply_sub(curr_bin[1]),apply_sub(curr_bin[2])]
            ]
            all_atoms_far = True
            index=0
            for bin_ids in bins_to_test:
                dict_key_id=str(np.array(bin_ids))
                try:
                    while all_atoms_far and index < len(all_bins[dict_key_id]):
                        dist = sqrt(((all_bins[dict_key_id][index][0]-xtemp)**2)+((all_bins[dict_key_id][index][1]-ytemp)**2) +((all_bins[dict_key_id][index][2]-ztemp)**2))
                        #print(dist,'distance from wrapped point')
                        if set_curr_point_wrapped:
                            dist = sqrt(((all_bins[dict_key_id][index][0]-wrap_ref_x)**2)+((all_bins[dict_key_id][index][1]-wrap_ref_y)**2) +((all_bins[dict_key_id][index][2]-wrap_ref_z)**2))
                            #print('distance from unwrapped point',dist)
                        if dist < r_equilibrium:
                            #print('all atoms not far')
                            all_atoms_far = False
                        index+=1
                except KeyError:
                    pass
            if not all_atoms_far:
                all_bins[str(curr_bin)].pop()
            if all_atoms_far:
                new_second_bead_xx = xtemp
                new_second_bead_yy = ytemp
                new_second_bead_zz = ztemp
                coords.append([new_second_bead_xx , new_second_bead_yy, new_second_bead_zz])
                second_bead_not_found = False
        
        while third_bead_not_found:    
            phi = np.random.uniform(0,np.pi*2)
            costheta = np.random.uniform(-1,1)
            theta = np.arccos( costheta )
            #### FIND A RANDOM POINT ON A SPHERE OF RADIUS EQUAL TO LENGTH #####
            xtemp = new_second_bead_xx + length*(sin(theta)*cos(phi))
            ytemp = new_second_bead_yy + length*(sin(theta)*sin(phi))
            ztemp = new_second_bead_zz + length*(cos(theta))
            wrap_ref_x=xtemp
            wrap_ref_y=ytemp
            wrap_ref_z=ztemp
            set_curr_point_wrapped=False
            if xtemp > xend:
                xtemp = xtemp - xend
                #print('periodicx',xtemp, wrap_ref_x)
                set_curr_point_wrapped=True
            elif xtemp < xstart:
                xtemp = xend + xtemp
                #print('periodicx',xtemp, wrap_ref_x)
                set_curr_point_wrapped=True
            if ytemp > yend:
                ytemp = ytemp - yend
                #print('periodicy',ytemp, wrap_ref_y)
                set_curr_point_wrapped=True
            elif ytemp < ystart:
                ytemp = yend + ytemp
                #print('periodicy',ytemp, wrap_ref_y)
                set_curr_point_wrapped=True
            if ztemp > zend:
                ztemp = ztemp - zend
                #print('periodicz',ztemp, wrap_ref_z)
                set_curr_point_wrapped=True
            elif ztemp < zstart:
                ztemp = zend + ztemp
                #print('periodicz',ztemp, wrap_ref_z)
                set_curr_point_wrapped=True
            curr_bin = np.digitize(np.array([xtemp,ytemp,ztemp]),edges[0])
            if str(curr_bin) not in all_bins:
                all_bins[str(curr_bin)]=[]
            all_bins[str(curr_bin)].append([xtemp,ytemp,ztemp])
            bins_to_test =[
            [apply_add(curr_bin[0]),curr_bin[1],curr_bin[2]],
            [apply_add(curr_bin[0]),apply_add(curr_bin[1]),curr_bin[2]],
            [apply_add(curr_bin[0]),apply_add(curr_bin[1]),apply_add(curr_bin[2])],
            [apply_add(curr_bin[0]),apply_add(curr_bin[1]),apply_sub(curr_bin[2])],
            [apply_add(curr_bin[0]),curr_bin[1],apply_add(curr_bin[2])],
            [apply_add(curr_bin[0]),curr_bin[1],apply_sub(curr_bin[2])],
            [apply_add(curr_bin[0]),apply_sub(curr_bin[1]),curr_bin[2]],
            [apply_add(curr_bin[0]),apply_sub(curr_bin[1]),apply_add(curr_bin[2])],
            [apply_add(curr_bin[0]),apply_sub(curr_bin[1]),apply_sub(curr_bin[2])],
            [curr_bin[0],apply_add(curr_bin[1]),curr_bin[2]],
            [curr_bin[0],apply_add(curr_bin[1]),apply_add(curr_bin[2])],
            [curr_bin[0],apply_add(curr_bin[1]),apply_sub(curr_bin[2])],
            [curr_bin[0],curr_bin[1],apply_add(curr_bin[2])],
            [curr_bin[0],curr_bin[1],apply_sub(curr_bin[2])],
            [curr_bin[0],apply_sub(curr_bin[1]),curr_bin[2]],
            [curr_bin[0],apply_sub(curr_bin[1]),apply_add(curr_bin[2])],
            [curr_bin[0],apply_sub(curr_bin[1]),apply_sub(curr_bin[2])],
            [apply_sub(curr_bin[0]),curr_bin[1],curr_bin[2]],
            [apply_sub(curr_bin[0]),apply_add(curr_bin[1]),curr_bin[2]],
            [apply_sub(curr_bin[0]),apply_add(curr_bin[1]),apply_add(curr_bin[2])],
            [apply_sub(curr_bin[0]),apply_add(curr_bin[1]),apply_sub(curr_bin[2])],
            [apply_sub(curr_bin[0]),curr_bin[1],apply_add(curr_bin[2])],
            [apply_sub(curr_bin[0]),curr_bin[1],apply_sub(curr_bin[2])],
            [apply_sub(curr_bin[0]),apply_sub(curr_bin[1]),curr_bin[2]],
            [apply_sub(curr_bin[0]),apply_sub(curr_bin[1]),apply_add(curr_bin[2])],
            [apply_sub(curr_bin[0]),apply_sub(curr_bin[1]),apply_sub(curr_bin[2])]
            ]
            all_atoms_far = True
            index=0
            for bin_ids in bins_to_test:
                dict_key_id=str(np.array(bin_ids))
                try:
                    while all_atoms_far and index < len(all_bins[dict_key_id]):
                        dist = sqrt(((all_bins[dict_key_id][index][0]-xtemp)**2)+((all_bins[dict_key_id][index][1]-ytemp)**2) +((all_bins[dict_key_id][index][2]-ztemp)**2))
                        #print(dist,'distance from wrapped point')
                        if set_curr_point_wrapped:
                            dist = sqrt(((all_bins[dict_key_id][index][0]-wrap_ref_x)**2)+((all_bins[dict_key_id][index][1]-wrap_ref_y)**2) +((all_bins[dict_key_id][index][2]-wrap_ref_z)**2))
                            #print('distance from unwrapped point',dist)
                        if dist < r_equilibrium:
                            #print('all atoms not far')
                            all_atoms_far = False
                        index+=1
                except KeyError:
                    pass
            if not all_atoms_far:
                all_bins[str(curr_bin)].pop()
            if all_atoms_far:
                new_third_bead_xx = xtemp
                new_third_bead_yy = ytemp
                new_third_bead_zz = ztemp
                coords.append([new_third_bead_xx , new_third_bead_yy, new_third_bead_zz])
                third_bead_not_found = False
        atoms.append([atomID,molID,1,0,new_first_bead_xx,new_first_bead_yy,new_first_bead_zz])
        bonds.append([bondID,1,chosen_bead_ID,atomID])
        atomID+=1
        bondID+=1
        atoms.append([atomID,molID,1,0,new_second_bead_xx,new_second_bead_yy,new_second_bead_zz])
        bonds.append([bondID,1,atomID-1,atomID])
        atomID+=1
        bondID+=1
        atoms.append([atomID,molID,decide_type(decide_charge(charge,molID)),decide_charge(charge, molID),new_third_bead_xx,new_third_bead_yy,new_third_bead_zz])
        bonds.append([bondID,1,atomID-1,atomID])
        atomID+=1
        bondID+=1
        if sidebead == num_of_charges_per_chain - 2:
            current_bead = final_bead + 1
            final_bead = final_bead + 4
        else:
            current_bead = final_bead + 1
            final_bead = final_bead + 7

    numberofchains2-=1
    starta = atomnum*j
    j+=1
    molID+=1

xcord = []
ycord = []
zcord = []
for line in atoms:
    xcord.append(line[4])
    ycord.append(line[5])
    zcord.append(line[6])

lo = int(min(min(xcord),min(ycord),min(zcord)) - 1)
hi = int(max(max(xcord),max(ycord),max(zcord)) + 1)

f=open('CG_%sC_%sDP_d0p%s_cp%s_cv%s_with3sidebeads.data' %(numberofchains,monomernum,int(density*100),charge_percent,charge),'w')
f.write('LAMMPS data file for Random Walk Polymer\n\n')
f.write('%s atoms\n' % (atomnum*numberofchains))
f.write('%s atom types\n' % (atomtypes))
f.write('%s bonds\n' % (bondnum*numberofchains))
f.write('%s bond types\n' % (bondtypes))
f.write('%s angles\n' % (anglenum*numberofchains))
f.write('%s angle types\n' % (angletypes))
f.write('%s dihedrals\n' % (dihedralnum*numberofchains))
f.write('%s dihedral types\n\n' % (dihedraltypes))
f.write('%s %s xlo xhi\n' % (lo, hi))
f.write('%s %s ylo yhi\n' % (lo, hi))
f.write('%s %s zlo zhi\n\n' % (lo, hi))
f.write('Masses\n\n')
for line in masses:
    f.write('%s %s\n' % (line[0], line[1]))
f.write('\n')
f.write('Atoms\n\n')
for line in atoms:
    f.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (line[0], line[1], line[2], line[3], line[4], line[5], line[6]))
f.write('\n')
f.write('Bonds\n\n')
for line in bonds:
    f.write('%s\t%s\t%s\t%s\n' % (line[0], line[1], line[2], line[3]))
f.close()
