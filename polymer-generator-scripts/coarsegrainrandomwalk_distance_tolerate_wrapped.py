
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
density = float(input('What is the desired approximate initial density? '))
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
molID = l = n = r = 1
o = 2
p = 3
q = 4
starta = startb = startn = startd = 0
coords=[]
while numberofchains2!=0:
    starting_point_far = False
    index = 0
    while not starting_point_far:
        print('new guess for starting point')
        xx_temp, yy_temp, zz_temp=random.randrange(xstart,xend), random.randrange(ystart,yend), random.randrange(zstart,zend)
        while not starting_point_far and index < len(coords):
            dist = sqrt( ((coords[index][0]-xx_temp)**2)+((coords[index][1]-yy_temp)**2) +((coords[index][2]-zz_temp)**2))
            if dist < r_equilibrium:
                print('wrong guess')
                break
            index +=1
        else:
            starting_point_far = True
            print('correct guess')
    xx = xx_temp
    yy = yy_temp
    zz = zz_temp
    '--This part is responsible for the multiple chain--'
    k = 0
    actual_moves = 0 
    #for i in range(starta, starta+atomnum):
    while actual_moves<atomnum:
        '--This part creates one chain--'
        i = starta + actual_moves
        phi = np.random.uniform(0,np.pi*2)
        costheta = np.random.uniform(-1,1)
        theta = np.arccos( costheta )
        #### FIND A RANDOM POINT ON A SPHERE OF RADIUS EQUAL TO LENGTH #####
        xtemp = xx + length*(sin(theta)*cos(phi))
        ytemp = yy + length*(sin(theta)*sin(phi))
        ztemp = zz + length*(cos(theta))
        #### WRAP THE NEW COORDINATES IN A FIXED BOX SIZE 
        all_atoms_far = True
        index = 0
        while all_atoms_far and index < len(coords):
            dist = sqrt( ((coords[index][0]-xtemp)**2)+((coords[index][1]-ytemp)**2) +((coords[index][2]-ztemp)**2))
            if dist < r_equilibrium:
                all_atoms_far = False
            index +=1

        if xtemp > xend:
            xtemp = xtemp - xend
        elif xtemp < xstart:
            xtemp = xend + xtemp
        if ytemp > yend:
            ytemp = ytemp - yend
        elif ytemp < ystart:
            ytemp = yend + ytemp
        if ztemp > zend:
            ztemp = ztemp - zend
        elif ztemp < zstart:
            ztemp = zend + ztemp

        index = 0
        
        while all_atoms_far and index < len(coords):
            dist = sqrt( ((coords[index][0]-xtemp)**2)+((coords[index][1]-ytemp)**2) +((coords[index][2]-ztemp)**2))
            if dist < r_equilibrium:
                all_atoms_far = False
            index +=1
            
        if all_atoms_far:
            xx = xtemp
            yy = ytemp
            zz = ztemp
            if (i+1)%11!=0:
                atoms.append([i+1, molID, 1, 0, xx, yy, zz])
                coords.append([xx,yy,zz])
                k+=1
            if (i+1)%11==0:
                count+=1
                if count <= (((numberofchains*atomnum)/2)//11):
                    atoms.append([i+1, molID, 2, 1, xx, yy, zz])
                    k +=1
                elif count > (((numberofchains*atomnum)/2)//11):
                    atoms.append([i+1, molID, 3, -1, xx, yy, zz])
                    k+=1
            if i%500==0:
                print('{} % work done'.format((i/(numberofchains*monomernum))*100))
            actual_moves+=1

    for i in range(startb,startb+bondnum):
        bonds.append([i+1, 1, l, l+1])
        l+=1
            
    for i in range(startn,startn+anglenum):
        angles.append([i+1, 1, n, n+1, n+2])
        n+=1

    for i in range(startd,startd+dihedralnum):
        dihedrals.append([i+1, 1, r, r+1, r+2, r+3])
        r+=1
    numberofchains2-=1
    starta = atomnum*j
    startb = bondnum*j
    l = bondnum*j + o
    startn = anglenum*j
    n = anglenum*j + p
    startd = dihedralnum*j
    r = dihedralnum*j + q
    j+=1
    o+=1
    p+=2
    q+=3
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

f=open('CG_%sC_%sDP.data' %(numberofchains,monomernum),'w')
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
f.write('\n')
f.write('Angles\n\n')
for line in angles:
    f.write('%s\t%s\t%s\t%s\t%s\n' % (line[0], line[1], line[2], line[3], line[4]))
f.write('\n')
f.write('Dihedrals\n\n')
for line in dihedrals:
    f.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (line[0], line[1], line[2], line[3], line[4], line[5]))
f.close()
