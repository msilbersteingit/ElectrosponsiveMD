import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analib import fileIO

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
    syntax (str): the format specification of the requested def file. Default
     is '18f'
    
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

def extract_unwrapped(simname,format_spec =['id','mol','type','xu','yu','zu']):
    """Extract coordinates of all atoms from unwrapped trajectory files with 
    format 'id','mol','type','xu','yu','zu'
    Args:
    simname (str): name of the LAMMPS simulation.

    Returns:
    coord (dict): Dictionary with the number of timesteps as keys and 
                  coordinates of all atoms at the corresponding timestep as 
                  a pandas dataframe.
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
                    #sort based on atom id so that future operations 
                    # are easy.
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
                  coordinates of all atoms at the corresponding timestep as 
                  a pandas dataframe. The format of keys is:
                  'timestep_<timestep>'
                  For ex:
                  'timestep_0'
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

def read_boxsize(simname,last_timestep=173):
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
    
    timestep=1
    prev_index=-10004
    ref_size_x = (float(unwrap.iloc[prev_index + 10009].str.split()[0][1]) 
                  - float(unwrap.iloc[prev_index + 10009].str.split()[0][0]))
    ref_size_y = (float(unwrap.iloc[prev_index + 10010].str.split()[0][1]) 
                  - float(unwrap.iloc[prev_index + 10010].str.split()[0][0]))
    ref_size_z = (float(unwrap.iloc[prev_index + 10011].str.split()[0][1]) 
                  - float(unwrap.iloc[prev_index + 10011].str.split()[0][0]))
    prop_constant=[None]*last_timestep
    while timestep<=last_timestep:
        curr_size_x = (float(unwrap.iloc[prev_index + 10009].str.split()[0][1])
                       - float(unwrap.iloc[prev_index + 10009].str.split()[0][0]))
        curr_size_y = (float(unwrap.iloc[prev_index + 10010].str.split()[0][1]) 
                       - float(unwrap.iloc[prev_index + 10010].str.split()[0][0]))
        curr_size_z = (float(unwrap.iloc[prev_index + 10011].str.split()[0][1]) 
                       - float(unwrap.iloc[prev_index + 10011].str.split()[0][0]))
        px=round(curr_size_x,2)
        py=round(curr_size_y,2)
        pz=round(curr_size_z,2)
        prop_constant[timestep-1]=[px,py,pz]
        prev_index+=10009
        timestep+=1
    return prop_constant


