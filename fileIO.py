import os
import string

default_path = r'c:\Users\Raiter\OneDrive - Cornell University\Thesis\Results\python_analysis\\'

def findInSubdirectory(filename, subdirectory=''):
    """Input a filename and find the path to that file if that file
    is in current directory. If the file is not present in the current
    directory, then provide the subdirectory which needs to be searched
    for the file.
    
    Args:
        filename (str): A string containing the name of the file that's 
                        requested. 
        subdirectory (optional,str): A string containing the sub-directory
                                     to be searched.
    
    Returns:
        path (str): Path to the requested file.
    """
    if subdirectory:
        path = subdirectory
    else:
        path=r'c:\Users\Raiter\OneDrive - Cornell University\Thesis\Results\simulation_files_synchronized_stampede2\\'
        #path=r'/mnt/c/Users/Raiter/OneDrive - Cornell University/Thesis/Results/python_analysis'
        #path = os.getcwd()
    for root, dirs, names in os.walk(path):
        if filename in names:
            print(os.path.join(root, filename))
            return os.path.join(root, filename)
    raise NameError('File not found!')

def retrieve_different_filetypes(rootname):
    """Based on a given file rootname, various files associated with that rootname such
    as .lammpstrj, .def1.txt,.def2.txt, etc. are returned.
    
    Args:
        rootname (str): Basename of the simulation (usually present on the first line of the 
            simulation input file -- variable simname equal <basename>
     
    Returns:
        dump_wrapped (str)
        dump_unwrapped (str)
        dump_def1 (str)
        dump_def2 (str)
        dump_def3 (str)
        log_file
    """

    dump_wrapped=rootname+'.lammpstrj'
    dump_unwrapped=rootname+'.unwrapped.lammpstrj'
    dump_def1=rootname+'.def1.txt'
    dump_def2=rootname+'.def2.txt'
    dump_def3=rootname+'.def3.txt'
    log_file='log.' + rootname + '.txt'
    
    return dump_wrapped, dump_unwrapped, dump_def1, dump_def2, dump_def3, log_file

def break_file(fname,lines_per_file=6151845):
    fpath=findInSubdirectory(fname)
    index=0
    smallfile = None
    with open(fpath) as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = fname.split('.')[0] + '_part{}.unwrapped.lammpstrj'.format(list(string.ascii_lowercase)[index])
                smallfile = open(small_filename, "w")
                index+=1
            smallfile.write(line)
        if smallfile:
            smallfile.close()