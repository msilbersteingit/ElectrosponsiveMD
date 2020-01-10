from analib import extract
from analib import compute
from analib import plotting_functions


def rdf(coordinates, box_l, simname, coords_loc=[3,6], nhis = 200,
    save=False,style='matplotlib'):
    """ Radial distribution function"""
    g,r = compute.radial_distribution_function(coordinates, box_l, simname, 
        coords_loc, nhis)
    plotting_functions.radial_distribution_function_plotting(g,r,simname,save,style)

def sk(coordinates, rdf_fname,box_l,simname,coords_loc=[3,6],
    save=False,style='matplotlib'):
    " Structure factor from rdf"
    k,sk = compute.structure_factor(coordinates, rdf_fname,box_l,simname,coords_loc)
    plotting_functions.structure_factor_plotting(k,sk,simname,save,style)