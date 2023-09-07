####################################

# Run the Tetrahedra Toy Model
# Jiamin Hou 2022 Jan

####################################



import os, pickle, importlib, glob, sys
from nbodykit.lab import *
sys.path.append("/orange/zslepian/octobers/NPCF/codes")
from mk_tetrahedra_sims import mk_cat_dat, mk_cat_dat_flextet, mk_cat_ran

imock = int(sys.argv[1])
N0 = 100 #40000
r_min, r_max = 15, 100
tet_type = 'Embed'

coords, coord_0, coord_1, coord_2, coord_3, rs = mk_cat_dat_flextet(N0=N0, r_min=r_min, r_max=r_max,
                                                 imock=imock, 
                                                 froot="/orange/zslepian/octobers/Simulations/Tetrahedra/",
                                                 tet_type=tet_type,
                                                 verbose=True)

