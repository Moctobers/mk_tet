########################################

# Generate Toy Tetrahedra 
# Jiamin Hou 2022 Jan.

########################################

import os, sys, time
import dask.array as da
sys.path.append("/global/homes/o/octobers/Projects/NPCF/codes")
from NPCF_utils import GetCoeffCpp

from nbodykit.lab import *
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

def mk_cat_dat(N0, rs=[10, 20, 30], which_hand="cw_only", imock=1, froot=None, verbose=False):
    
    '''
        generate orthogonal tetrahedra

        which_hand options: cw_only, aw_only, cwaw
    '''
    
    # Init. primary galaxies
    s = numpy.random.randint(0 + max(rs), 1000-max(rs), size=(N0, 3))
    
    # Clean galaxies falls in neighbour radius (30 Mpc/h)
    kd_tree1 = KDTree(s)
    indexes = kd_tree1.query_ball_tree(kd_tree1, r=numpy.ceil(max(rs)*2))
    try:
        neighbour_index = numpy.array(list(set(numpy.hstack(
                                      numpy.array([i for i in indexes if (len(i)>1)])))))
    except:
        neighbour_index = []                            
    coord_0 = []
    for i in range(0, N0):
        if i not in neighbour_index:
            coord_0.append(s[i,:])
    coord_0 = numpy.array(coord_0)
    if verbose:
        print ("After cleaning, N_primary gals = ", len(coord_0))

    # Number of primary galaxies after cleaning
    N0 = len(coord_0)

    # Init. second. galaxies
    if which_hand == "cw_only":
        if verbose: print("do clockwise only")
        name_ext = '_cw'
        N0_use = N0
        r1 = numpy.array([rs[0], 0, 0] * N0_use).reshape(N0_use, 3)
        r2 = numpy.array([0, rs[1], 0] * N0_use).reshape(N0_use, 3)
        r3 = numpy.array([0, 0, rs[2]] * N0_use).reshape(N0_use, 3)
        dr1 = numpy.random.randint(0, 20, size=(int(N0_use))) * 0.1
        dr2 = numpy.random.randint(-20, 30, size=(int(N0_use))) * 0.1
        dr3 = numpy.random.randint(-10, 0, size=(int(N0_use))) * 0.1
        r1[:,0] = r1[:,0] + dr1
        r2[:,1] = r2[:,1] + dr2
        r3[:,2] = r3[:,2] + dr3
    elif which_hand == "aw_only":
        if verbose: print("do anti-clockwise only")
        name_ext = '_aw'
        N0_use = N0
        r1 = numpy.array([0, rs[0], 0] * N0_use).reshape(N0_use, 3)
        r2 = numpy.array([rs[1], 0, 0] * N0_use).reshape(N0_use, 3)
        r3 = numpy.array([0, 0, rs[2]] * N0_use).reshape(N0_use, 3)
        dr1 = numpy.random.randint(0, 20, size=(int(N0_use))) * 0.1
        dr2 = numpy.random.randint(-20, 30, size=(int(N0_use))) * 0.1
        dr3 = numpy.random.randint(-10, 0, size=(int(N0_use))) * 0.1
        r1[:,1] = r1[:,1] + dr1
        r2[:,0] = r2[:,0] + dr2
        r3[:,2] = r3[:,2] + dr3
    elif which_hand == "cwaw":
        if verbose: print("do both clockwise and anti-clockwise.")
        name_ext = '_cwaw'
        if numpy.mod(N0, 2) == 0:
            N_cw = int(N0/2)
            N_aw = N_cw
            N0_use = N0
        else:
            N_cw = int(numpy.floor(N0-1)/2)
            N_aw = N_cw
            N0_use = N_aw + N_cw
            coord_0 = coord_0[:N0_use, :]
        # clock-wise (cw)
        r1_cw = numpy.array([rs[0], 0, 0] * N_cw).reshape(N_cw, 3)
        r2_cw = numpy.array([0, rs[1], 0] * N_cw).reshape(N_cw, 3)
        r3_cw = numpy.array([0, 0, rs[2]] * N_cw).reshape(N_cw, 3)
        dr1 = numpy.random.randint(0, 20, size=(int(N_cw))) * 0.1
        dr2 = numpy.random.randint(-20, 30, size=(int(N_cw))) * 0.1
        dr3 = numpy.random.randint(-10, 0, size=(int(N_cw))) * 0.1
        r1_cw[:,0] = r1_cw[:,0] + dr1
        r2_cw[:,1] = r2_cw[:,1] + dr2
        r3_cw[:,2] = r3_cw[:,2] + dr3
        # anti-clockwise (aw)
        r1_aw = numpy.array([0, rs[0], 0] * N_aw).reshape(N_aw, 3)
        r2_aw = numpy.array([rs[1], 0, 0] * N_aw).reshape(N_aw, 3)
        r3_aw = numpy.array([0, 0, rs[2]] * N_aw).reshape(N_aw, 3)
        dr1 = numpy.random.randint(0, 20, size=(int(N_aw))) *0.1
        dr2 = numpy.random.randint(-20, 30, size=(int(N_aw))) * 0.1
        dr3 = numpy.random.randint(-10, 0, size=(int(N_aw))) * 0.1
        r1_aw[:,1] = r1_aw[:,1] + dr1
        r2_aw[:,0] = r2_aw[:,0] + dr2
        r3_aw[:,2] = r3_aw[:,2] + dr3
        r1 = numpy.vstack([r1_cw, r1_aw])
        r2 = numpy.vstack([r2_cw, r2_aw])
        r3 = numpy.vstack([r3_cw, r3_aw])
         
#         mask = numpy.random.randint(0, N0_use, size=(int(N0_use/2)))
#         r1[mask, :] = -r1[mask, :]
#         r2[mask, :] = -r2[mask, :]
#         r3[mask, :] = -r3[mask, :]

    # Generate arbitrary rotation angles along x, y, z axis
    rot_degx = numpy.random.randint(-179, 179, size=(N0_use)) 
    rot_degy = numpy.random.randint(-179, 179, size=(N0_use)) 
    rot_degz = numpy.random.randint(-179, 179, size=(N0_use)) 
    rot_radx = numpy.radians(rot_degx)
    rot_rady = numpy.radians(rot_degy)
    rot_radz = numpy.radians(rot_degz)

    # Define rotation axis
    xaxis = numpy.array([1, 0, 0])
    yaxis = numpy.array([0, 1, 0])
    zaxis = numpy.array([0, 0, 1])

    # Rotation vector in x,y,z
    rot_vecx = [i * xaxis for i in rot_radx]
    rot_vecy = [i * yaxis for i in rot_rady]
    rot_vecz = [i * zaxis for i in rot_radz]

    # Rotation operation in x,y,z
    rot_x = Rotation.from_rotvec(rot_vecx)
    rot_y = Rotation.from_rotvec(rot_vecx)
    rot_z = Rotation.from_rotvec(rot_vecx)

    r1_rotd = []
    r2_rotd = []
    r3_rotd = []

    for i in range(0, len(coord_0)):
        # rotate along the first axis (x-axis)
        rot1_1axis = rot_x[i].apply(r1[i, :])
        rot2_1axis = rot_x[i].apply(r2[i, :])
        rot3_1axis = rot_x[i].apply(r3[i, :])
        # rotate along the second axis (y-axis)
        rot1_2axis = rot_y[i].apply(rot1_1axis)
        rot2_2axis = rot_y[i].apply(rot2_1axis)
        rot3_2axis = rot_y[i].apply(rot3_1axis)
        # rotate along the third axis (z-axis)
        rot1_3axis = rot_z[i].apply(rot1_2axis)
        rot2_3axis = rot_z[i].apply(rot2_2axis)
        rot3_3axis = rot_z[i].apply(rot3_2axis)
        r1_rotd.append(rot1_3axis)
        r2_rotd.append(rot2_3axis)
        r3_rotd.append(rot3_3axis)

    r1_rotd = numpy.vstack(r1_rotd)
    r2_rotd = numpy.vstack(r2_rotd)
    r3_rotd = numpy.vstack(r3_rotd)

    coord_1 = coord_0 + r1_rotd
    coord_2 = coord_0 + r2_rotd
    coord_3 = coord_0 + r3_rotd

    coords = numpy.vstack([coord_0, coord_1, coord_2, coord_3])
    
    # Now use only galaxies inside the box range
    list_bc = []
    for i, coord in enumerate(coords):
        if (coord<=1000).all() and (coord>=0).all():
            list_bc.append(i)
    coords_bc = coords[list_bc]
    
    if verbose: 
        print("Galaxies outside boundry", len(coords)-len(coords_bc))
        print("Total galaxies", len(coords_bc))
        
    dat_cat = numpy.ones([len(coords_bc), 4])
    dat_cat[:,:3] = coords_bc
    dat_cat[:,3] = numpy.ones(len(coords_bc))
    if froot is not None:
        name_save = froot + f"/tet{name_ext}_box1000_10x20x30_{imock:d}.dat"
        numpy.savetxt(name_save, dat_cat)
        if verbose: print("Save to", name_save)

    return coords, coord_0, coord_1, coord_2, coord_3

def mk_cat_dat_flextet(N0, r_min=20, r_max=160, which_hand="cw_only", tet_type='Trirec',
                       imock=1, froot=None, verbose=False):
    
    '''
        Generate tetrahedra with flexible shapes

        N0: number of "primary" galaxies, must be integer number of 3
        which_hand options: cw_only, aw_only, cwaw
        tet_type: Embed, Trirec
    '''
        
    rs = numpy.random.uniform(r_min, r_max, N0*3)
    rs = numpy.sort(rs.reshape(N0, 3), axis=1)   
    r1s, r2s, r3s = rs[:,0], rs[:,1], rs[:,2]
    
    zero_arr = numpy.zeros(N0)
    
    # Init. primary galaxies
    s = numpy.random.randint(0 + r_max, 1000-r_max, size=(N0, 3))
    coord_0 = s
    
    if tet_type == 'Embed':
        # generate embedded tetrahedra
        rs = numpy.random.uniform(r_min, r_max, N0*3)
        rs = numpy.sort(rs.reshape(N0, 3), axis=1) 
        zero_arr = numpy.zeros(N0)
        l1, l2, l3 = rs[:,0], rs[:,1], rs[:,2]
        l1_3d = numpy.vstack([l1, zero_arr, zero_arr]).T
        l2_3d = numpy.vstack([zero_arr, l2, zero_arr]).T
        l3_3d = numpy.vstack([zero_arr, zero_arr, l3]).T

        if which_hand == "cw_only":
            name_ext = '_cw'
            r1 = numpy.sqrt(l2_3d**2 + l3_3d**2)
            r2 = numpy.sqrt(l1_3d**2 + l2_3d**2)
            r3 = numpy.sqrt(l1_3d**2 + l3_3d**2)

    elif tet_type == 'Trirec':    
        # Init. second. galaxies
        if which_hand == "cw_only":
            if verbose: print("do clockwise only")
            name_ext = '_cw'
            r1 = numpy.vstack([r1s, zero_arr, zero_arr]).T
            r2 = numpy.vstack([zero_arr, r2s, zero_arr]).T
            r3 = numpy.vstack([zero_arr, zero_arr, r3s]).T
        elif which_hand == "aw_only":
            if verbose: print("do anti-clockwise only")
            name_ext = '_aw'
            r1 = numpy.vstack([zero_arr, r1s, zero_arr]).T
            r2 = numpy.vstack([r2s, zero_arr, zero_arr]).T
            r3 = numpy.vstack([zero_arr, zero_arr, r3s]).T

    # Generate arbitrary rotation angles along x, y, z axis
    rot_degx = numpy.random.randint(-179, 179, size=(N0)) 
    rot_degy = numpy.random.randint(-179, 179, size=(N0)) 
    rot_degz = numpy.random.randint(-179, 179, size=(N0)) 
    rot_radx = numpy.radians(rot_degx)
    rot_rady = numpy.radians(rot_degy)
    rot_radz = numpy.radians(rot_degz)

    # Define rotation axis
    xaxis = numpy.array([1, 0, 0])
    yaxis = numpy.array([0, 1, 0])
    zaxis = numpy.array([0, 0, 1])

    # Rotation vector in x,y,z
    rot_vecx = [i * xaxis for i in rot_radx]
    rot_vecy = [i * yaxis for i in rot_rady]
    rot_vecz = [i * zaxis for i in rot_radz]

    # Rotation operation in x,y,z
    rot_x = Rotation.from_rotvec(rot_vecx)
    rot_y = Rotation.from_rotvec(rot_vecx)
    rot_z = Rotation.from_rotvec(rot_vecx)

    r1_rotd = []
    r2_rotd = []
    r3_rotd = []

    for i in range(0, len(coord_0)):
        # rotate along the first axis (x-axis)
        rot1_1axis = rot_x[i].apply(r1[i, :])
        rot2_1axis = rot_x[i].apply(r2[i, :])
        rot3_1axis = rot_x[i].apply(r3[i, :])
        # rotate along the second axis (y-axis)
        rot1_2axis = rot_y[i].apply(rot1_1axis)
        rot2_2axis = rot_y[i].apply(rot2_1axis)
        rot3_2axis = rot_y[i].apply(rot3_1axis)
        # rotate along the third axis (z-axis)
        rot1_3axis = rot_z[i].apply(rot1_2axis)
        rot2_3axis = rot_z[i].apply(rot2_2axis)
        rot3_3axis = rot_z[i].apply(rot3_2axis)
        r1_rotd.append(rot1_3axis)
        r2_rotd.append(rot2_3axis)
        r3_rotd.append(rot3_3axis)

    r1_rotd = numpy.vstack(r1_rotd)
    r2_rotd = numpy.vstack(r2_rotd)
    r3_rotd = numpy.vstack(r3_rotd)

    coord_1 = coord_0 + r1_rotd
    coord_2 = coord_0 + r2_rotd
    coord_3 = coord_0 + r3_rotd

    coords = numpy.vstack([coord_0, coord_1, coord_2, coord_3])
    
    # Now use only galaxies inside the box range
    list_bc = []
    for i, coord in enumerate(coords):
        if (coord<=1000).all() and (coord>=0).all():
            list_bc.append(i)
    coords_bc = coords[list_bc]
    
    if verbose: 
        print("Galaxies outside boundry", len(coords)-len(coords_bc))
        print("Total galaxies", len(coords_bc))
        
    dat_cat = numpy.ones([len(coords_bc), 4])
    dat_cat[:,:3] = coords_bc
    dat_cat[:,3] = numpy.ones(len(coords_bc))
    if froot is not None:
        name_save = froot + f"tet{name_ext}_box1000_N0{N0:.1E}_type{tet_type}_rmin{r_min}_rmax{r_max}_{imock:d}.dat"
        numpy.savetxt(name_save, dat_cat)
        if verbose: print("Save to", name_save)

    return coords, coord_0, coord_1, coord_2, coord_3, rs

def mk_cat_ran(N_ran, N_dat, froot=None, fname='tet_box1000_10x20x30'):

    r = numpy.random.randint(0, 1000, size=(N_ran, 3))

    # initialise name list for 
    # new random catalogue
    name_list = ['x', 'y', 'z', 'w']
    dtype = []
    for name in name_list:
        dtype.append((name, 'f8'))

    split_size = int(N_ran/32)
    for ii in range(0,32):  
        rsub = r[ii*split_size:(ii+1)*split_size, :]
        norm = len(rsub)/N_dat
        # initialize i-th of 32 catalogues 
        # and write their Cartesian coordinates
        rset = numpy.empty(split_size, dtype=dtype)
        rset['x'] = r[ii*split_size:(ii+1)*split_size, 0]
        rset['y'] = r[ii*split_size:(ii+1)*split_size, 1]
        rset['z'] = r[ii*split_size:(ii+1)*split_size, 2]
        rset['w'] = -1.0 / norm
        if froot is not None:
            name_save = froot + f"/"+ fname + f".{ii:02d}.ran"
            numpy.savetxt(name_save, rset)
            print("\r", ">>> Save to", name_save, end='')
        
    return rset

