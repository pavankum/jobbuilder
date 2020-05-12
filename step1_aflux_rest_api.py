import json, sys, os, fileinput
import shutil
from urllib.request import urlopen, urlretrieve
import numpy as np
import ase.io.vasp

#PLACING A HIGER U VALUE AT THE REQUIRED LOCALIZATION

###################################################################################################################
################################		 POSCAR ELONGATE FUNCTION			 ################################################
###################################################################################################################
def poscar_elongate(poscarfname, aroundatom_pos):
    fname = poscarfname
    aroundatom = aroundatom_pos
    change = 0.15  # Angstrom
    cutoff = 2.6  # float(sys.argv[4])
    f = open(fname, 'r+')
    contents = f.readlines()
    if (np.longdouble(str.split(contents[1])[0]) == 1):
        cell_a = [np.longdouble(numeric_string) for numeric_string in str.split(contents[2])]
        cell_b = [np.longdouble(numeric_string) for numeric_string in str.split(contents[3])]
        cell_c = [np.longdouble(numeric_string) for numeric_string in str.split(contents[4])]
    cell = [cell_a, cell_b, cell_c]
    natoms = sum([np.longdouble(numeric_string) for numeric_string in str.split(contents[6])])
    xyz = []
    a = np.sqrt(np.sum(np.square(cell_a)))
    b = np.sqrt(np.sum(np.square(cell_b)))
    c = np.sqrt(np.sum(np.square(cell_c)))
    if (str.split(contents[7])[0] == "Direct"):
        print("Fractional coordinates are present, okay :) :")
        for i in range(1, int(natoms) + 1):
            xyz.append([np.longdouble(numeric_string) for numeric_string in str.split(contents[i + 7])])
    xyz = np.matrix(xyz)
    #From ase.geometry.cell
    #https://wiki.fysik.dtu.dk/ase/_modules/ase/geometry/cell.html
    lengths = [a, b, c]
    angles = []
    
    for i in range(3):
        j = i - 1
        k = i - 2
        ll = lengths[j] * lengths[k]
        if ll > 1e-16:
            x = np.dot(cell[j], cell[k]) / ll
            angle = 180.0 / np.pi * arccos(x)
        else:
            angle = 90.0
        angles.append(angle)
    
    alpha = angles[0]
    beta  = angles[1]
    gamma = angles[2]
    #In radians
    a2r = np.pi/180.0
    alpha = a2r * alpha
    beta  = a2r * beta
    gamma = a2r * gamma
    print("a=", '%.5f'%a, "b=", '%.5f'%b, "c=", '%.5f'%c, "alpha=", '%.5f'%angles[0], "beta=", '%.5f'%angles[1], "gamma=", '%.5f'%angles[2])
    
    v = a*b*c*np.sqrt(1 -cos(alpha)*cos(alpha) - cos(beta)*cos(beta) - cos(gamma)*cos(gamma) + 2*cos(alpha)*cos(beta)*cos(gamma))
    
    #for cart to frac: frac = cart*tmat.T
    tmat = np.matrix( [
          [ 1.0 / a, -cos(gamma)/(a*sin(gamma)), b*c*(cos(alpha)*cos(gamma)-cos(beta)) / (v*sin(gamma))  ],
          [ 0.0,     1.0 / (b*sin(gamma)),         a*c*(cos(beta) *cos(gamma)-cos(alpha))/ (v*sin(gamma))  ],
          [ 0.0,     0.0,                        a*b*sin(gamma) / v                                    ] ]
          )
    
    #for frac to cart: cart = frac*op_mat.T
    op_mat = np.matrix( [
          [   a,   b*cos(gamma),  c*cos(beta)                                        ],
          [ 0.0,   b*sin(gamma),  c*((cos(alpha)-cos(beta) *cos(gamma))/sin(gamma))  ],
          [ 0.0,            0.0,  v/(a*b*sin(gamma))                                 ] ]
          )
    
    xyz = xyz*op_mat.T
    for i in range(0, int(natoms)):
        temp_around_x = xyz[aroundatom - 1, 0]
        temp_around_y = xyz[aroundatom - 1, 1]
        temp_around_z = xyz[aroundatom - 1, 2]
        if (i != (aroundatom - 1)):
            temp_dist0 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] - xyz[i, 0]) + np.square(xyz[aroundatom - 1, 1] - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] - xyz[i, 2]))
            if temp_dist0 < cutoff:
                vector_dir = np.subtract(xyz[i], xyz[aroundatom - 1])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #1 +X
            temp_dist1 = np.sqrt(np.square(xyz[aroundatom - 1, 0] + a - xyz[i, 0]) + np.square(
                xyz[aroundatom - 1, 1] - xyz[i, 1]) + np.square(xyz[aroundatom - 1, 2] - xyz[i, 2]))
            if temp_dist1 < cutoff:
                temp_around_x += a
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #2 -X
            temp_dist1 = np.sqrt(np.square(xyz[aroundatom - 1, 0] - a - xyz[i, 0]) + np.square(
                xyz[aroundatom - 1, 1] - xyz[i, 1]) + np.square(xyz[aroundatom - 1, 2] - xyz[i, 2]))
            if temp_dist1 < cutoff:
                temp_around_x -= a
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
    
            #3 +X +Y
            temp_dist2 = np.sqrt(np.square(xyz[aroundatom - 1, 0] + a - xyz[i, 0]) + np.square(
                xyz[aroundatom - 1, 1] + b - xyz[i, 1]) + np.square(xyz[aroundatom - 1, 2] - xyz[i, 2]))
            if temp_dist2 < cutoff:
                temp_around_x += a
                temp_around_y += b
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #4 -X -Y
            temp_dist2 = np.sqrt(np.square(xyz[aroundatom - 1, 0] - a - xyz[i, 0]) + np.square(
                xyz[aroundatom - 1, 1] - b - xyz[i, 1]) + np.square(xyz[aroundatom - 1, 2] - xyz[i, 2]))
            if temp_dist2 < cutoff:
                temp_around_x -= a
                temp_around_y -= b
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #5 -X +Y
            temp_dist2 = np.sqrt(np.square(xyz[aroundatom - 1, 0] - a - xyz[i, 0]) + np.square(
                xyz[aroundatom - 1, 1] + b - xyz[i, 1]) + np.square(xyz[aroundatom - 1, 2] - xyz[i, 2]))
            if temp_dist2 < cutoff:
                temp_around_x -= a
                temp_around_y += b
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #6 +X -Y
            temp_dist2 = np.sqrt(np.square(xyz[aroundatom - 1, 0] + a - xyz[i, 0]) + np.square(
                xyz[aroundatom - 1, 1] - b - xyz[i, 1]) + np.square(xyz[aroundatom - 1, 2] - xyz[i, 2]))
            if temp_dist2 < cutoff:
                temp_around_x += a
                temp_around_y -= b
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
    
            #7 +Y
            temp_dist3 = np.sqrt(np.square(xyz[aroundatom - 1, 0] - xyz[i, 0]) + np.square(
                xyz[aroundatom - 1, 1] + b - xyz[i, 1]) + np.square(xyz[aroundatom - 1, 2] - xyz[i, 2]))
            if (temp_dist3 < cutoff):
                temp_around_y += b
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
            #8 -Y
            temp_dist3 = np.sqrt(np.square(xyz[aroundatom - 1, 0] - xyz[i, 0]) + np.square(
                xyz[aroundatom - 1, 1] - b - xyz[i, 1]) + np.square(xyz[aroundatom - 1, 2] - xyz[i, 2]))
            if (temp_dist3 < cutoff):
                temp_around_y -= b
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
    
            #9 +Y +Z
            temp_dist4 = np.sqrt(np.square(xyz[aroundatom - 1, 0] - xyz[i, 0]) + np.square(
                xyz[aroundatom - 1, 1] + b - xyz[i, 1]) + np.square(xyz[aroundatom - 1, 2] + c - xyz[i, 2]))
            if (temp_dist4 < cutoff):
                temp_around_y += b
                temp_around_z += c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #10 -Y -Z
            temp_dist4 = np.sqrt(np.square(xyz[aroundatom - 1, 0] - xyz[i, 0]) + np.square(
                xyz[aroundatom - 1, 1] - b - xyz[i, 1]) + np.square(xyz[aroundatom - 1, 2] - c - xyz[i, 2]))
            if (temp_dist4 < cutoff):
                temp_around_y -= b
                temp_around_z -= c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #11 -Y +Z
            temp_dist4 = np.sqrt(np.square(xyz[aroundatom - 1, 0] - xyz[i, 0]) + np.square(
                xyz[aroundatom - 1, 1] - b - xyz[i, 1]) + np.square(xyz[aroundatom - 1, 2] + c - xyz[i, 2]))
            if (temp_dist4 < cutoff):
                temp_around_y -= b
                temp_around_z += c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #12 +Y -Z
            temp_dist4 = np.sqrt(np.square(xyz[aroundatom - 1, 0] - xyz[i, 0]) + np.square(
                xyz[aroundatom - 1, 1] + b - xyz[i, 1]) + np.square(xyz[aroundatom - 1, 2] - c - xyz[i, 2]))
            if (temp_dist4 < cutoff):
                temp_around_y += b
                temp_around_z -= c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
    
            #13 +Z
            temp_dist5 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] - xyz[i, 0]) + np.square(xyz[aroundatom - 1, 1] - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] + c - xyz[i, 2]))
            if (temp_dist5 < cutoff):
                temp_around_z += c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #14 -Z
            temp_dist5 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] - xyz[i, 0]) + np.square(xyz[aroundatom - 1, 1] - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] - c - xyz[i, 2]))
            if (temp_dist5 < cutoff):
                temp_around_z -= c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
    
            #15 +X +Z
            temp_dist6 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] + a - xyz[i, 0]) + np.square(
                    xyz[aroundatom - 1, 1] - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] + c - xyz[i, 2]))
            if (temp_dist6 < cutoff):
                temp_around_x += a
                temp_around_z += c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #16 -X -Z
            temp_dist6 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] - a - xyz[i, 0]) + np.square(
                    xyz[aroundatom - 1, 1] - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] - c - xyz[i, 2]))
            if (temp_dist6 < cutoff):
                temp_around_x -= a
                temp_around_z -= c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #17 +X -Z
            temp_dist6 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] + a - xyz[i, 0]) + np.square(
                    xyz[aroundatom - 1, 1] - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] - c - xyz[i, 2]))
            if (temp_dist6 < cutoff):
                temp_around_x += a
                temp_around_z -= c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #18 -X +Z
            temp_dist6 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] - a - xyz[i, 0]) + np.square(
                    xyz[aroundatom - 1, 1] - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] + c - xyz[i, 2]))
            if (temp_dist6 < cutoff):
                temp_around_x -= a
                temp_around_z += c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
    
    
            #19 +X +Y +Z
            temp_dist7 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] + a - xyz[i, 0]) + np.square(
                    xyz[aroundatom - 1, 1] + b - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] + c - xyz[i, 2]))
            if (temp_dist7 < cutoff):
                temp_around_x += a
                temp_around_y += b
                temp_around_z += c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #20 -X -Y -Z
            temp_dist7 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] - a - xyz[i, 0]) + np.square(
                    xyz[aroundatom - 1, 1] - b - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] - c - xyz[i, 2]))
            if (temp_dist7 < cutoff):
                temp_around_x -= a
                temp_around_y -= b
                temp_around_z -= c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #21 -X -Y +Z
            temp_dist7 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] - a - xyz[i, 0]) + np.square(
                    xyz[aroundatom - 1, 1] - b - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] + c - xyz[i, 2]))
            if (temp_dist7 < cutoff):
                temp_around_x -= a
                temp_around_y -= b
                temp_around_z += c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #22 -X +Y -Z
            temp_dist7 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] - a - xyz[i, 0]) + np.square(
                    xyz[aroundatom - 1, 1] + b - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] - c - xyz[i, 2]))
            if (temp_dist7 < cutoff):
                temp_around_x -= a
                temp_around_y += b
                temp_around_z -= c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #23 -X +Y +Z
            temp_dist7 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] - a - xyz[i, 0]) + np.square(
                    xyz[aroundatom - 1, 1] + b - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] + c - xyz[i, 2]))
            if (temp_dist7 < cutoff):
                temp_around_x -= a
                temp_around_y += b
                temp_around_z += c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #24 +X -Y -Z
            temp_dist7 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] + a - xyz[i, 0]) + np.square(
                    xyz[aroundatom - 1, 1] - b - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] - c - xyz[i, 2]))
            if (temp_dist7 < cutoff):
                temp_around_x += a
                temp_around_y -= b
                temp_around_z -= c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #25 +X +Y -Z
            temp_dist7 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] + a - xyz[i, 0]) + np.square(
                    xyz[aroundatom - 1, 1] + b - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] - c - xyz[i, 2]))
            if (temp_dist7 < cutoff):
                temp_around_x += a
                temp_around_y += b
                temp_around_z -= c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #26 +X -Y +Z
            temp_dist7 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] + a - xyz[i, 0]) + np.square(
                    xyz[aroundatom - 1, 1] - b - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] + c - xyz[i, 2]))
            if (temp_dist7 < cutoff):
                temp_around_x += a
                temp_around_y -= b
                temp_around_z += c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
            #27 +X +Y -Z
            temp_dist7 = np.sqrt(
                np.square(xyz[aroundatom - 1, 0] + a - xyz[i, 0]) + np.square(
                    xyz[aroundatom - 1, 1] + b - xyz[i, 1]) + np.square(
                    xyz[aroundatom - 1, 2] - c - xyz[i, 2]))
            if (temp_dist7 < cutoff):
                temp_around_x += a
                temp_around_y += b
                temp_around_z -= c
                vector_dir = np.subtract(xyz[i], [temp_around_x, temp_around_y, temp_around_z])
                magnitude = np.linalg.norm(vector_dir)
                xyz[i] += np.multiply(change, np.divide(vector_dir, magnitude))
                continue
    
    
    sys.stdout.flush()
    f.close()
    
    cell_vector = np.array([a, b, c])
    xyz = xyz*tmat.T
    np.savetxt('POSCAR', xyz,
               header=f'{contents[0]}{contents[1]}{contents[2]}{contents[3]}{contents[4]}{contents[5]}{contents[6]}Direct',
               fmt=' %0.9f',
               delimiter=' ', comments='')
    print("poscar modified")


###################################################################################################################
#################################      Generating the required input files     ####################################
###################################################################################################################

# Compound space consists of the ABX3 compound names in the format [A B X]
compound_space = np.genfromtxt("compound_space.txt", delimiter='\t', dtype=str)
compound_space_orig = np.empty_like(compound_space)
compound_space_orig[:] = compound_space
# Stoichiometry to specify the ABX3 composition
stoichiometry_compound = np.tile([0.2, 0.2, 0.6], (199,))
stoichiometry_compound = stoichiometry_compound.reshape(199, 3)
# AFLOWLIB entries are alphabetically arranged so need to sort the compound name
# alphabetically along with the attached stoichiometry
index_alphabetic_compound = compound_space.argsort()
stoichiometry_compound = np.take(stoichiometry_compound, index_alphabetic_compound)
compound_space.sort()
# Path to create the runs folder
mypath = 'C:/Users/Maverick/PycharmProjects/aflow_query_job_builder/runs2/'
PP_path = 'C:/Users/Maverick/PycharmProjects/aflow_query_job_builder/potpaw_PBE.54/'
run_num = 0
for iter in range(compound_space.shape[0]):
    species = f'{compound_space[iter,0]},{compound_space[iter,1]},{compound_space[iter,2]}'
    stoichiometry = f'{stoichiometry_compound[iter][0]},{stoichiometry_compound[iter][1]},{stoichiometry_compound[iter][2]}'
    dft_type = "PAW_PBE"
    files = ["CONTCAR.relax", "aflowlib.json"]

    SERVER = "http://aflowlib.duke.edu"
    API = "/search/API/?"
    SUMMONS = "species(" + species + "),stoichiometry('" + stoichiometry + "'),dft_type(" + dft_type + ")," \
                                                                                                       "energy_cell,species_pp,delta_electronic_energy_threshold,ldau_TLUJ,$compound,aurl," \
                                                                                                       "species_pp_ZVAL,composition,energy_cutoff,code,$paging(-1,1)"
    print(SERVER + API + SUMMONS)
    response = json.loads(urlopen(SERVER + API + SUMMONS).read().decode("utf-8"))
    if response:
        run_num += 1
        run_folder_name = f'{mypath}run_{run_num}'
        os.makedirs(run_folder_name, exist_ok=True)
        for datum in response:
            print(f"{datum['auid']}, {datum['energy_cell']}, {datum['dft_type']}, {datum['stoichiometry']}, "
                  f"{datum['code']}, {datum['species_pp']}")
            for file in files:
                filename = file + "_" + datum['auid'].split(":")[1]
                url_name = "http://" + datum['aurl'] + "/" + file
                url_name_list = url_name.split("edu:A")
                url_name = "edu/A".join(url_name_list[:])
                file_url = urlopen(url_name)
                download = open(filename, "wb")
                fifth_line = species.replace(',', ' ')
                download.write(file_url.read())
                download.close()
                with open(filename, 'r') as f:
                    lines = f.readlines()
                with open(filename, 'w') as f:
                    for i, line in enumerate(lines):
                        if i == 5:
                            f.write(fifth_line + '\n')
                        f.write(line)
                shutil.copy(filename, run_folder_name)
                if file.startswith('CONTCAR'):
                    shutil.copy(run_folder_name + '/' + filename, run_folder_name + '/POSCAR')
            PP_names = datum['species_pp'].split(',')
            filenames = [PP_path + PP_names[0] + '/POTCAR', PP_path + PP_names[1] + '/POTCAR',
                         PP_path + PP_names[2] + '/POTCAR']
            with open(run_folder_name + '/POTCAR', 'w') as outfile:
                for fname in filenames:
                    with open(fname) as infile:
                        for line in infile:
                            outfile.write(line)
            cell = ase.io.vasp.read_vasp(run_folder_name + '/POSCAR')
            composition = np.array(datum['composition'].split(',')).astype(np.float)
            num_atoms = np.sum(composition)
            if num_atoms < 7:
                a_super = 3
                b_super = 2
                c_super = 2
                ase.io.vasp.write_vasp(run_folder_name + '/POSCAR', cell * (a_super, b_super, c_super),
                                       label='322supercell', direct=True,
                                       sort=True,
                                       vasp5=True)
            elif num_atoms>6 and num_atoms < 10:
                a_super = 2
                b_super = 2
                c_super = 2
                ase.io.vasp.write_vasp(run_folder_name + '/POSCAR', cell * (a_super, b_super, c_super),
                                       label='222supercell', direct=True,
                                       sort=True,
                                       vasp5=True)
            else:
                a_super = 2
                b_super = 2
                c_super = 1
                ase.io.vasp.write_vasp(run_folder_name + '/POSCAR', cell * (a_super, b_super, c_super),
                                       label='221supercell', direct=True,
                                       sort=True,
                                       vasp5=True)

            cutoff_incar = f'ENCUT = {datum["energy_cutoff"]}'
            system_incar = f'SYSTEM={compound_space_orig[iter,0]}{compound_space_orig[iter,1]}{compound_space_orig[iter,2]}3,[{datum["species"]}]'

            species_pp_ZVAL = np.array(datum['species_pp_ZVAL'].split(',')).astype(np.float)

            super_abc = a_super * b_super * c_super
            num_val_electrons = super_abc * np.sum(np.dot(species_pp_ZVAL, composition)) + 1
            if datum['ldau_TLUJ']:
                ldau_list = datum['ldau_TLUJ'].split(';')
                ldau_type = ldau_list[0]
                ldau_L = ldau_list[1].replace(',', ' ')
                # ldau_U = ldau_list[2].replace(',', ' ')
                ldau_J = ldau_list[3].replace(',', ' ')

            else:
                ldau_line = 'LDAU = .FALSE.'
            if index_alphabetic_compound[iter, 0] == 1:
                if datum['ldau_TLUJ']:
                    ldau_U = '6 0 0'
                magmom_incar = f'1*3.0 {np.int(super_abc*composition[0]-1)}*0.0 {np.int(super_abc*composition[1])}*0.0 {np.int(super_abc*composition[2])}*0.0 '
                poscar_elongate(run_folder_name + '/POSCAR', 1)
            elif index_alphabetic_compound[iter, 1] == 1:
                if datum['ldau_TLUJ']:
                    ldau_U = '0 6 0'
                magmom_incar = f'{np.int(super_abc*composition[0])}*0.0 1*3.0 {np.int(super_abc*composition[1]-1)}*0.0 {np.int(super_abc*composition[2])}*0.0 '
                poscar_elongate(run_folder_name + '/POSCAR', np.int(super_abc * composition[0]) + 1)
            else:
                if datum['ldau_TLUJ']:
                    ldau_U = '0 0 6'
                magmom_incar = f' {np.int(super_abc*composition[0])}*0.0  {np.int(super_abc*composition[1])}*0.0  {np.int(super_abc*composition[2] - 1)}*0.0  1*3.0'
                poscar_elongate(run_folder_name + '/POSCAR', np.int(super_abc * np.sum(composition)))

            if datum['ldau_TLUJ']:
                ldau_line = f'LDAU = .TRUE. \n LDAUTYPE = {ldau_type}\n LDAUL = {ldau_L} \n LDAUU = {ldau_U} \n ' \
                            f'LDAUJ = {ldau_J} \n LDAUPRINT = 1\n'

            shutil.copy('INCAR', run_folder_name)
            with open(run_folder_name + '/INCAR', 'r') as incar:
                lines = incar.readlines()

            with open(run_folder_name + '/INCAR', 'w') as incar:
                for i, line in enumerate(lines):
                    if i == 1:
                        incar.write(f' {system_incar}\n'
                                    f' NELECT = {np.int(num_val_electrons)} \n MAGMOM = {magmom_incar}\n'
                                    f' {cutoff_incar} \n'
                                    f' {ldau_line}')
                    incar.write(line)

            shutil.copy('KPOINTS', run_folder_name)
            shutil.copy('slurmscript', run_folder_name)
            job_name = f'#SBATCH --job-name={run_num}_{species}'
            with open(run_folder_name + '/slurmscript', 'r') as f:
                lines = f.readlines()
            with open(run_folder_name + '/slurmscript', 'w') as f:
                for i, line in enumerate(lines):
                    if i == 7:
                        f.write(job_name + '\n')
                    f.write(line)
    else:
        print('No data')
