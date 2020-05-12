import numpy as np
import sys
import subprocess
from numpy import cos, sin, arccos


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


#########################################################################
#########################################################################
PP_list = np.genfromtxt("/projects/academic/mdupuis2/pavan/runs_aflow/pseudopotential_choices.txt", delimiter='\t',
                        dtype=str)
keys = PP_list[:, 0]
values = PP_list[:, 1]
PP_dictionary = dict(zip(keys, values))

with open('POSCAR') as f:
    contents = f.readlines()

elems = contents[5].split()[0:3]
PP_path = '/projects/academic/mdupuis2/software/VASP/pseudopotentials/potpaw_PBE.54/'
zval_pp = []
for i in range(3):
    zvalues = []
    file_name = PP_path + PP_dictionary[elems[i]] + '/POTCAR'
    # print(file_name)
    with open(file_name) as f:
        zvalues.extend(f.readline() for i in range(2))
    zval_pp.append(float(zvalues[1].split()[0]))
p, q, r = 2, 2, 2
# ABX3
# print(zval_pp)
# print(elems)
n_a, n_b, n_x = 4, 4, 12
nelect = p * q * r * (n_a * zval_pp[0] + n_b * zval_pp[1] + n_x * zval_pp[2])
nelect += 1
mag_a = p*q*r*n_a
mag_b = p*q*r*n_b
mag_x = p*q*r*n_x
site1 = 56
site2 = 12
if elems[1] not in ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra']:
    magmom = f'MAGMOM = {mag_a}*0.0 {site1-mag_a-1}*0.0 1*3.0 {mag_a+mag_b-site1}*0.0 {mag_x}*0.0'
    ldau_type = 2
    ldau_L = '0 2 2 2 0'
    ldau_U = '0 6 8 6 0'
    ldau_J = '0 0 0 0 0'
    ldau_line = f'LDAU = .TRUE. \nLDAUTYPE = {ldau_type}\nLDAUL = {ldau_L} \nLDAUU = {ldau_U} \n' \
                f'LDAUJ = {ldau_J} \nLDAUPRINT = 1\n'
    PP_names = [PP_dictionary[elems[0]], PP_dictionary[elems[1]], PP_dictionary[elems[1]], PP_dictionary[elems[1]],
                PP_dictionary[elems[2]]]
    filenames = [PP_path + PP_names[0] + '/POTCAR', PP_path + PP_names[1] + '/POTCAR',
                 PP_path + PP_names[2] + '/POTCAR', PP_path + PP_names[3] + '/POTCAR',
                 PP_path + PP_names[4] + '/POTCAR']
    with open('POTCAR', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    elems_poscar = f'{elems[0]} {elems[1]} {elems[1]} {elems[1]} {elems[2]}\n'
    elem_num_poscar = f'{mag_a} {site1-mag_a-1} 1 {mag_a+mag_b-site1} {mag_x}\n'
    contents[5] = elems_poscar
    contents[6] = elem_num_poscar
    with open('POSCAR', 'w') as outfile:
        for item in contents:
            outfile.write("%s" % item)
    poscar_elongate('POSCAR', site1)
elif elems[0] not in ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra']:
    magmom = f'MAGMOM = {site2-1}*0.0 1*3.0 {mag_a-site2}*0.0 {mag_b}*0.0 {mag_x}*0.0'
    ldau_type = 2
    ldau_L = '2 2 2 0 0'
    ldau_U = '6 8 6 0 0'
    ldau_J = '0 0 0 0 0'
    ldau_line = f'LDAU = .TRUE. \nLDAUTYPE = {ldau_type}\nLDAUL = {ldau_L} \nLDAUU = {ldau_U} \n' \
                f'LDAUJ = {ldau_J} \nLDAUPRINT = 1\n'
    PP_names = [PP_dictionary[elems[0]], PP_dictionary[elems[0]], PP_dictionary[elems[0]], PP_dictionary[elems[1]],
                PP_dictionary[elems[2]]]
    filenames = [PP_path + PP_names[0] + '/POTCAR', PP_path + PP_names[1] + '/POTCAR',
                 PP_path + PP_names[2] + '/POTCAR', PP_path + PP_names[3] + '/POTCAR',
                 PP_path + PP_names[4] + '/POTCAR']
    with open('POTCAR', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    elems_poscar = f'{elems[0]} {elems[0]} {elems[0]} {elems[1]} {elems[2]}\n'
    elem_num_poscar = f'{site2-1} 1 {mag_a-site2} {mag_b} {mag_x}\n'
    contents[5] = elems_poscar
    contents[6] = elem_num_poscar
    with open('POSCAR', 'w') as outfile:
        for item in contents:
            outfile.write("%s" % item)
    poscar_elongate('POSCAR', site2)

else:
    magmom = f'MAGMOM = {mag_a}*0.0 {site1-mag_a-1}*0.0 1*3.0 {mag_a+mag_b-site1}*0.0 {mag_x}*0.0'
    ldau_line = "\n"
    print("Check this config")
    PP_names = [PP_dictionary[elems[0]], PP_dictionary[elems[1]], PP_dictionary[elems[1]], PP_dictionary[elems[1]],
                PP_dictionary[elems[2]]]
    filenames = [PP_path + PP_names[0] + '/POTCAR', PP_path + PP_names[1] + '/POTCAR',
                 PP_path + PP_names[2] + '/POTCAR', PP_path + PP_names[3] + '/POTCAR',
                 PP_path + PP_names[4] + '/POTCAR']
    with open('POTCAR', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    elems_poscar = f'{elems[0]} {elems[1]} {elems[1]} {elems[1]} {elems[2]}\n'
    elem_num_poscar = f'{mag_a} {site1-mag_a-1} 1 {mag_a+mag_b-site1} {mag_x}\n'
    contents[5] = elems_poscar
    contents[6] = elem_num_poscar
    with open('POSCAR', 'w') as outfile:
        for item in contents:
            outfile.write("%s" % item)
    poscar_elongate('POSCAR', site1)

# 27 is initial 29 is final
with open('INCAR', 'r') as incar:
    lines = incar.readlines()

with open('INCAR', 'w') as incar:
    for i, line in enumerate(lines):
        if i == 1:
            incar.write(f'NELECT = {np.int(nelect)} \n' \
                        f'{ldau_line} \n' \
                        f'{magmom} \n')
        incar.write(line)
