import math
import numpy as np
import scipy.optimize as optimize
# import matplotlib.pyplot as plt

"""
Script to find B field strength and orientation from measured NV and P1 transitions, or calculate the transitions from
known magnetic field. Here, B_z is aligned along one NV axis, the other NV axes are calculated by rotating the magnetic 
field. 
"""

#### Definitions
# Rotation Matrices between the different crystallographic axes in diamond (e1 parallel to z-axis)
# x-axis is symmetry axis: going from B_x to -B_x changes transitions in NVs 3 and 4
# e1-->e2
R_12 = np.matrix([[3., 0., 0.],
                 [0, -1., -2*math.sqrt(2)],
                 [0., 2*math.sqrt(2), -1.]])/3.

#e1-->e3
R_22 = np.matrix([[-3., 2*math.sqrt(2.), 8.],
                 [6*math.sqrt(2.), 1., 2*math.sqrt(2.)],
                 [0, 6*math.sqrt(2.), -3.]])/9.

# e1-->e4
R_32 = np.matrix([[-3., -2*math.sqrt(2.), -8.],
                 [-6*math.sqrt(2.), 1., 2*math.sqrt(2.)],
                 [0, 6*math.sqrt(2.), -3.]])/9.

# Rotate a vector given in the form [x, y, z] by the rotation matrix R. Works with the way the matrixes are defined here
def rotate(b, r):
    b_vec = np.matrix([[b[0]],
                       [b[1]],
                       [b[2]]])
    b_rot = np.matrix(r.T * b_vec)
    return [float(b_rot[0]), float(b_rot[1]), float(b_rot[2])]


#Spin Operators
#Spin-1-System (NV)
Sx = np.matrix([[0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]])/math.sqrt(2)
Sy = np.matrix([[0, 1, 0],
                [-1, 0, 1],
                [0, -1, 0]])/(math.sqrt(2)*1j)
Sz = np.matrix([[1, 0, 0],
                [0, 0, 0],
                [0, 0, -1]])
S1 = np.matrix([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

#P1 Electron Spin: Spin 1/2
P1_Sx = np.matrix([[0, 1],
                  [1, 0]])/2.
P1_Sy = np.matrix([[0, 1],
                  [-1, 0]])/(2.*1j)
P1_Sz = np.matrix([[1, 0],
                  [0, -1]])/2.
P1_S1 = np.matrix([[1, 0],
                  [0, 1]])

#P1 N14 Nuclear Spin: Spin 1
#seperate naming for clarity
Ix = Sx
Iy = Sy
Iz = Sz
I1 = S1

#constants
muB = 9.274e-24     #J/T
h = 6.626e-34       #Js
#NV
#D = 2.87e9          #Hz doi.org/10.1016/j.jmr.2015.11.005
g_NV = 2.0028        #doi.org/10.1016/j.jmr.2015.11.005
P_NV = -4.8e6       #Hz, doi.org/10.1016/j.jmr.2015.11.005 : Px,y = 1.6MHz, Pz = -3.2MHz
#P1
g_P1 = 2.0024       #Loubser
P = -3.97e6       #Hz
A_para = 114e6    #Hz Loubser: 113.95
A_perp = 81.3e6   #Hz Loubser: 81.304


#Angle between two vectors
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

       #     >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
       #     >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
       #     >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/np.pi    #for °: *180/np.pi

# #Magnetic field
# #B = [float(B[0]), float(B[1]), float(B[2])]
def absoluteValue(B):
    return math.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)

#Calculates the transitions for the 4 different NV orientations
#Output are the two allowed transitions in Hz
def calc_transitions(E, B, D):
    a1 = np.sort(np.linalg.eig(NV_Hamiltonian(E, B, 2.87e9)[0])[0].real)  # eig[0] is the array of eigenvalues (eig[1] = the diagonal matrix)
    a2 = np.sort(np.linalg.eig(NV_Hamiltonian(E, B, 2.87e9)[1])[0].real)
    a3 = np.sort(np.linalg.eig(NV_Hamiltonian(E, B, 2.87e9)[2])[0].real)
    a4 = np.sort(np.linalg.eig(NV_Hamiltonian(E, B, 2.87e9)[3])[0].real)
    return np.concatenate((a1[1:3] - a1[0], a2[1:3] - a2[0], a3[1:3] - a3[0], a4[1:3] - a4[0]), axis=0)


#Calculates the transitions for the 4 different P1 orientations
#Output are the three allowed transitions (per orientation) in Hz
def calc_P1_transitions(B):
    a1 = np.sort(np.linalg.eig(P1_Hamiltonian(B)[0])[0].real)  # eig[0] is the array of eigenvalues (eig[1] = the diagonal matrix)
    a2 = np.sort(np.linalg.eig(P1_Hamiltonian(B)[1])[0].real)
    a3 = np.sort(np.linalg.eig(P1_Hamiltonian(B)[2])[0].real)
    a4 = np.sort(np.linalg.eig(P1_Hamiltonian(B)[3])[0].real)
    return a1[3] - a1[2], a1[4] - a1[1], a1[5] - a1[0],\
           a2[3] - a2[2], a2[4] - a2[1], a2[5] - a2[0],\
           a3[3] - a3[2], a3[4] - a3[1], a3[5] - a3[0],\
           a4[3] - a4[2], a4[4] - a4[1], a4[5] - a4[0]
    # returns lower, central, upper energies (|-1/2,+1> -> |+1/2, +1>; |-1/2,0> -> |+1/2, 0>; |-1/2,-1> -> |+1/2, -1>)
	
#Output are the four forbidden transitions (per orientation) in Hz
def calc_P1_forbidden_transitions(B):
    a1 = np.sort(np.linalg.eig(P1_Hamiltonian(B)[0])[0].real)  # eig[0] is the array of eigenvalues (eig[1] = the diagonal matrix)
    a2 = np.sort(np.linalg.eig(P1_Hamiltonian(B)[1])[0].real)
    a3 = np.sort(np.linalg.eig(P1_Hamiltonian(B)[2])[0].real)
    a4 = np.sort(np.linalg.eig(P1_Hamiltonian(B)[3])[0].real)
    return a1[4] - a1[2], a1[4] - a1[0], a1[3] - a1[1], a1[5] - a1[1], \
           a2[4] - a2[2], a2[4] - a2[0], a2[3] - a2[1], a2[5] - a2[1], \
           a3[4] - a3[2], a3[4] - a3[0], a3[3] - a3[1], a3[5] - a3[1], \
           a4[4] - a4[2], a4[4] - a4[0], a4[3] - a4[1], a4[5] - a4[1]
        # returns energies |-1/2,-1> -> |+1/2, 0>; |-1/2,+1> -> |+1/2,0>; |-1/2,0> -> |+1/2,-1>; |-1/2,0> -> |+1/2,+1>

def calc_NC60_transitions(B):
	a = np.sort(np.linalg.eig(P1_Hamiltonian(B)[0])[0].real)
	return a1[3] - a1[2], a1[4] - a1[1], a1[5] - a1[0]
	
#Hamiltonian of NV centers, returns matrixes for the four orientations
def NV_Hamiltonian(E, B, D = 2.87e9):
    B1 = B
    B2 = rotate(B, R_12)
    B3 = rotate(B, R_22)
    B4 = rotate(B, R_32)

    H1 = np.matrix(D * Sz * Sz + g_NV * muB / h * (B1[0] * Sx + B1[1] * Sy + B1[2] * Sz) + E * Sz * Sz + E * (
    Sy * Sy - Sx * Sx) + E * (Sx * Sy + Sy * Sx))
    H2 = np.matrix(D * Sz * Sz + g_NV * muB / h * (B2[0] * Sx + B2[1] * Sy + B2[2] * Sz) + E * Sz * Sz + E * (
    Sy * Sy - Sx * Sx) + E * (Sx * Sy + Sy * Sx))
    H3 = np.matrix(D * Sz * Sz + g_NV * muB / h * (B3[0] * Sx + B3[1] * Sy + B3[2] * Sz) + E * Sz * Sz + E * (
    Sy * Sy - Sx * Sx) + E * (Sx * Sy + Sy * Sx))
    H4 = np.matrix(D * Sz * Sz + g_NV * muB / h * (B4[0] * Sx + B4[1] * Sy + B4[2] * Sz) + E * Sz * Sz + E * (
    Sy * Sy - Sx * Sx) + E * (Sx * Sy + Sy * Sx))
    return H1, H2, H3, H4

#Hamiltonian of P1 centers, returns matrixes for the four orientations
def P1_Hamiltonian(B):
    B1 = B
    B2 = rotate(B, R_12)
    B3 = rotate(B, R_22)
    B4 = rotate(B, R_32)

    H1 = np.matrix(muB/h * g_P1 * np.kron((B1[0]*P1_Sx + B1[1]*P1_Sy + B1[2]*P1_Sz), S1) +
                   A_perp * np.kron(P1_Sx, Ix) +
                   A_perp * np.kron(P1_Sy, Iy) +
                   A_para * np.kron(P1_Sz, Iz) +
                   P * np.kron(P1_S1, (Ix*Ix + Iy*Iy + Iz*Iz)))
    H2 = np.matrix(muB/h * g_P1 * np.kron((B2[0]*P1_Sx + B2[1]*P1_Sy + B2[2]*P1_Sz), S1) +
                   A_perp * np.kron(P1_Sx, Ix) +
                   A_perp * np.kron(P1_Sy, Iy) +
                   A_para * np.kron(P1_Sz, Iz) +
                   P * np.kron(P1_S1, (Ix*Ix + Iy*Iy + Iz*Iz)))
    H3 = np.matrix(muB/h * g_P1 * np.kron((B3[0]*P1_Sx + B3[1]*P1_Sy + B3[2]*P1_Sz), S1) +
                   A_perp * np.kron(P1_Sx, Ix) +
                   A_perp * np.kron(P1_Sy, Iy) +
                   A_para * np.kron(P1_Sz, Iz) +
                   P * np.kron(P1_S1, (Ix*Ix + Iy*Iy + Iz*Iz)))
    H4 = np.matrix(muB/h * g_P1 * np.kron((B4[0]*P1_Sx + B4[1]*P1_Sy + B4[2]*P1_Sz), S1) +
                   A_perp * np.kron(P1_Sx, Ix) +
                   A_perp * np.kron(P1_Sy, Iy) +
                   A_para * np.kron(P1_Sz, Iz) +
                   P * np.kron(P1_S1, (Ix*Ix + Iy*Iy + Iz*Iz)))
    return H1, H2, H3, H4

def NC60_Hamiltonian(B):
	H = np.matrix(muB/h * g_NC60 * np.kron((B1[0]*NC60_Sx + B1[1]*NC60_Sy + B1[2]*NC60_Sz), S1) +
			   A_perp * np.kron(NC60_Sx, Ix) +
			   A_perp * np.kron(NC60_Sy, Iy) +
			   A_para * np.kron(NC60_Sz, Iz)
			   )
	return H
	
def find_B_NV(f_transitions):
    """
    Calculates the magnetic field that induces the given transition frequencies in an electronic S=1 system (NV-defect).
    Parameters are the 8 transition frequencies (no hyperfine). For the best initial guess, the first pair of values
    should be from the NV with the highest splitting (=best aligned). Format is
    [NV1(lower), NV1(higher), NV2(lower), NV2(higher), NV3(lower), NV3(higher), NV4(lower), NV4(higher)] in Hz
    :param f_transitions: Electron spin transition frequencies
    :return: B-Field array [x, y, z] in NV-coordinate-system
    """

    def f_NV(param):
        return calc_transitions(param[0], [param[1], param[2], param[3]], param[4]) - \
               np.array([f_transitions[0], f_transitions[1], f_transitions[2], f_transitions[3],
                         f_transitions[4], f_transitions[5], f_transitions[6], f_transitions[7]])

    #x0 = E, B, D
    x0 = [0.0,
          0.1 * (2.87000e9 - f_transitions[0]) / 28e9, 0.1 * (2.87000e9 - f_transitions[0]) / 28e9, (2.8700e9 - f_transitions[0]) / 28e9,
          2.87e9]                                   #x0: initial guess, here:  E=0, Bx, By = 0.1*Bz, D=2.88e9
    NV_values = optimize.root(f_NV, x0, method='lm')      #'lm': Solve for least squares with Levenberg-Marquardt. Without this an error message will be displayed.
    # print(NV_values['message'])
    # return NV_values['x'], absoluteValue([NV_values['x'][1], NV_values['x'][2], NV_values['x'][3]])

    B = NV_values['x']
    result = {"B_abs": absoluteValue(B[1:4]), "Bx": B[1], "By": B[2], "Bz": B[3], "theta_z_degrees": angle(B[1:4], (0, 0, 1))}
    return result


def find_B_P1(f_transitions):
    """
    Calculates the magnetic field that induces the given transition frequencies in P1 center.
    Parameters are the 12 transition frequencies. For the best initial guess, the first pair of values
    should be from the P1 with the highest splitting (=best aligned). Format is
    [P1 on axis(lower), P1 on axis (central), P1 on axis (higher), P1 off axis lower, central higher,... in Hz
    :param f_transitions: Electron spin transition frequencies
    :return: B-Field array [x, y, z] in diamond coordinate system (z-axis along one diamond crystallographic orientation)
    """

    def f_P1(param):
        return calc_P1_transitions([param[0], param[1], param[2]]) - \
               np.array([f_transitions[0], f_transitions[1], f_transitions[2],
                         f_transitions[3], f_transitions[4], f_transitions[5],
                         f_transitions[6], f_transitions[7], f_transitions[8],
                         f_transitions[9], f_transitions[10], f_transitions[11]])

    #x0 = B
    x0 = [0.1 * f_transitions[1] / 28e9, 0.1 * f_transitions[1] / 28e9, f_transitions[1] / 28e9]    #x0: initial guess, here:  Bx, By = 0.1*Bz, Bz = central transition 1
    P1_values = optimize.root(f_P1, x0, method='lm')      #'lm': Solve for least squares with Levenberg-Marquardt. Without this an error message will be displayed.
    print(P1_values['message'])
    return P1_values['x'], absoluteValue(P1_values['x'])
