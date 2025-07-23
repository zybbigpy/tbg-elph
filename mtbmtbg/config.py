import numpy as np
from math import pi, sqrt


def _set_rt_mtrx(theta: float) -> np.ndarray:
    """set 2D rotation matrix 

    Args:
        theta (float): rotation angle in radius

    Returns:
        np.ndarray: 2x2 rotation matrix
    """
    rt_mtrx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    return rt_mtrx

class Structure():
    """ parameters for graphene structure
    """
    # lattice constant (angstrom)
    A_C = 2.4683456
    # parameter for hBN
    # A_C = 2.502
    # old prameter for Graphene
    # A_C = 2.48
    # C-C bond
    A_EDGE = A_C/np.sqrt(3)

    # moire information (angstrom)
    D1_LAYER = 3.433333333
    D2_LAYER = 0.027777778
    D_AB = 3.35

    # hBN Parameter
    # D1_LAYER = 6.617
    # D2_LAYER = 0
    # D_AB = 0


    def __init__(self, symm: str='d3', rot: float=0):
        # unit vector for atomic system
        self.a_unitvec_1 = np.array([np.sqrt(3)*Structure.A_C/2, -Structure.A_C/2])
        self.a_unitvec_2 = np.array([np.sqrt(3)*Structure.A_C/2, Structure.A_C/2])

        self.a_g_unitvec_1 = np.array([2*np.pi/(3*Structure.A_EDGE), -2*np.pi/(np.sqrt(3)*Structure.A_EDGE)])
        self.a_g_unitvec_2 = np.array([2*np.pi/(3*Structure.A_EDGE), 2*np.pi/(np.sqrt(3)*Structure.A_EDGE)])

        rot_mat = _set_rt_mtrx(rot)

        self.a_unitvec_1 = self.a_unitvec_1@rot_mat
        self.a_unitvec_2 = self.a_unitvec_2@rot_mat
        self.a_g_unitvec_1 = self.a_g_unitvec_1@rot_mat
        self.a_g_unitvec_2 = self.a_g_unitvec_2@rot_mat

        self.atom_k_1 = -1/3*self.a_g_unitvec_1+1/3*self.a_g_unitvec_2
        self.atom_k_2 = 1/3*self.a_g_unitvec_1-1/3*self.a_g_unitvec_2

        if symm=='d3':
            self.atom_pstn_1 = 0*self.a_unitvec_1+0*self.a_unitvec_2
            self.atom_pstn_2 = 2/3*self.a_unitvec_1+2/3*self.a_unitvec_2
        
        elif symm=='d6':
            self.atom_pstn_1 = 1/3*self.a_unitvec_1+1/3*self.a_unitvec_2
            self.atom_pstn_2 = 2/3*self.a_unitvec_1+2/3*self.a_unitvec_2
        else:
            self.atom_pstn_1 = 0*self.a_unitvec_1+0*self.a_unitvec_2
            self.atom_pstn_2 = 2/3*self.a_unitvec_1+2/3*self.a_unitvec_2

    def print_info(self):

        print("="*50)
        print("atomic unit vectors:", self.a_unitvec_1, self.a_unitvec_2)
        print("atomic reciprocal vectors:", self.a_g_unitvec_1, self.a_g_unitvec_2)
        print("atomic K K' points:", self.atom_k_1, self.atom_k_2)
        print("atomic postions:", self.atom_pstn_1, self.atom_pstn_2)
        print("="*50)


    # # atom postion in graphene, for genarating D3 TBG strucuture
    # ATOM_PSTN_1 = np.array([0, 0])
    # ATOM_PSTN_2 = np.array([2*A_C/np.sqrt(3), 0])
    # # atom postion in graphene, for genararing D6 TBG structure
    # ATOM_PSTN_1_D6 = 1/3*A_UNITVEC_1+1/3*A_UNITVEC_2
    # ATOM_PSTN_2_D6 = 2/3*A_UNITVEC_1+2/3*A_UNITVEC_2
    # # atomic K point
    # ATOM_K_1 = -1/3*A_G_UNITVEC_1+1/3*A_G_UNITVEC_2
    # ATOM_K_2 = 1/3*A_G_UNITVEC_1-1/3*A_G_UNITVEC_2


class TBInfo:
    """ parameters for SK tight binding scheme
    """
    # eV
    VPI_0 = -2.7
    # eV
    VSIGMA_0 = 0.48
    # Ang
    R_RANGE = 0.184*Structure.A_C


class DataType:
    """Different atomic data type
    """
    RIGID = 'rigid'
    RELAX = 'relax'
    CORRU = 'corrugation'
    CORRU_D6 = 'corrugation_d6'
    PHONON = 'phonon'


class EngineType:
    """name for different engine types
    """
    TBPLW = 'TB'
    TBFULL = 'tbfull'
    TBSPARSE = 'tbsparse'


class ValleyType:
    """different type of valleys
    """
    VALLEYK1 = 'valleyk1'
    VALLEYK2 = 'valleyk2'
    # not move
    VALLEYG = 'valleyg'
    # combined two valleys
    VALLEYC = 'valleyc'


class Cont:
    # two paramters, unit eV (chiral limit U1 = U2)
    # U1 = 0.0797
    # U2 = 0.0975
    U1 = 0.0858
    U2 = 0.1032
    # pauli matrices
    SIGMA_X = np.array([[0, 1], [1, 0]])
    SIGMA_Y = np.array([[0, -1j], [1j, 0]])
    SIGMA_Z = np.array([[1, 0], [0, -1]])
    # fermi velocity 2.1354eV*a
    HBARVF = 2.1354*Structure.A_C


class Phonon:
    # Carbon mass
    CARBON_MASS = 12.0107  # [AMU]
    # physical constant
    PlanckConstant = 4.13566733e-15  # [eV s]
    Hbar = PlanckConstant/(2*pi)  # [eV s]
    SpeedOfLight = 299792458  # [m/s]
    AMU = 1.6605402e-27  # [kg]
    EV = 1.60217733e-19  # [J]
    Angstrom = 1.0e-10  # [m]
    THz = 1.0e12  # [/s]
    VaspToTHz = sqrt(EV/AMU)/Angstrom/(2*pi)/1e12  # [THz] 15.633302
    THzToCm = 1.0e12/(SpeedOfLight*100)  # [cm^-1] 33.356410
