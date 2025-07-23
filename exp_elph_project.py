import numpy as np

import sys

import mtbmtbg.moire_setup as mset
import mtbmtbg.moire_gk as mgk
import mtbmtbg.moire_plot as mplot
from mtbmtbg.moire_shuffle import cont_shuffle_to_tbplw
from mtbmtbg.config import Cont, Structure

import matplotlib.pyplot as plt
import matplotlib.colors as colors

# atomic structure
ATOM = Structure(symm='d3', rot=0)
# reciprocal unit vector for atom system
A_G_UNITVEC_1 = ATOM.a_g_unitvec_1
A_G_UNITVEC_2 = ATOM.a_g_unitvec_2


def _set_kpt(rotmat):

    kpt = (-A_G_UNITVEC_1+A_G_UNITVEC_2)/3
    # print("kpt:", kpt)
    # after rotation
    kpt1 = kpt@rotmat.T
    kpt2 = kpt@rotmat

    return {'kpt1': kpt1, 'kpt2': kpt2}


def _check_eq(vec1, vec2):

    assert vec1.shape == vec2.shape

    if np.linalg.norm(vec1-vec2)<1E-9:
        return True
    else:
        return False


def _set_g_vec_list_valley(n_moire: int, g_vec_list: np.ndarray, m_basis_vecs: dict, valley: int) -> np.ndarray:
    """set Glist containg one specific valley or all valleys

    Args:
        n_moire (int): an integer describe the moire system
        g_vec_list (np.ndarray): original Glist and G[0, 0] = [0, 0]
        m_basis_vecs (dict): moire basis vecs dictionary
        valley (ValleyType.VALLEYK1): valley

    Returns:
        np.ndarray: Glist for computation
    """

    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']
    offset_v1 = n_moire*m_g_unitvec_1+n_moire*m_g_unitvec_2
    offset_v2 = (n_moire+1)*(m_g_unitvec_1+m_g_unitvec_2)


    gv1 = g_vec_list+offset_v1
    gv2 = g_vec_list-offset_v2

    if valley == 1:
        return gv1
    elif valley == -1:
        return gv2
    else:  # default use VALLEYK1
        return gv1


def _make_transfer_const(m_basis_vecs, valley):

    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']

    # three nearest g vec
    g1 = np.array([0, 0])
    g2 = valley*m_g_unitvec_1
    g3 = -valley*m_g_unitvec_2

    omega1, omega2 = np.exp(1j*2*np.pi/3)**valley, np.exp(-1j*2*np.pi/3)**valley

    print("u, u'", Cont.U1, Cont.U2)
    t1 = np.array([[Cont.U1, Cont.U2], [Cont.U2, Cont.U1]])
    t2 = np.array([[Cont.U1, Cont.U2*omega1], [Cont.U2*omega2, Cont.U1]])
    t3 = t2.T

    return (g1, g2, g3, t1, t2, t3)


def _make_t(glist, m_basis_vecs, valley):
    """
    calculate interlayer interaction hamiltonian element
    """
    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']
    glist_size = np.shape(glist)[0]

    tmat = np.zeros((2*glist_size, 2*glist_size), complex)
    (g1, g2, g3, t1, t2, t3) = _make_transfer_const(m_basis_vecs, valley)

    for i in range(glist_size):
        for j in range(glist_size):
            delta_k = glist[j]-glist[i]
            # matrix element in three cases:
            if _check_eq(delta_k, g1):
                tmat[2*i:2*i+2, 2*j:2*j+2] = t1
            if _check_eq(delta_k, g2):
                tmat[2*i:2*i+2, 2*j:2*j+2] = t2
            if _check_eq(delta_k, g3):
                tmat[2*i:2*i+2, 2*j:2*j+2] = t3

    return tmat


def _make_h(glist, k, kpt, rotmat, valley):
    """
    calculate first layer hamiltonian, approximated by dirac hamiltonian
    """

    glist_size = np.shape(glist)[0]
    h1mat = np.zeros((2*glist_size, 2*glist_size), complex)

    for i in range(glist_size):
        q = k+glist[i]-valley*kpt
        #q = q@rotmat
        dirac = Cont.HBARVF*(valley*Cont.SIGMA_X*q[1]-Cont.SIGMA_Y*q[0])
        h1mat[2*i:2*i+2, 2*i:2*i+2] = dirac

    return h1mat


def _make_hamk(k, kpts, glist, rt_mtrx_half, tmat, valley):
    """
    generate total hamiltonian 
    """
    kpt1 = kpts['kpt1']
    kpt2 = kpts['kpt2']

    h1mat = _make_h(glist, k, kpt1, rt_mtrx_half, valley)
    h2mat = _make_h(glist, k, kpt2, rt_mtrx_half.T, valley)
    tmat_cc = np.conj(np.transpose(tmat))
    hamk = np.block([[h1mat, tmat], [tmat_cc, h2mat]])

    return hamk


def _make_intralayer_interv(m_basis_vecs, g, gp):
    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']

    # three nearest g vec
    g1 = np.array([0, 0])
    g2 = m_g_unitvec_1
    g3 = -m_g_unitvec_2

    omega1, omega2 = np.exp(1j*2*np.pi/3), np.exp(-1j*2*np.pi/3)


    t1 = np.array([[gp, g], [g, gp]])
    t2 = np.array([[gp*omega1, g], [g, gp*omega2]])
    t3 = np.array([[gp*omega2, g], [g, gp*omega1]])

    return (g1, g2, g3, t1, t2, t3)




def _make_inter_valley(glist, m_basis_vecs, g, gp, gamma):

    glist_size = np.shape(glist)[0]

    tmat_l1 = np.zeros((2*glist_size, 2*glist_size), complex)
    tmat_l2 = np.zeros((2*glist_size, 2*glist_size), complex)

    (g1, g2, g3, t1, t2, t3) = _make_intralayer_interv(m_basis_vecs, g, gp)


    for i in range(glist_size):
        for j in range(glist_size):
            delta_k = glist[j]-glist[i]
            # matrix element in three cases:
            if _check_eq(delta_k, g1):
                tmat_l1[2*i:2*i+2, 2*j:2*j+2] = t1
            if _check_eq(delta_k, g2):
                tmat_l1[2*i:2*i+2, 2*j:2*j+2] = t2
            if _check_eq(delta_k, g3):
                tmat_l1[2*i:2*i+2, 2*j:2*j+2] = t3


    for i in range(glist_size):
        for j in range(glist_size):
            delta_k = glist[j]-glist[i]
            # matrix element in three cases:
            if _check_eq(delta_k, g1):
                tmat_l2[2*i:2*i+2, 2*j:2*j+2] = t1
            if _check_eq(delta_k, -g2):
                tmat_l2[2*i:2*i+2, 2*j:2*j+2] = t2
            if _check_eq(delta_k, -g3):
                tmat_l2[2*i:2*i+2, 2*j:2*j+2] = t3
    
    # gamma = np.ones(2*glist_size)*g/2.5
    # gamma_mat = np.diag(gamma)

    gamma_m = np.array([[0, gamma],[gamma, 0]])
    gamma_mat = np.kron(np.diag(np.ones(glist_size)), gamma_m)

    print("gamma shape:", gamma_mat.shape)
    zeros = np.zeros((glist_size*2, glist_size*2))
    inter_valley_mat = np.block([[tmat_l1, gamma_mat], [gamma_mat, tmat_l2]])

    return inter_valley_mat



def _make_total_hamk(hamkv1, hamkv2, inter_valley_mat):

    size = hamkv1.shape[0]
    hamk_zeros = np.zeros((size, size), dtype=complex)

    inter_valley_mat_cc = np.conj(np.transpose(inter_valley_mat))
    original = np.block([[hamkv1, hamk_zeros],[hamk_zeros, hamkv2]])
    w, v  = np.linalg.eigh(original)
    n_band = size*2
    elph = np.block([[hamk_zeros, inter_valley_mat], [inter_valley_mat_cc, hamk_zeros]])
    total = np.block([[hamkv1, inter_valley_mat],[inter_valley_mat_cc, hamkv2]])
    projector = v[:,n_band//2-2:n_band//2+2]
    #four_eigs = w[n_band//2-2:n_band//2+2]
    #four_band = np.diag(four_eigs)
    four_band_v = projector.T@total@(np.conj(projector))
    
    # w1, v1 = np.linalg.eigh(four_band_v)
    # print("w1", w1)
    # res = four_band.astype(complex)+four_band_v
    #total = np.block([[hamk_zeros, inter_valley_mat], [inter_valley_mat_cc, hamk_zeros]])
    #total = np.block([[hamkv1, inter_valley_mat],[inter_valley_mat_cc, hamkv2]])

    
    return original, elph

def cont_solver(n_moire: int, n_g: int, n_k: int, g:float, gp:float, gamma:float, disp: bool = True) -> dict:
    """
    continuum model solver for TBG system
    """

    dmesh = []
    emesh = []
    kline = 0
    emax = -1000
    emin = 1000
    count = 1

    # construct moire info
    rt_angle_r, rt_angle_d = mset._set_moire_angle(n_moire)
    rt_mtrx_half = mset._set_rt_mtrx(rt_angle_r/2)
    (_, m_basis_vecs, high_symm_pnts) = mset._set_moire(n_moire)
    # set up g list
    o_g_vec_list = mgk.set_g_vec_list(n_g, m_basis_vecs)
    # move to specific valley or combined valley
    g_vec_list_v1 = _set_g_vec_list_valley(n_moire, o_g_vec_list, m_basis_vecs, 1)
    g_vec_list_v2 = _set_g_vec_list_valley(n_moire, o_g_vec_list, m_basis_vecs, -1)
    # interlayer interaction
    tmat_v1 = _make_t(g_vec_list_v1, m_basis_vecs, 1)
    tmat_v2 = _make_t(g_vec_list_v2, m_basis_vecs, -1)
    # make inter valley
    inter_valley_mat = _make_inter_valley(o_g_vec_list, m_basis_vecs, g, gp, gamma)
    # atomic K points
    kpts = _set_kpt(rt_mtrx_half)
    print("kpnts", kpts)
    print("mg1, mg2", m_basis_vecs['mg1'], m_basis_vecs['mg2'])
    print("mL1, mL2", m_basis_vecs['mu1'], m_basis_vecs['mu2'])
    v0 = np.zeros((488, 4), dtype=complex)
    mk0 = np.zeros((4, 4), dtype=complex)
    
    hkmesh = []
    hk = 0
    elph = 0

    if disp:
        (kline, kmesh) = mgk.set_tb_disp_kmesh(n_k, high_symm_pnts)
    else:
        kmesh = mgk.set_kmesh(n_k, m_basis_vecs)

    for k in kmesh:
        print("k sampling process, counter:", count)
        count += 1
        hamk_v1 = _make_hamk(k, kpts, g_vec_list_v1, rt_mtrx_half, tmat_v1, 1)
        hamk_v2 = _make_hamk(k, kpts, g_vec_list_v2, rt_mtrx_half, tmat_v2, -1)
        hamk, elph = _make_total_hamk(hamk_v1, hamk_v2, inter_valley_mat)
        hk += hamk
        hkmesh.append(hamk)
        #print(hamk)
        #print("hamk shape", hamk.shape)
        #eigen_val, eigen_vec = np.linalg.eigh(elph)
    
        #print("eigs shape", eigen_val.shape)
        #np.savetxt("eigs.txt", eigen_val)
        # print((eigen_val*1e3)**2/(150)**2)
        #emesh.append((eigen_val*1e3)**2/(150)**2)
        #dmesh.append(hamk)
    
    # ksize = kmesh.shape[0]
    # print("ksize:", ksize, kmesh.shape)
    # v0 = v0/ksize
    # Mbar = (v0.T)@elph@v0
    # print(Mbar.shape)
    # eigs, _ = np.linalg.eigh(Mbar)
    # print((eigs*1e3)**2/(150)**2)
    hk = hk/(n_k**2)

    return {'emesh': np.array(emesh), 'dmesh': np.array(dmesh), 'kline': kline, 'kmesh': kmesh,
            'mvecs': m_basis_vecs, 'hkmesh': np.array(hkmesh), 'elph':elph, 'gvec_v1':g_vec_list_v1, 'gvec_v2':g_vec_list_v2,
            'hk': hk}



n_moire = 30
n_g = 5
n_k = int(sys.argv[1])
disp = int(sys.argv[2])
bands = 0

# scale = [i+1 for i in range(8)]

# for s in scale:
#     g = 9.37e-4*s
#     gp = 7.517e-5*s
#     gamma = 2.228e-5*s

#     fig, ax = plt.subplots()
#     ret = cont_solver(n_moire, n_g, n_k, g, gp, gamma)
#     emesh = ret['emesh']
#     kline = ret['kline']
#     mplot.band_plot_module(ax, kline, emesh, n_k, bands)
   # plt.savefig("C://Users//Wangqian Miao//Desktop//elph2cont//"+str(s)+"cont.png", dpi=500)

scale = 14
g = 0.4685*1e-3*scale
gp = 0.03758*1e-3*scale
gamma = 0.01114*1e-3*scale

if disp == 0:
    ret = cont_solver(n_moire, n_g, n_k, g, gp, gamma, disp=False)
else:
    ret = cont_solver(n_moire, n_g, n_k, g, gp, gamma, disp=True) 

hkmesh = ret['hkmesh']
kline = ret['kline']
kmesh = ret['kmesh']
mvecs = ret['mvecs']
elph  = ret['elph']
mL1 = mvecs['mu1']
mL2 = mvecs['mu2']
gvec_v1 = ret['gvec_v1']
gvec_v2 = ret['gvec_v2']
hk = ret['hk']


grid_real_space = []
for i in range(n_k):
    for j in range(n_k):
        pos = i*mL1+j*mL2
        grid_real_space.append(pos)


print("kmesh should be:", n_k, "x", n_k)
print("hkmesh shape:", hkmesh.shape)
np.save('./kmesh_nk_'+str(n_k), kmesh)
np.save('./hkmesh_nk_'+str(n_k),hkmesh)
np.save('./kline_nk_'+str(n_k), kline)
np.save('./real_space_nk_'+str(n_k), np.array(grid_real_space))
np.save('./elph', elph)
np.save('gvec_v1', gvec_v1)
np.save('gvec_v2', gvec_v2)
np.save('hk', hk)
# #print(emesh[0])
#mplot.band_plot_module(ax, kline, emesh, n_k, bands)

# p = ax.scatter(kmesh[:, 0], kmesh[:, 1], c=emesh, vmin=0, vmax=np.max(emesh), s=100, cmap='binary')
# ax.scatter(kmesh[:, 0], kmesh[:, 1], marker='o', s=100, c='w', edgecolors='w', alpha=0.1)
# cbar = fig.colorbar(p, ax=ax)
# plt.show()

# mL1 = mvecs['mu1']
# mL2 = mvecs['mu2']

# kmesh_len = kmesh.shape[0]

# hk_real_space = []
# grid_real_space = []
# for i in range(n_k):
#     for j in range(n_k):
#         pos = i*mL1+j*mL2
#         grid_real_space.append(pos)
#         res = 0
#         print("FT, i, j", i, j)
#         for k in range(kmesh_len):
#             res += np.exp(1j*np.dot(kmesh[k], pos))*dmesh[k]
        
#         print("res shape:", res.shape)
#         hk_real_space.append(res)
    
# grid_real_space = np.array(grid_real_space)
# hk_real_space = np.array(hk_real_space)
# np.save('real_space_grid', grid_real_space)
# np.save('real_space_hk', hk_real_space)
# print(np.imag(hk_real_space))
# hk_real_space = np.abs(hk_real_space)
# print(hk_real_space.shape)

# for i in range(4):
#     for j in range(4):
#         fig, ax = plt.subplots()
#         ax.set_aspect('equal')
#         p = ax.scatter(grid_real_space[:, 0], grid_real_space[:, 1], c=hk_real_space[:,i,j], s=100, cmap='binary')
#         ax.scatter(grid_real_space[:,0], grid_real_space[:, 1], marker='o', s=100, c='w', edgecolors='k', alpha=0.1)
#         cbar = fig.colorbar(p, ax=ax)
#         plt.savefig("/Users/wmiao/Desktop"+str(i)+str(j)+".png")
#         plt.clf()