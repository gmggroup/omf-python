import numpy as np
import matplotlib.pyplot as plt
import z_order_utils
from mpl_toolkits.mplot3d import axes3d, Axes3D

def _get_vecs(bc, bs):
    
    vec_x = np.arange(bc[0] + 1) * np.repeat(bs[0], bc[0] + 1)
    vec_y = np.arange(bc[1] + 1) * np.repeat(bs[1], bc[1] + 1)
    vec_z = np.arange(bc[2] + 1) * np.repeat(bs[2], bc[2] + 1)
    return vec_x, vec_y, vec_z


def _plot_rbm(bc, bs, corner, vecs=None, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.subplot(111, projection='3d')

    if vecs is None:
        vec_x, vec_y, vec_z = _get_vecs(bc, bs)
    else:
        vec_x, vec_y, vec_z = vecs

    x_lines = []
    for z in vec_z:
        for y in vec_y:
            x_lines += [[vec_x[0], y, z], [vec_x[-1], y, z], [np.nan]*3]
    x_lines = np.array(x_lines)


    y_lines = []
    for z in vec_z:
        for x in vec_x:
            y_lines += [[x, vec_y[0], z], [x, vec_y[-1], z], [np.nan]*3]

    z_lines = []
    for x in vec_x:
        for y in vec_y:
            z_lines += [[x, y, vec_z[0]], [x, y, vec_z[-1]], [np.nan]*3]


    X, Y, Z = np.array(x_lines), np.array(y_lines), np.array(z_lines)
    XS = np.r_[X[:, 0], Y[:, 0], Z[:, 0]] + corner[0]
    YS = np.r_[X[:, 1], Y[:, 1], Z[:, 1]] + corner[1]
    ZS = np.r_[X[:, 2], Y[:, 2], Z[:, 2]] + corner[2]
    plt.plot(XS, YS, zs=ZS)
    
    return ax
    
    
def plot_rbm(rbm):
    _plot_rbm(rbm.block_count, rbm.block_size, rbm.corner)

def plot_tbm(tbm):
    vecs = (
        np.r_[0, np.cumsum(tbm.tensor_u)],
        np.r_[0, np.cumsum(tbm.tensor_v)],
        np.r_[0, np.cumsum(tbm.tensor_w)]
    )
    _plot_rbm(tbm.block_count, np.nan, tbm.corner, vecs=vecs)

def plot_rsbm(rsbm, ax=None):
    pbc = rsbm.parent_block_count
    pbs = rsbm.parent_block_size
    isb = rsbm.is_sub_blocked
    sbc = rsbm.sub_block_count
    sbs = rsbm.sub_block_size
    corner = rsbm.corner
    
    ax = _plot_rbm(pbc, pbs, corner, ax=ax)
    
    for k in range(0, pbc[2]):
        for j in range(0, pbc[1]):
            for i in range(0, pbc[0]):
                ind = rsbm._get_parent_index([i, j, k])
                if isb[ind]:
                    sub_corner = corner + pbs * np.array([i, j, k])
                    _plot_rbm(sbc, sbs, sub_corner, ax=ax)

                    
def plot_osbm(osbm, ax=None):
    pbc = osbm.parent_block_count
    pbs = osbm.parent_block_size
    isb = osbm.is_sub_blocked
    corner = osbm.corner
    
    max_lvl = z_order_utils.level_width(0)
    
    ax = _plot_rbm(pbc, pbs, corner, ax=ax)
    
    vec_x, vec_y, vec_z = _get_vecs(pbc, pbs)
    
    def plot_block(index, corner):
        pnt, lvl = z_order_utils.get_pointer(index)
        bs = [s * (z_order_utils.level_width(lvl) / max_lvl) for s in pbs]
        cnr = [
            c + s * (p / max_lvl)
            for c, p, s in zip(corner, pnt, pbs)
        ]
        _plot_rbm([1, 1, 1], bs, cnr, ax=ax)
    
    for parent_index, tree in enumerate(osbm._get_forest()):
        i, j, k = np.unravel_index(
            parent_index,
            osbm.parent_block_count,
            order='F'
        )
        parent_corner = vec_x[i], vec_y[j], vec_z[k]

        for block in tree:
            if block == 0:
                # plotted as a parent above
                continue
            plot_block(block, parent_corner)
    
    
def plot_asbm(asbm, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.subplot(111, projection='3d')
    
    def plot_block(centroid, size):
        cnr = [
            c - s / 2.0
            for c, s in zip(centroid, size)
        ]
        _plot_rbm([1, 1, 1], size, cnr, ax=ax)
        
    for centroids, sizes in asbm._get_lists():
        for i in range(centroids.shape[0]):
            plot_block(centroids[i, :], sizes[i, :])