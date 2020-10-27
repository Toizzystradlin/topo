import autograd.numpy as np
from neural_structural_optimization import autograd_lib
#rom neural_structural_optimization import caching

def real_density(x, args):
    shape = (args['nely'], args['nelx'])
    #x = x.reshape(shape)

    x_designed = sigmoid_with_constrained_mean(x, args['volfrac'])
    #print(x_designed)
   # x = x_designed.reshape(shape)
    #print('first x: ' + str(x.shape))
    #print('x_designed:' + str (x_designed.shape))
    np.reshape(x_designed, shape)
    x = x_designed
    #print('x:' + str(x.shape))
    #print(np.min(x))
    #print(np.max(x))
    return x_designed


def make_stiffness_matrix(young, poisson):
    e, nu = young, poisson
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                  -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    return e/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                 [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                 [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                 [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                 [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                 [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                 [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                 [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
                                 ])



def _get_dof_indices(freedofs, fixdofs, k_xlist, k_ylist):
    index_map = autograd_lib.inverse_permutation(
        np.concatenate([freedofs, fixdofs]))
    keep = np.isin(k_xlist, freedofs) & np.isin(k_ylist, freedofs)
    i = index_map[k_ylist][keep]
    j = index_map[k_xlist][keep]
    return index_map, keep, np.stack([i, j])


def displace(x_phys, ke, forces, freedofs, fixdofs, args, penal=3, e_min=1e-9, e_0=1):
    stiffness = e_min + x_phys ** 3 * (e_0 - e_min)
    #np.reshape(stiffness, (args['nely'], args['nelx']))
    k_entries, k_ylist, k_xlist = get_k(stiffness, ke, args)

    index_map, keep, indices = _get_dof_indices(
        freedofs, fixdofs, k_ylist, k_xlist
    )
    u_nonzero = autograd_lib.solve_coo(k_entries[keep], indices, forces[freedofs],
                                       sym_pos=True)
    u_values = np.concatenate([u_nonzero, np.zeros(len(fixdofs))])

    return u_values[index_map]


def get_k(stiffness, ke, args):
    np.reshape(stiffness, (args['nelx'], args['nely']))
    nely, nelx = stiffness.shape

    ely, elx = np.meshgrid(range(nely), range(nelx))
    ely, elx = ely.reshape(-1, 1), elx.reshape(-1, 1)

    n1 = (nely+1)*(elx+0) + (ely+0)
    n2 = (nely+1)*(elx+1) + (ely+0)
    n3 = (nely+1)*(elx+1) + (ely+1)
    n4 = (nely+1)*(elx+0) + (ely+1)
    edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])
    edof = edof.T[0]
    x_list = np.repeat(edof, 8)
    y_list = np.tile(edof, 8).flatten()
    kd = np.reshape(stiffness.T, (nelx*nely, 1, 1))
    value_list = (kd * np.tile(ke, kd.shape)).flatten()
    return value_list, y_list, x_list


def compliance(x_phys, u, ke, *, e_min=1e-9, e_0=1):
    nely, nelx = x_phys.shape
    ely, elx = np.meshgrid(range(nely), range(nelx))

    n1 = (nely+1)*(elx+0) + (ely+0)
    n2 = (nely+1)*(elx+1) + (ely+0)
    n3 = (nely+1)*(elx+1) + (ely+1)
    n4 = (nely+1)*(elx+0) + (ely+1)
    all_ixs = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])

    u_selected = u[all_ixs]

    ke_u = np.einsum('ij,jkl->ikl', ke, u_selected)
    ce = np.einsum('ijk,ijk->jk', u_selected, ke_u)
    C = e_min + x_phys ** 3 * (e_0 - e_min) * ce.T
    return np.sum(C)


def logit(p):
    p = np.clip(p, 0, 1)
    return np.log(p) - np.log1p(-p)


def sigmoid_with_constrained_mean(x, density):

    def sigmoidd(x, y):
        #return ((np.tanh(0.5 * (x + y)) * 0.5 + 0.5) ).mean() - density
        return (np.tanh(0.5*(x + y))).mean() - density
    lower_bound = logit(density) - np.max(x)
    upper_bound = logit(density) - np.min(x)
    b = autograd_lib.find_root(sigmoidd, x, lower_bound, upper_bound)
    #print('b:' + str(b))
    #print('x:' + str(x))
    #print('tanh(x+b):' + str(np.tanh(x+b)))
    #return np.tanh(0.4*(x + b)*0.7) + 0.5
    return np.tanh(0.3*(x + b)) + 0.3
    #return np.tanh(x + 0.5)


def objective(x, ke, args):
    np.reshape(x, (args['nely'], args['nelx']))
    kwargs = dict( e_min=args['young_min'], e_0=args['young'])
    x_phys = real_density(x, args)
    #forces = args['forces']
    u = displace(x_phys, ke, args['forces'], args['freedofs'], args['fixdofs'], args)
    c = compliance(x_phys, u, ke, **kwargs)
    return c


