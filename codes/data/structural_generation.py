import numpy as np

actions = {
0: "up",
1: "down",
2: "left",
3: "right",
4: "mixed_all"
}

def is_white(c):
    if c == 0:
        return 1
    else:
        return 0

def simulate_sem(W, X, Z):
    W = W.reshape(-1,1)
    Z = Z.reshape(-1,1)
    if W.shape[0] != 0:
        output = X @ W
        output = output + Z
    else:
        output = Z
    return output

def generate_structure(height, n_data, action):
    vars = ['bias', 'ax_t1', 'ay_t1', 'ac_t1', 'ux_t1', 'uy_t1', 'uc_t1', 'dx_t1',
             'dy_t1', 'dc_t1', 'lx_t1', 'ly_t1', 'lc_t1', 'rx_t1', 'ry_t1', 'rc_t1',
             'ax_t2', 'ay_t2']

    #Let's assume action is left
    N = len(vars)
    G = np.zeros((N,N))
    W_true = np.zeros((N,N))
    parents = {}
    edges = {}
    w = {}

    for i in range(N):
        for j in range(N):
            p = vars[i]
            c = vars[j]
            edges[(p,c)] = 0
            w[(p,c)] = 0
            parents[j] = []

    # agent's x position at t2
    edges[('ax_t1', 'ax_t2')] = 1
    w[('ax_t1', 'ax_t2')] = 1

    # agent's y position at t2
    edges[('ay_t1', 'ay_t2')] = 1
    w[('ay_t1', 'ay_t2')] = 1

    if actions[action] == "up":
        w[('uc_t1', 'ax_t2')] = -1

    if actions[action] == "down":
        w[('dc_t1', 'ax_t2')] = 1

    if actions[action] == "left":
        w[('lc_t1', 'ay_t2')] = -1

    if actions[action] == "right":
        w[('rc_t1', 'ay_t2')] = 1


    # upper neighbor's x position
    edges[('ax_t1', 'ux_t1')] = 1
    w[('ax_t1', 'ux_t1')] = 1
    w[('bias', 'ux_t1')] = -1

    # upper neighbor's y position
    edges[('ay_t1', 'uy_t1')] = 1
    w[('ay_t1', 'uy_t1')] = 1

    # down neighbor's x position
    edges[('ax_t1', 'dx_t1')] = 1
    w[('ax_t1', 'dx_t1')] = 1
    w[('bias', 'dx_t1')] = 1

    # down neighbor's y position
    edges[('ay_t1', 'dy_t1')] = 1
    w[('ay_t1', 'dy_t1')] = 1

    # left neighbor's x position
    edges[('ax_t1', 'lx_t1')] = 1
    w[('ax_t1', 'lx_t1')] = 1

    # left neighbor's y position
    edges[('ay_t1', 'ly_t1')] = 1
    w[('ay_t1', 'ly_t1')] = 1
    w[('bias', 'ly_t1')] = -1

    # right neighbor's x position
    edges[('ax_t1', 'rx_t1')] = 1
    w[('ax_t1', 'rx_t1')] = 1

    # right neighbor's y position
    edges[('ay_t1', 'ry_t1')] = 1
    w[('ay_t1', 'ry_t1')] = 1
    w[('bias', 'ry_t1')] = 1

    for i in range(N):
        for j in range(N):
            p = vars[i]
            c = vars[j]
            G[i,j] = edges[(p,c)]
            W_true[i,j] = w[(p,c)]
            if W_true[i,j] != 0:
                parents[j].append(i)

    # Data generation using specified structural equation  model
    M = n_data
    Z = np.zeros((M, N))
    X = np.zeros((M, N), dtype = "int")
    n_colors = 2
    # environment's height and width

    #generation of dataset from true parameters
    Z[:,0] = 1
    Z[:,1] = np.random.randint(1, high = height, size = M)
    Z[:,2] = np.random.randint(1, high = height, size = M)
    Z[:,3] = 4

    # agent's up color
    Z[:,6] = np.random.randint(n_colors, size = M)

    # agent's down color
    Z[:,9] = np.random.randint(n_colors, size = M)

    # agent's left color
    Z[:,12] = np.random.randint(n_colors, size = M)

    # agent's right color
    Z[:,15] = np.random.randint(n_colors, size = M)

    ordered_vertices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

    for j in ordered_vertices:
        # print(vars[j], [vars[i] for i in parents[j]], W_true[parents[j], j], X[:, parents[j]], Z[:,j])
        result = simulate_sem(W_true[parents[j], j], X[:, parents[j]], Z[:,j])
        result = np.squeeze(result, axis = 1)
        X[:,j] = result

    X_all = np.load("./codes/data/mat/SEM/oo_s_form_{}.npz".format(action), mmap_mode='r', allow_pickle=True)["mat"]
    # maze = np.zeros((M, height+1, height+1))
    # for i in range(M):
    #     maze[i, X[i,1], X[i,2]] = X[i,3]
    #     maze[i, X[i,4], X[i,5]] = abs(X[i,6] - 1)
    #     maze[i, X[i,7], X[i,8]] = abs(X[i,9] - 1)
    #     maze[i, X[i,10], X[i,11]] = abs(X[i,12] - 1)
    #     maze[i, X[i,13], X[i,14]] = abs(X[i,15] - 1)
    #     start_idx = [[X[i,1], X[i,2]]]
    return X, W_true, X_all, Z
