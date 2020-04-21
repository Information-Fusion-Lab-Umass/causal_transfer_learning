from structural_generation import *
from notears import *
import argparse
import igraph as ig
import matplotlib.pyplot as plt
from codes.utils import plot_weight_sem

actions = {
0: "up",
1: "down",
2: "left",
3: "right",
}
plot_dir = "./codes/data/mat/SEM/plots/"
parser = argparse.ArgumentParser("Arguments for environment generation for causal concept understanding")
parser.add_argument('--height', default= 10, type = int, help='height of the grid')
parser.add_argument('--n_data', default= 100, type = int, help='number of examples')
parser.add_argument('--mode', default = "eval", choices = ['train', 'eval', 'both'], help ='', required = True)

args = parser.parse_args()

vars = ['bias', 'ax_t1', 'ay_t1', 'ac_t1', 'ux_t1', 'uy_t1', 'uc_t1', 'dx_t1',
         'dy_t1', 'dc_t1', 'lx_t1', 'ly_t1', 'lc_t1', 'rx_t1', 'ry_t1', 'rc_t1',
         'ax_t2', 'ay_t2']

def plot_graph(W):
    abs_w = abs(W) > 0
    g = ig.Graph().Adjacency(abs_w.tolist())
    g.es['weight'] = W[W.nonzero()]
    g.vs['label'] = vars
    g.es['label'] = list(map("{:.2f}".format, W[W.nonzero()]))


    visual_style = {}
    # layout = g.layout('kk')
    visual_style["vertex_label_dist"] = 3
    visual_style["margin"] = 20

    # visual_style["layout"] = layout
    ig.plot(g, **visual_style)

for i in range(1):
    X, W_true, X_all, Z = generate_structure(args.height, args.n_data, i)
    plot_graph(W_true)
    X_all = X_all.astype("int")
    idx = np.random.shuffle(np.arange(X_all.shape[0]))
    X_train = np.squeeze(X_all[idx], axis = 0)

    p = [0,1,2,3,6,9,12,15]
    q = []

    for j in range(len(vars)):
        if j not in p:
            q.append(j)
    Z = np.zeros(X_train.shape)
    Z[:,p] = X_train[:,p]
    # M = X_train @ W_true + Z
    # M = M.astype(int)
    # R = X_train - M
    # loss = 0.5 / X.shape[0] * (R ** 2).sum()

    if args.mode in ["train", "both"]:
        W_est = notears_linear(X_train, Z, lambda1=0.1, loss_type='l2')
        np.savez('./codes/data/mat/SEM/csv/W_est_{}.npz'.format(actions[i]), w = W_est)
        np.savez('./codes/data/mat/SEM/csv/W_true_{}.npz'.format(actions[i]), w = W_true)
    else:
        W_est = np.load('./codes/data/mat/SEM/csv/W_est_{}.npz'.format(actions[i]))["w"]

    plot_graph(W_est)
    true_plot_name = plot_dir + "w_true_{}".format(actions[i])
    est_plot_name = plot_dir + "w_est_{}".format(actions[i])

    x_indices = q
    y_indices = p
    y_label = [vars[k] for k in p]
    x_label = [vars[k] for k in q]
    plot_weight_sem(W_true, true_plot_name, x_indices, y_indices, x_label, y_label)
    plot_weight_sem(W_est, est_plot_name, x_indices, y_indices, x_label, y_label)
