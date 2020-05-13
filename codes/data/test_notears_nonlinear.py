from structural_generation import *
from notears_nonlinear import *
import argparse
import igraph as ig
import matplotlib.pyplot as plt
from codes.utils import plot_weight_sem
import os

actions = {
0: "up",
1: "down",
2: "left",
3: "right",
}


parser = argparse.ArgumentParser("Arguments for environment generation for causal concept understanding")
parser.add_argument('--mode', default = "eval", choices = ['train', 'eval', 'both'], help ='Train or Evaluate')
parser.add_argument('--disp', default = False, help = 'True or False')
parser.add_argument('--game_type', default = "bw", choices = ["bw", "trigger", "all_random"], help = "Type of game", required = True)
parser.add_argument('--l1', default = 0.01, type = float, help = 'lambda 1: penalty for regularizer')
parser.add_argument('--l2', default = 0.01, type = float, help = 'lambda2: penalty for regularizer')

args = parser.parse_args()

vars = ['bias', 'ax_t1', 'ay_t1', 'ac_t1', 'ux_t1', 'uy_t1', 'uc_t1', 'dx_t1',
         'dy_t1', 'dc_t1', 'lx_t1', 'ly_t1', 'lc_t1', 'rx_t1', 'ry_t1', 'rc_t1',
         'ax_t2', 'ay_t2']

plot_dir = "./codes/plots/{}/lambda1_{}_lambda2_{}".format(args.game_type, args.l1, args.l2)
data_dir = "./codes/data/mat/{}/matrices/".format(args.game_type)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def plot_graph(W, action, type):
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
    out = ig.plot(g, plot_dir + "W_{}_{}.pdf".format(action, type), **visual_style)
    # out.save()

for i in range(4):
    X, W_true, Z = generate_structure(args.height, args.n_data, i)
    plot_graph(W_true, actions[i], "true")
    filename = data_dir + "oo_action_{}_{}.npz".format(i, args.game_type)
    f = np.load(filename, mmap_mode='r', allow_pickle=True)
    X_all = f["mat"]
    c_dict = f["c_dict"]
    X_all = X_all.astype("int")
    print(actions[i], X_all[:4], c_dict)
    idx = np.random.shuffle(np.arange(X_all.shape[0]))
    X_train = np.squeeze(X_all[idx], axis = 0)

    p = [0,1,2,3,6,9,12,15]
    q = []

    for j in range(len(vars)):
        if j not in p:
            q.append(j)
    Z = np.zeros(X_train.shape)
    Z[:,p] = X_train[:,p]

    model = NotearsMLP(dims=[X_train.shape[1], 10, 1], bias=True)
    if args.mode in ["train", "both"]:
        W_est = notears_nonlinear(model, X, Z, lambda1=0.01, lambda2=0.01)
        # W_est = notears_linear(X_train, Z, lambda1 = args.l, rho = args.rho, alpha = args.alpha, disp = args.disp)
        np.savez(data_dir + 'W_est_{}.npz'.format(actions[i]), w = W_est)
        np.savez(data_dir + 'W_true_{}.npz'.format(actions[i]), w = W_true)
    else:
        W_est = np.load(data_dir + 'W_est_{}.npz'.format(actions[i]))["w"]

    plot_graph(W_est, actions[i], "est")
    true_plot_name = plot_dir + "w_true_{}".format(actions[i])
    est_plot_name = plot_dir + "w_est_{}".format(actions[i])

    x_indices = q
    y_indices = p
    y_label = [vars[k] for k in p]
    x_label = [vars[k] for k in q]
    plot_weight_sem(W_true, true_plot_name, x_indices, y_indices, x_label, y_label)
    plot_weight_sem(W_est, est_plot_name, x_indices, y_indices, x_label, y_label)