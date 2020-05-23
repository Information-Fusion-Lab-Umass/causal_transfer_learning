from structural_generation import *
from notears_nonlinear import *
import argparse
import igraph as ig
import matplotlib.pyplot as plt
from codes.utils import *
import os
import torch

torch.set_printoptions(precision=3, sci_mode = False)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
actions = {
0: "up",
1: "down",
2: "left",
3: "right",
}


parser = argparse.ArgumentParser("Arguments for environment generation for causal concept understanding")
parser.add_argument('--height', default = 10, type = int, help='Height of the environment')
parser.add_argument('--n_data', default = 100, type = int, help='Number of data points')
parser.add_argument('--mode', default = "eval", choices = ['train', 'eval', 'both'], help ='Train or Evaluate')
parser.add_argument('--disp', default = False, help = 'True or False')
parser.add_argument('--game_type', default = "bw", choices = ["bw", "all_random_invert", "all_random"], help = "Type of game", required = True)
parser.add_argument('--l1', default = 0.01, type = float, help = 'lambda 1: penalty for regularizer')
parser.add_argument('--l2', default = 0.01, type = float, help = 'lambda2: penalty for regularizer')
parser.add_argument('--rho', default = 0.0, type = float, help = 'rho: penalty for regularizer for acyclicity')
parser.add_argument('--save_results', default = 1, choices = [0,1], type = int, help = 'flag to save results')
parser.add_argument('--train_frac', default = 100, type = float, help = 'fraction of data to be trained on')

args = parser.parse_args()

vars = ['bias', 'ax_t1', 'ay_t1', 'ac_t1', 'ux_t1', 'uy_t1', 'uc_t1', 'dx_t1',
         'dy_t1', 'dc_t1', 'lx_t1', 'ly_t1', 'lc_t1', 'rx_t1', 'ry_t1', 'rc_t1',
         'ax_t2', 'ay_t2']

plot_dir = "./codes/plots/{}/train_{}/lambda1_{}_lambda2_{}_rho_{}/".format(args.game_type, args.train_frac, args.l1, args.l2, args.rho)
data_dir = "./codes/data/mat/{}/matrices/".format(args.game_type)
model_dir =  "./codes/data/models/{}/train_{}/".format(args.game_type, args.train_frac)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

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
    filename = data_dir + "oo_action_{}_{}.npz".format(i, args.game_type)
    f = np.load(filename, mmap_mode='r', allow_pickle=True)
    X_all = f["mat"]
    c_dict = f["c_dict"][0]
    X_all = X_all.astype("int")
    train_size = int((args.train_frac/100) * X_all.shape[0])
    print("============= Train Percentage {} Train Data {}============".format(args.train_frac, train_size))
    idx = np.arange(X_all.shape[0])
    np.random.shuffle(idx)
    idx = np.random.choice(idx, size = train_size, replace = False)
    # X_train = np.squeeze(X_all[idx], axis = 0)
    X_train = X_all[idx]

    p = [0,1,2,3,6,9,12,15]
    q = []

    for j in range(len(vars)):
        if j not in p:
            q.append(j)
    Z = np.zeros(X_train.shape)
    Z[:,p] = X_train[:,p]

    model = NotearsMLP(dims=[X_train.shape[1], 10, 1], bias=True)
    model_name = model_dir + "{}_l1_{:.2f}_l2_{:.2f}_rho_{:.2f}".format(actions[i], args.l1, args.l2, args.rho)
    if args.mode in ["train", "both"]:
        W_est = notears_nonlinear(model, X_train, Z, model_name = model_name, rho = args.rho, lambda1=args.l1, lambda2=args.l2)
        # W_est = notears_linear(X_train, Z, lambda1 = args.l, rho = args.rho, alpha = args.alpha, disp = args.disp)

        if args.save_results == 1:
            np.savez(data_dir + 'W_est_{}.npz'.format(actions[i]), w = W_est)
            np.savez(data_dir + 'W_true_{}.npz'.format(actions[i]), w = W_true)
    else:
        if args.save_results == 1:
            W_est = np.load(data_dir + 'W_est_{}.npz'.format(actions[i]))["w"]
        model.load_state_dict(torch.load(model_name))

    with torch.no_grad():
        X_torch = torch.from_numpy(X_train).type(torch.FloatTensor)
        Z_torch = torch.from_numpy(Z).type(torch.FloatTensor)
        train_pred = model(X_torch, Z_torch)
        train_loss = squared_loss(train_pred, X_torch)
        X_eng = analyze(X_torch[1].reshape(1,-1), c_dict)
        print(len(vars), X_eng.shape, train_pred.shape)
        print("Train loss {}".format(train_loss.item()))
        print("============== Action {} ==================".format(actions[i]))
        for j in range(X_torch.shape[1]):
            print(vars[j], X_eng[0, j], train_pred[1, j].item())


    if args.save_results == 1:
        W = model.fc1_to_adj()
        true_plot_name = plot_dir + "w_true_{}".format(actions[i])
        est_plot_name = plot_dir + "w_est_{}".format(actions[i])

        x_indices = q
        y_indices = p
        y_label = [vars[k] for k in p]
        x_label = [vars[k] for k in q]

        # plot_weight_sem(W_true, true_plot_name, x_indices, y_indices, x_label, y_label)
        plot_weight_sem(W, est_plot_name, x_indices, y_indices, x_label, y_label)
