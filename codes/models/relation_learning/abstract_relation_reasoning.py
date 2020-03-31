import numpy as np
from simple_nn import RelationalNN
from sklearn.model_selection import train_test_split
import torch
from codes.utils import analyze, get_named_layers, plot_weight
import argparse

torch.set_printoptions(sci_mode = False)

actions = {
0: "up",
1: "down",
2: "left",
3: "right",
4: "mixed_all"
}

actions_rev = {
"up": 0,
"down": 1,
"left": 2,
"right": 3,
"mixed_all": 4,
}

parser = argparse.ArgumentParser("Arguments for environment generation for causal concept understanding")

parser.add_argument('--model', default="non_linear", choices = ['linear', 'non-linear'], help='type of model', required = True)
parser.add_argument('--sparse', default = 1, choices = [1, 0], type = int, help='type of model', required = True)
parser.add_argument('--mode', default="both", choices = ['train', 'eval', 'both'], help ='', required = True)
parser.add_argument('--action', default="all", choices = ['up', 'down', 'left', 'right', 'all', 'mixed_all'], help ='', required = True)

args = parser.parse_args()

torch.manual_seed(0)

current_dir = "./codes/models/relation_learning/"
#load data
f = np.load("./codes/data/mat/oo_transition_matrix.npz", mmap_mode='r', allow_pickle=True)
inp = f["mat"][:,0,:]
c_dict = f["c_dict"][0]

# get input indexed at even indices
even_indices = [i for i in range(inp.shape[0]) if i % 2 == 0]

# get only outputs indexed at odd indices
odd_indices = [i for i in range(inp.shape[0]) if i % 2 != 0]

x_all = inp[even_indices,:]
y_all = inp[odd_indices,:]

x_all = x_all[:,1:]
y_all = y_all[:,1:3]

for a in range(5):
    if a == 4 and args.action != "mixed_all":
        continue

    if args.action not in ["all"]:
        if actions_rev[args.action] != a:
            continue

    print("============== Action {} ===============".format(actions[a]))

    if args.mode in ["both", "train"]:
        if args.action == "mixed_all":
            a_indices = np.arange(0, x_all.shape[0], 1)
        else:
            a_indices = np.arange(a, x_all.shape[0], 4)
        x = x_all[a_indices]
        y = y_all[a_indices]
        print(x.shape, y.shape)


        N = x.shape[0]
        x_dim = x.shape[1]
        y_dim = y.shape[1]
        h_dim = 10

        # # split data in train and test
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)
        xtr, ytr = torch.from_numpy(X_train).type(torch.FloatTensor), torch.from_numpy(y_train).type(torch.FloatTensor)
        xte, yte = torch.from_numpy(X_test).type(torch.FloatTensor), torch.from_numpy(y_test).type(torch.FloatTensor)

        if args.model == "non_linear":
            linear_flag = False

        if args.model == "linear":
            linear_flag = True

        model = RelationalNN(x_dim, h_dim, y_dim, c_dict, linear_flag = linear_flag, sparse = args.sparse)
        n_epochs = 200
        for i in range(n_epochs):
            model.train(i, xtr, ytr)
            model.test(i, xte, yte)

        torch.save(model.state_dict(), current_dir + "saved_models/model_" + args.model + "_" + actions[a] + "_sparse_" + str(args.sparse))

    #load model
    if args.mode in ["both", "eval"]:
        eval_model = RelationalNN(x_dim, h_dim, y_dim, c_dict, linear_flag = linear_flag, sparse = args.sparse)
        eval_model.load_state_dict(torch.load(current_dir + "saved_models/model_" + args.model + "_" + actions[a] + "_sparse_" + str(args.sparse)))

        x_eval = torch.from_numpy(x).type(torch.FloatTensor)
        y_eval = torch.from_numpy(y).type(torch.FloatTensor)

        y_pred = eval_model.forward(x_eval)
        eval_loss = eval_model.loss(y_pred, y_eval)
        model.logger.info("Loss on entire data {}".format(eval_loss.item()))
        print(x[:1], y[:1], y_pred[:1])
        print(analyze(x[:1], c_dict), y[:1], y_pred[:1])
        print(model.fcl.weight)
        plot_name = "model_" + args.model + "_" + actions[a] + "_sparse_" + str(args.sparse)
        plot_weight(model.fcl.weight.detach().numpy(), plot_name, dir = current_dir + "saved_models/")
