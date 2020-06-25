import numpy as np
from codes.utils import analyze, discretize, append_cons_ts, structural_form
import argparse
import os
import pandas as pd


parser = argparse.ArgumentParser("Arguments for analyzing the environment for causal concept understanding")
parser.add_argument("--start", type = int, required = True, help = "Starting height of environment")
parser.add_argument("--stop", type = int, required = True, help = "Maximum height of the environment")
parser.add_argument('--game_type', default = "bw", choices = ["bw", "all_random_invert", "all_random", "trigger_non_markov", "trigger_non_markov_random", "trigger_non_markov_flip"], help = "Type of game", required = True)
args = parser.parse_args()
np.printoptions(scientific = False)

data_dir = "./codes/data/mat/{}/matrices/".format(args.game_type)
actions = {
0: "up",
1: "down",
2: "left",
3: "right",
}
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

all_data = None
sum = 0

# def all_discrete(data):
values = []
for i in range(args.start, args.stop, 5):
    filename = data_dir + "oo_transition_matrix_{}.npz".format(i)
    f = np.load(filename, mmap_mode='r', allow_pickle=True)
    inp = f["mat"]
    sum = sum + inp.shape[0]
    # result = analyze(inp[:,1:], c_dict)
    if all_data is None:
        all_data = inp
    else:
        all_data = np.concatenate((all_data, inp), axis=0)

print(all_data.shape)
names = ['t_1','ax_t1', 'ay_t1', 'ac_t1', 'ux_t1', 'uy_t1', 'uc_t1', 'dx_t1',
          'dy_t1', 'dc_t1', 'lx_t1', 'ly_t1', 'lc_t1', 'rx_t1', 'ry_t1', 'rc_t1', 'a_t1', 'r_t1', 'num_s_t1']

# df = pd.DataFrame(all_data, columns = names)
# print(df.groupby(["r_t1", "num_s_t1"]).count())

# r_idx = all_data[:,17] > 0
# check = np.arange(all_data.shape[0])[r_idx]
# print(all_data[check[0:4]])
# p = [0,1,2,5,8,11,14,15,17]

for i in range(4):
    action_data = all_data[all_data[:, 16] == i]
    s_form = structural_form(action_data[:, 1:])
    np.savez(data_dir + "oo_action_{}_{}.npz".format(i, args.game_type), mat = s_form)
