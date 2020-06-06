import numpy as np
from codes.utils import analyze, discretize, append_cons_ts, structural_form
import argparse
import os


parser = argparse.ArgumentParser("Arguments for analyzing the environment for causal concept understanding")
parser.add_argument("--start", type = int, required = True, help = "Starting height of environment")
parser.add_argument("--stop", type = int, required = True, help = "Maximum height of the environment")
parser.add_argument('--game_type', default = "bw", choices = ["bw", "all_random_invert", "all_random", "trigger_non_markov", "trigger_non_markov_random"], help = "Type of game", required = True)
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
    c_dict = f["c_dict"][0]
    sum = sum + inp.shape[0]
    # result = analyze(inp[:,1:], c_dict)
    if all_data is None:
        all_data = inp
    else:
        all_data = np.concatenate((all_data, inp), axis=0)

print(all_data.shape)
print(all_data[:2])
print(c_dict)
# discrete = discretize(all_data[:,1:], c_dict)
# result = append_cons_ts(discrete)
# result = result.astype("int")
#
# names = ['bias', 'ax_t1', 'ay_t1', 'ac_t1', 'ux_t1', 'uy_t1', 'uc_t1', 'dx_t1',
#          'dy_t1', 'dc_t1', 'lx_t1', 'ly_t1', 'lc_t1', 'rx_t1', 'ry_t1', 'rc_t1', 'a_t1', 'num_s_t1',
#          'ax_t2', 'ay_t2', 'r_t1']
#
for i in range(4):
    action_data = all_data[all_data[:, 16] == i]
    # print("action {}".format(i))
    # print(action_data[:4])
    s_form = structural_form(action_data)
    # print(s_form[:2])
    np.savez(data_dir + "oo_action_{}_{}.npz".format(i, args.game_type), mat = s_form, c_dict = [c_dict])

# # # names = ['ax_t1', 'ay_t1', 'ac_t1', 'ux_t1', 'uy_t1', 'uc_t1', 'dx_t1',
# # 'dy_t1', 'dc_t1', 'rx_t1', 'ry_t1', 'rc_t1', 'lx_t1', 'ly_t1', 'lc_t1', 'a_t1',
# # 'ax_t2', 'ay_t2', 'ac_t2', 'ux_t2', 'uy_t2', 'uc_t2', 'dx_t2',
# # 'dy_t2', 'dc_t2', 'rx_t2', 'ry_t2', 'rc_t2', 'lx_t2', 'ly_t2', 'lc_t2', 'a_t2']
#
