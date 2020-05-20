import numpy as np
from codes.utils import analyze, discretize, append_cons_ts, structural_form
import argparse
import os


parser = argparse.ArgumentParser("Arguments for analyzing the environment for causal concept understanding")
parser.add_argument("--start", type = int, required = True, help = "Starting height of environment")
parser.add_argument("--stop", type = int, required = True, help = "Maximum height of the environment")
parser.add_argument('--game_type', default = "bw", choices = ["bw", "all_random_invert", "all_random"], help = "Type of game", required = True)
args = parser.parse_args()

data_dir = "./codes/data/mat/{}/matrices/".format(args.game_type)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

all_data = None
sum = 0

# def all_discrete(data):
values = []
for i in range(args.start, args.stop, 5):
    filename = data_dir + "oo_transition_matrix_{}.npz".format(i)
    f = np.load(filename, mmap_mode='r', allow_pickle=True)
    inp = f["mat"][:,0,:]
    c_dict = f["c_dict"][0]
    sum = sum + inp.shape[0]
    # result = analyze(inp[:,1:], c_dict)
    if all_data is None:
        all_data = inp
    else:
        all_data = np.concatenate((all_data, inp), axis=0)

discrete = discretize(all_data[:,1:], c_dict)
result = append_cons_ts(discrete)
result = result.astype("int")

names = ['bias', 'ax_t1', 'ay_t1', 'ac_t1', 'ux_t1', 'uy_t1', 'uc_t1', 'dx_t1',
         'dy_t1', 'dc_t1', 'lx_t1', 'ly_t1', 'lc_t1', 'rx_t1', 'ry_t1', 'rc_t1', 'a_t1',
         'ax_t2', 'ay_t2']

for i in range(4):
    s_form = structural_form(discrete, action = i)
    print(discrete.shape, s_form.shape)
    # np.savetxt(current_dir + "mat/R/oo_s_form_{}.csv".format(i), s_form[:20], delimiter = ",")
    np.savez(data_dir + "oo_action_{}_{}.npz".format(i, args.game_type), mat = s_form, c_dict = [c_dict])

# names = ['ax_t1', 'ay_t1', 'ac_t1', 'ux_t1', 'uy_t1', 'uc_t1', 'dx_t1',
# 'dy_t1', 'dc_t1', 'rx_t1', 'ry_t1', 'rc_t1', 'lx_t1', 'ly_t1', 'lc_t1', 'a_t1',
# 'ax_t2', 'ay_t2', 'ac_t2', 'ux_t2', 'uy_t2', 'uc_t2', 'dx_t2',
# 'dy_t2', 'dc_t2', 'rx_t2', 'ry_t2', 'rc_t2', 'lx_t2', 'ly_t2', 'lc_t2', 'a_t2']

# values = [ np.max(result[:,i]) for i in range(result.shape[1])]
# discrete = ['d' for i in range(len(names))]
# header = np.array([names, values, discrete])
# np.savetxt(current_dir + "mat/R/oo_transition_matrix.csv", result, delimiter = ",")

# np.savetxt(current_dir + "mat/R/header.csv", header, delimiter = "," , fmt="%s")
# np.savez(current_dir + "mat/oo_transition_matrix.npz", mat = all_data, c_dict = [c_dict])
