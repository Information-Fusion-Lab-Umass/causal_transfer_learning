import numpy as np
from codes.utils import analyze, discretize, append_cons_ts
current_dir = "./codes/data/"
all_data = None
sum = 0

# def all_discrete(data):
values = []
for i in range(5,10, 5):
    f = np.load("./codes/data/mat/oo_transition_matrix_{}.npz".format(i), mmap_mode='r', allow_pickle=True)
    inp = f["mat"][:,0,:]
    c_dict = f["c_dict"][0]
    sum = sum + inp.shape[0]
    # result = analyze(inp[:,1:], c_dict)
    if all_data is None:
        all_data = inp
    else:
        all_data = np.concatenate((all_data, inp), axis=0)

print(all_data.shape)
result = append_cons_ts(discretize(all_data[:,1:], c_dict))

result = result.astype("int")
names = ['ax_t1', 'ay_t1', 'ac_t1', 'ux_t1', 'uy_t1', 'uc_t1', 'dx_t1',
'dy_t1', 'dc_t1', 'rx_t1', 'ry_t1', 'rc_t1', 'lx_t1', 'ly_t1', 'lc_t1', 'a_t1',
'ax_t2', 'ay_t2', 'ac_t2', 'ux_t2', 'uy_t2', 'uc_t2', 'dx_t2',
'dy_t2', 'dc_t2', 'rx_t2', 'ry_t2', 'rc_t2', 'lx_t2', 'ly_t2', 'lc_t2', 'a_t2']
values = [ np.max(result[:,i]) for i in range(result.shape[1])]
discrete = ['d' for i in range(len(names))]
header = np.array([names, values, discrete])
print(header)
print(result.shape)
print(result[:2])
print(c_dict)
np.savetxt(current_dir + "mat/R/oo_transition_matrix.csv", result, delimiter = ",")
np.savetxt(current_dir + "mat/R/header.csv", header, delimiter = "," , fmt="%s")
np.savez(current_dir + "mat/oo_transition_matrix.npz", mat = all_data, c_dict = [c_dict])
