import numpy as np
from codes.utils import analyze, discretize
current_dir = "./codes/data/"
all_data = None
sum = 0

# def all_discrete(data):

for i in range(5,40, 5):
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
result = discretize(all_data[:,1:], c_dict)
np.savetxt(current_dir + "mat/oo_transition_matrix.csv", result, delimiter = ",")
np.savez(current_dir + "mat/oo_transition_matrix.npz", mat = all_data, c_dict = [c_dict])
