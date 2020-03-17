import numpy as np
from causal_transfer_learning.experiments.utils import analyze


f = np.load("./mat/oo_transition_matrix.npz", mmap_mode='r', allow_pickle=True)
inp = f["mat"][:,0,:]
c_dict = f["c_dict"][0]
result = analyze(inp, c_dict)
print(result.shape)
print(result[:8])
