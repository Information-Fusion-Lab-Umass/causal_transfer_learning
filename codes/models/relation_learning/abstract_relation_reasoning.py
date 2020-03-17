import numpy as np
from simple_nn import RelationalNN
from sklearn.model_selection import train_test_split
import torch
import sys
sys.path.insert(0, '../utils/')
from causal_transfer_learning.experiments.utils import analyze

#load data
f = np.load("../data/mat/oo_transition_matrix.npz", mmap_mode='r', allow_pickle=True)
inp = f["mat"][:,0,:]
c_dict = f["c_dict"][0]

# get input indexed at even indices
even_indices = [i for i in range(inp.shape[0]) if i % 2 == 0]

# get only outputs indexed at odd indices
odd_indices = [i for i in range(inp.shape[0]) if i % 2 != 0]

x = inp[even_indices,:]
y = inp[odd_indices,:]

print(analyze(x[:8],c_dict))
print(analyze(y[:8],c_dict))

# N = x.shape[0]
# x_dim = x.shape[1]
# y_dim = y.shape[1]
# h_dim = 10
#
# # split data in train and test
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)
# xtr, ytr = torch.from_numpy(X_train).type(torch.FloatTensor), torch.from_numpy(y_train).type(torch.FloatTensor)
# xte, yte = torch.from_numpy(X_test).type(torch.FloatTensor), torch.from_numpy(y_test).type(torch.FloatTensor)
#
# # define model
# model = RelationalNN(x_dim, h_dim, y_dim)
# print(xtr.dtype, ytr.dtype)
# n_epochs = 100
# for i in range(n_epochs):
#     model.train(i, xtr, ytr)
#     model.test(i, xte, yte)
