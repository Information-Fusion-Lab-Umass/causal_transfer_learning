import numpy as np
from simple_nn import RelationalNN
from sklearn.model_selection import train_test_split
import torch


# load data
eng = np.load("transition_matrix_eng.npz")['mat']
binary = np.load("transition_matrix.npz")['mat']
mixed = np.load("transition_matrix_mixed.npz")['mat']


even_indices = [i for i in range(eng.shape[0]) if i % 2 == 0]
x_eng = eng[even_indices]

# get only outputs indexed at odd indices
odd_indices = [i for i in range(eng.shape[0]) if i % 2 != 0]
y_eng = eng[odd_indices]

# # Actual matrix x
mixed = np.load("transition_matrix_mixed.npz")['mat']

x = mixed[even_indices,0,:]
y = mixed[odd_indices,0,0:2]


N = x.shape[0]
x_dim = x.shape[1]
y_dim = y.shape[1]
h_dim = 10

# split data in train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)
xtr, ytr = torch.from_numpy(X_train).type(torch.FloatTensor), torch.from_numpy(y_train).type(torch.FloatTensor)
xte, yte = torch.from_numpy(X_test).type(torch.FloatTensor), torch.from_numpy(y_test).type(torch.FloatTensor)

# define model
model = RelationalNN(x_dim, h_dim, y_dim)
print(xtr.dtype, ytr.dtype)
n_epochs = 100
for i in range(n_epochs):
    model.train(i, xtr, ytr)
    model.test(i, xte, yte)
