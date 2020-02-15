import numpy as np
eng = np.load("transition_matrix_eng.npz")['mat']
x_eng = eng[:-1,4,:]
y_eng = eng[1:,4,:]

# Actual matrix x
x = np.load("transition_matrix.npz")['mat']
y = x[1:,4,9:17]
x = x[:-1,4,:]
N = x.shape[0]
M = x.shape[1]

# Schema Network

w = np.load("w_optimal.npz")['w']

a = w[0, :, :4]
print(a)
