import numpy as np
from scipy.optimize import linprog
from itertools import repeat
import cvxpy as cp
from sklearn.metrics import accuracy_score

#
# # Define eng, x_eng, y_eng for interpretability

eng = np.load("transition_matrix_eng.npz")['mat']
x_eng = eng[:-1,4,:]
y_eng = eng[1:,4,:]

# Actual matrix x
x = np.load("transition_matrix.npz")['mat']
y = x[1:,4,9:17]
x = x[:-1,4,:]
N = x.shape[0]
M = x.shape[1]

# # Define (1-x)
c = (1 - x)
# Select output from t+1 for yellow object and select only position coordinates.

Z = y.shape[1]

# # select points where y is equal to 1
#
# Loop over different attributes
L = 20
w_optimal = np.zeros((Z, M, L))
for z in [0,7]:
    print("                                                            ")
    print("################Attribute {}######################".format(z))
    y_n1 = np.where(y[:,z] == 1)[0]
    y_n0 = np.where(y[:,z] == 0)[0]
    solved = np.array([], dtype = "int")
    c_1 = c[y_n1]
    c_0 = c[y_n0]
    y_l1 = y_n1
    j = 0

    # for i in range(1):
    #     print(x_eng[y_n1][i])
    #     print(y_eng[y_n1][i])
    while True:
        if j >= L:
            break

        if len(solved) == len(y_n1):
            break
        print("                                                            ")
        print("###########Iteration {}##############".format(j))
        c_solved = c[solved]
        y_l1 = np.setdiff1d(y_n1, solved)
        print("Total one attribute {}".format(len(y_l1)))
        print("Total zero attribute {}".format(len(y_n0)))
        c_l1 = c[y_l1]
        print("Total solved rows {}".format(len(solved)))

        w = cp.Variable(M, boolean = True)
        objective = sum(c_l1 @ w)
        constraints = []
        for i in range(c_0.shape[0]):
            constraints.append(c_0[i] * w >= 1)

        for i in range(c_solved.shape[0]):
            constraints.append(c_solved[i] * w == 0)

        for i in range(M):
            constraints.append(w[i] <= 1)

        for i in range(M):
            constraints.append(w[i] >= 0)

        problem = cp.Problem(cp.Minimize(objective), constraints = constraints)
        problem.solve()
        print("status:", problem.status)
        print("optimal value:", problem.value)
        # print("optimal var:", w.value)

        w_opt = np.zeros_like(w.value)
        w_opt[w.value > 0.5] = 1
        w_opt[w.value < 0.5] = 0
        y_pred_all = c @ w_opt

        curr_solved = np.where(abs(y_pred_all) < 0.5)[0]

        idx = curr_solved
        for i in range(3):
            print("x {}".format(x_eng[idx][i]))
            print("y {}".format(y_eng[idx][i]))
            print("input {}".format(x[idx][i]))
            print("opt_w {}".format(w_opt))
            print(np.dot(1-x[idx][i], w_opt))

        solved = np.union1d(curr_solved, solved)
        w_optimal[z, :, j] = w_opt
        j = j + 1

    y_pred = c @ w_optimal[z,:,:j]
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    out = (1 - y_pred) @ np.ones(y_pred.shape[1])
    out[out > 0.5] = 1
    out[out < 0.5] = 0

    print("Attribute accuracy score {}".format(accuracy_score(out, y[:,z])))

np.savez("w_optimal.npz", w = w_optimal)
# print(w_optimal[7])
