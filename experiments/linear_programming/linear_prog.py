import numpy as np
from scipy.optimize import linprog
from itertools import repeat
import cvxpy as cp
from sklearn.metrics import accuracy_score

actions = {
0: "up",
1: "down",
2: "left",
3: "right"
}
color = {
0: "black",
1: "green",
2: "red",
3: "white",
4: "yellow"
}

def analyze(idx, x_eng, y_eng, x, w_opt = None, bin = False):
    for i in range(len(idx)):
        print("x {}".format(x_eng[idx][i]))
        print("y {}".format(y_eng[idx][i]))
        if bin:
            print("input {}".format(decode_w(x[idx][i])))
            if w_opt is not None:
                print("opt_w {}".format(decode_w(w_opt)))

def decode_onehot(a):
    return np.where(a == 1)[0]

def decode_pos(a):
    num = 0
    for b in a:
        num = 2 * num + b
    return num

def decode_w(w_bin):
    result = []
    a = decode_onehot(w_bin[:4])
    if len(a):
        result.append(actions[a[0]])
    else:
        result.append('')

    t = 4
    for i in range(5):
        c = decode_onehot(w_bin[t:t+5])
        t = t+5
        if len(c):
            result.append(color[c[0]])
        else:
            result.append('')
        result.append(decode_pos(w_bin[t:t+4]))
        t = t + 4
        result.append(decode_pos(w_bin[t:t+4]))
        t = t+ 4
    return result

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a])

def binarize(x):
    st = "{0:04b}".format(int(x))
    return st

def print_accuracy(c, w_opt, y):
    y_pred = c @ w_opt
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    out = (1 - y_pred) @ np.ones(y_pred.shape[1])
    out[out > 0.5] = 1
    out[out < 0.5] = 0
    k = out - y
    unsolved = np.where(k!= 0)
    print("Attribute accuracy score {}".format(accuracy_score(out, y)))
    return unsolved[0]

def encode_w(beta):
    w = np.zeros(69)
    if len(beta[0]):
        for k,v in actions.items():
            if v == beta[0]:
                w[:4] = one_hot(k, 4)

    t = 1
    u = 4
    for i in range(5):
        if len(beta[t]):
            for k,v in color.items():
                if v == beta[t]:
                    w[u:u+5] = one_hot(k, 5)
        t = t+1
        u = u + 5
        if beta[t] != 0:
            w[u:u+4] = list(binarize(beta[t]))
        t = t + 1
        u = u + 4

        if beta[t] != 0:
            w[u:u+4] = list(binarize(beta[t]))
        t = t + 1
        u = u + 4
    return w

#
# # Define eng, x_eng, y_eng for interpretability

eng = np.load("transition_matrix_eng.npz")['mat']
print(eng.shape)
# get only yellow objects
eng = eng[:,0,:]

# get only inputs indexed at even indices
even_indices = [i for i in range(eng.shape[0]) if i % 2 == 0]
x_eng = eng[even_indices]

# get only outputs indexed at odd indices
odd_indices = [i for i in range(eng.shape[0]) if i % 2 != 0]
y_eng = eng[odd_indices]

# # Actual matrix x
binary = np.load("transition_matrix.npz")['mat']

x = binary[even_indices,0,:]
y = binary[odd_indices,0,9:17]



N = x.shape[0]
M = x.shape[1]
#
# # # Define (1-x)
c = (1 - x)
# # Select output from t+1 for yellow object and select only position coordinates.
#
Z = 1
#
# # # select points where y is equal to 1
# #
# Loop over different attributes
L = 4
# [action, agent_x, agent_y, left_x, left_y, right_x, right_y, down_x, down_y, up_x, up_y]
beta = [['down', 'yellow', 7, 0, '', 0, 0, '', 0, 0, 'white', 8, 0, '', 0, 0],
        ['down', 'yellow', 8, 0, '', 0, 0, '', 0, 0, 'black', 9, 0, '', 0, 0],
        ['left', 'yellow', 8, 0, '', 0, 0, '', 0, 0, '', 0, 0, '', 0, 0],
        ['right', 'yellow', 8, 0, '', 0, 0, '', 0, 0, '', 0, 0, '', 0, 0],
        ['up', 'yellow', 8, 6, '', 0, 0, '', 0, 0, '', 0, 0, 'red', 7, 0]]

w_gt = np.zeros((M, len(beta)))
for i in range(len(beta)):
    w_gt[:, i] = encode_w(beta[i])

w_optimal = np.zeros((Z, M, L))
for z in range(Z):
    print("                                                            ")
    print("################Attribute {}######################".format(z))
    y_n1 = np.where(y[:,z] == 1)[0]
    y_n0 = np.where(y[:,z] == 0)[0]
    solved = np.array([], dtype = "int")
    c_1 = c[y_n1]
    c_0 = c[y_n0]
    y_l1 = y_n1
    j = 0

    # analyze(y_n1, x_eng, y_eng, x)

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
        print("Total solved rows {} {}".format(len(solved), solved))


        w = cp.Variable(M, boolean = True)
        objective = sum(c_l1 @ w)
        constraints = []
        for i in range(c_0.shape[0]):
            constraints.append(c_0[i] * w >= 1)

        for i in range(M):
            constraints.append(w[i] <= 1)

        for i in range(M):
            constraints.append(w[i] >= 0)

        problem = cp.Problem(cp.Minimize(objective), constraints = constraints)
        problem.solve()
        print("status:", problem.status)
        print("optimal value:", problem.value)
        w_opt = np.zeros_like(w.value)
        w_opt[w.value > 0.5] = 1
        w_opt[w.value < 0.5] = 0
        y_pred_all = c @ w_opt

        curr_solved = np.where(abs(y_pred_all) < 0.5)[0]

        analyze(curr_solved, x_eng, y_eng, x, w_opt=w_opt, bin = True)
        solved = np.union1d(curr_solved, solved)
        w_optimal[z, :, j] = w_opt
        print("opt_w {}".format(decode_w(w_opt)))
        j = j + 1


    # unsolved = print_accuracy(c, w_optimal[0,:,:j], y[:,z])
    # analyze(unsolved, x_eng, y_eng, x, w_opt=w_optimal[0,:,:j], bin = True)

    # unsolved = print_accuracy(c, w_gt, y[:,z])
    # analyze(unsolved, x_eng, y_eng, x, w_opt=w_gt, bin = True)
    #


# np.savez("w_optimal.npz", w = w_optimal)
