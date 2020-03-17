import numpy as np


actions = {
0: "up",
1: "down",
2: "left",
3: "right"
}
def analyze(inp, colors_dict):
    print(inp.shape)
    result = np.zeros((inp.shape[0], 17), dtype=np.dtype('a16'))
    for i in range(len(inp)):
        result[i, 0] = inp[i,0]
        idx = 0
        count = 1
        for j in range(5):
            result[i, count] = inp[i, idx + 1]
            result[i, count + 1] = inp[i, idx + 2]
            color_ohe = decode_onehot(inp[i,idx + 3: idx + 8])
            result[i, count + 2] = colors_dict[color_ohe]
            idx = idx + 7
            count = count + 3
        result[i, -1] = actions[decode_onehot(inp[i, 36:40])]
    return result

def decode_onehot(a):
    return np.where(a == 1)[0][0]

f = np.load("./mat/oo_transition_matrix.npz", mmap_mode='r', allow_pickle=True)
inp = f["mat"][:,0,:]
c_dict = f["c_dict"][0]
result = analyze(inp, c_dict)
print(result.shape)
print(result[:8])
