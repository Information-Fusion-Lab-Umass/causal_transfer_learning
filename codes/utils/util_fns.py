import numpy as np
actions = {
0: "up",
1: "down",
2: "left",
3: "right"
}

def analyze(inp, colors_dict):
    if not isinstance(inp, np.ndarray):
        inp = inp.detach().numpy()
    result = np.zeros((inp.shape[0], 16), dtype=np.dtype('a16'))
    for i in range(len(inp)):
        idx = 0
        count = 0
        for j in range(5):
            result[i, count] = inp[i, idx]
            result[i, count + 1] = inp[i, idx + 1]
            color_ohe = decode_onehot(inp[i,idx + 2: idx + 7])
            result[i, count + 2] = colors_dict[color_ohe]
            idx = idx + 7
            count = count + 3
        result[i, -1] = actions[decode_onehot(inp[i, 35:39])]
    return result

def decode_onehot(a):
    return np.where(a == 1)[0][0]
