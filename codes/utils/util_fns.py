import numpy as np
import matplotlib.pyplot as plt
import torch
actions = {
0: "up",
1: "down",
2: "left",
3: "right"
}

def append_cons_ts(data):
    m,n = data.shape
    m_hat, n_hat = int(m/2), 2*n
    result = np.zeros((m_hat, n_hat))
    for i in range(m_hat):
        result[i,:n] = data[2*i,:]
        result[i,n:n_hat] = data[2*i+1,:]

    return result

def structural_form(data, action = -1):
    m,n = data.shape
    print(m,n)
    m_hat, n_hat = int(m/2), (n+2)
    result = np.zeros((m_hat, n_hat))
    for i in range(m_hat):
        result[i,0] = 1
        result[i,1:16] = data[2*i,:15]
        result[i,16:18] = data[2*i+1,:2]

    if action == -1:
        return result
    else:
        a_indices = np.arange(action, result.shape[0], 4)
        return result[a_indices]

def discretize(inp, colors_dict):
    n_colors = len(colors_dict)
    if not isinstance(inp, np.ndarray):
        inp = inp.detach().numpy()
    result = np.zeros((inp.shape[0], 16))
    for i in range(len(inp)):
        idx = 0
        count = 0
        for j in range(5):
            result[i, count] = inp[i, idx]
            result[i, count + 1] = inp[i, idx + 1]
            color_ohe = decode_onehot(inp[i,idx + 2: idx + 2 + n_colors])
            result[i, count + 2] = color_ohe
            idx = idx + n_colors + 2
            count = count + 3
        result[i, -1] = decode_onehot(inp[i, -4:])
    return result

def analyze(inp, colors_dict):
    n_colors = len(colors_dict)
    if not isinstance(inp, np.ndarray):
        inp = inp.detach().numpy()
    result = np.zeros((inp.shape[0], 16), dtype=np.dtype('a16'))
    for i in range(len(inp)):
        idx = 0
        count = 0
        for j in range(5):
            result[i, count] = inp[i, idx] + 1
            result[i, count + 1] = inp[i, idx + 1]
            color_ohe = decode_onehot(inp[i,idx + 2: idx + 2 + n_colors])
            result[i, count + 2] = colors_dict[color_ohe]
            idx = idx + n_colors + 2
            count = count + 3
        # result[i, -1] = actions[decode_onehot(inp[i, -4:])]
    return result

def decode_onehot(a):
    return np.where(a == 1)[0][0]

def get_named_layers(net):
    conv2d_idx = 0
    convT2d_idx = 0
    linear_idx = 0
    batchnorm2d_idx = 0
    named_layers = {}
    for mod in net.modules():
        if isinstance(mod, torch.nn.Conv2d):
            layer_name = 'Conv2d{}_{}-{}'.format(
                conv2d_idx, mod.in_channels, mod.out_channels
            )
            named_layers[layer_name] = mod
            conv2d_idx += 1
        elif isinstance(mod, torch.nn.ConvTranspose2d):
            layer_name = 'ConvT2d{}_{}-{}'.format(
                conv2d_idx, mod.in_channels, mod.out_channels
            )
            named_layers[layer_name] = mod
            convT2d_idx += 1
        elif isinstance(mod, torch.nn.BatchNorm2d):
            layer_name = 'BatchNorm2D{}_{}'.format(
                batchnorm2d_idx, mod.num_features)
            named_layers[layer_name] = mod
            batchnorm2d_idx += 1
        elif isinstance(mod, torch.nn.Linear):
            layer_name = 'Linear{}_{}-{}'.format(
                linear_idx, mod.in_features, mod.out_features
            )
            named_layers[layer_name] = mod
            linear_idx += 1
    return named_layers

def get_attribute_sem():
    result = []
    for j in range(5):
        if j == 0:
            pre = "a."
        else:
            pre = actions[j-1][0] + "."
        result.append(pre + "x")
        result.append(pre + "y")
        for i in range(1):
            result.append("")
        result.append(pre + "color")
        for i in range(1):
            result.append("")

    for i in range(2):
        result.append("")
    result.append("action")
    for i in range(1):
        result.append("")

    return result

def plot_weight(w, plot_name, dir):
    w = np.absolute(w)
    m, n = w.shape
    x_label = get_attribute_sem()
    y_label = ["a.x", "a.y"]
    fig, ax = plt.subplots(figsize=(15,2))

    ax.matshow(w, cmap=plt.cm.Blues)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    # We want to show all ticks...
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(m))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_label)
    ax.set_yticklabels(y_label)

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(m):
    #     for j in range(n):
    #         text = ax.text(i, j, "{:.0f}".format(w[i,j]),
    #                        ha="center", va="center")

    ax.set_title("Weight values")
    plt.savefig(plot_name)


def plot_weight_sem(w, plot_name, x_indices, y_indices, x_label, y_label):
    m, n = len(x_indices), len(y_indices)
    W = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            W[i,j] = w[y_indices[i], x_indices[j]]

    fig, ax = plt.subplots(figsize=(8,9))

    ax.matshow(abs(W), cmap=plt.cm.Blues)
    ax.tick_params(axis="x", bottom=True, rotation = (45), top=False, labelbottom=True, labeltop=False, pad = 20)
    ax.tick_params(axis="y", left = True, rotation = (45), right = False, labelleft = True, labelright = False, pad = 20)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_label)))
    ax.set_yticks(np.arange(len(y_label)) - 0.5)
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_label, va="center")
    ax.set_yticklabels(y_label, va="center")

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(m):
        for j in range(n):
            if W[j,i] != 0:
                t = "{:.1f}".format(W[j,i])
            else:
                t = ""
            text = ax.text(i, j,t ,
                           ha="center", va="center")

    ax.set_title("Weight values")
    plt.savefig(plot_name)
