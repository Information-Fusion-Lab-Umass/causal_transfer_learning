import numpy as np
import matplotlib.pyplot as plt
import torch
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
        for i in range(2):
            result.append("")
        result.append(pre + "color")
        for i in range(2):
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
    plt.savefig(dir + "{}_embed.png".format(plot_name))
