import numpy as np

def get_neighboring_objects(X, action, reward, switch_count):
    Y = []
    idx = 1
    for i in range(X.shape[0]):
        color = X[i,0].decode("utf-8")
        if color not in ["black", "white", "red", "green"]:
            features = []
            features.append(X[i,idx:])
            for j in range(X.shape[0]):
                if i != j:
                    x_diff = int(X[i,2]) - int(X[j,2])
                    y_diff = int(X[i,3]) - int(X[j,3])
                    if y_diff == -1 and x_diff == 0:
                        r = X[j,idx+1:]

                    if y_diff == 1 and x_diff == 0:
                        l = X[j,idx+1:]

                    if x_diff == -1 and y_diff == 0:
                        d = X[j,idx+1:]

                    if x_diff == 1 and y_diff == 0:
                        u = X[j,idx+1:]

            try:
                features.append(u)
                features.append(d)
                features.append(l)
                features.append(r)
                features.append([action])
                features.append([reward])
                features.append([switch_count])
            except Exception as e:
                 print(e)
                 print(i, X[i], X)

            flat_list = []
            for sublist in features:
                for item in sublist:
                    flat_list.append(item)
            Y.append(flat_list)
    Y = np.asarray(Y).astype("float32")
    return Y

def get_oo_repr(t, objects, action, reward, n_colors, n_actions):
    A = []
    for o in objects:
        if o.name == "switch":
            switch_count = len(o.positions)
        for p1, p2 in o.positions:
                A.append([o.colorname, p1, p2])

    # print(" Time {} OO Objects {}".format(t, objects))
    A = np.asarray(A)
    colors = A[:,0]
    uniq_colors, colors_int = np.unique(colors, return_inverse=True)
    colors_dict = {}
    for i in range(len(colors)):
        colors_dict[colors_int[i]] = colors[i]
    X = np.zeros((A.shape[0], 5), dtype=np.dtype('a16'))

    X[:, 0] = colors
    X[:, 1] = np.ones(A.shape[0])* t
    X[:, 2] = A[:, 1]
    X[:, 3] = A[:, 2]
    X[:, 4] = colors_int
    # X[:, 4: 4+n_colors] = one_hot(colors_int, n_colors)
    # action_ohe = one_hot(np.ones(1, dtype = "int")* int(action), n_actions)
    nbrs = get_neighboring_objects(X, action, reward, switch_count)
    return nbrs, colors_dict

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
