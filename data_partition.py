import numpy as np






def iid_data_partitioning(x_data, y_data=None, N=1):

    data_size = len(x_data)
    indices = np.random.permutation(data_size)
    x_data = x_data[indices]

    if y_data is not None:
        y_data = y_data[indices]

    if type(N) is int:
        fraction_splits = np.int32(np.linspace(0, data_size, N + 1))
        nr = N
    elif type(N) == np.ndarray or type(N) == list:
        parts = np.array(N) * data_size / np.sum(N)
        fraction_splits = np.int32([np.sum(parts[:i]) for i in range(len(parts) + 1)])
        nr = len(parts)

    x_list = []
    y_list = []
    for i in range(nr):
        x_list.append(x_data[fraction_splits[i]:fraction_splits[i + 1]])

        if y_data is not None:
            y_list.append(y_data[fraction_splits[i]:fraction_splits[i + 1]])
            return x_list, y_list
        else:
            return x_list


def non_iid_classification_data_partitioning(x_data, y_data, N=1, M=2):

    data_size = len(x_data)

    if len(np.array(y_data).shape) == 1:
        classes = np.unique(y_data)
        sorted_ind = np.array([np.where(y_data == c)[0] for c in range(len(classes))])

    else:
        #y_data is one hot type
        classes = np.arange(np.array(y_data).shape[-1])
        l = np.eye(10)
        sorted_ind = np.array([np.where([list(y_) == list(l[i]) for y_ in y_data])[0] for i in range(len(classes))])

    part_len = N * M // len(classes)
    extra_parts = N * M % len(classes)
    parts = np.array([part_len for i in range(len(classes))])
    parts[:extra_parts] += 1
    c = np.array([np.int32(np.linspace(0, len(sorted_ind[i]), parts[i] + 1)) for i in range(len(classes))])
    new_sorted_ind = np.array(
        [[sorted_ind[i][c[i][j]:c[i][j + 1]] for j in range(parts[i])] for i in range(len(classes))])

    x_list = list(np.zeros(([M, 0] + list(x_data.shape[1:]))))
    y_list = list(np.zeros(([M, 0] + list(y_data.shape[1:]))))

    available_members = np.arange(M)
    for i in range(len(classes)):
        if len(available_members) >= len(new_sorted_ind[i]):
            draw_members, available_members = np.split(np.random.permutation(available_members),
                                                       [len(new_sorted_ind[i])])

        else:
            draw_members = available_members
            draw_ = np.random.permutation(np.array(list(set(np.arange(M)) - set(draw_members))))[:(len(new_sorted_ind[i]) - len(available_members))]
            draw_members = np.concatenate((draw_members,draw_))
            available_members = np.array(list(set(np.arange(M)) - set(draw_)),dtype=np.int32)

        k = 0
        for j in draw_members:
            y_list[j] = np.concatenate((y_list[j], y_data[new_sorted_ind[i][k]]))
            x_list[j] = np.concatenate((x_list[j], x_data[new_sorted_ind[i][k]]))
            k += 1

    return np.array(x_list), np.array(y_list)







