import numpy as np



def one_hot(class_data, classes=16):

    ret = np.zeros((len(class_data),classes))
    ret[np.arange(len(class_data)),class_data] = 1
    return ret


def split_train_test_set(image_data, class_data, split=0.1):

    classes = np.unique(class_data)
    class_ind = [[]]*len(classes)
    i=0
    for c in classes:
        #print("c: ", c)
        class_ind[i] = np.where(class_data == c)[0]
        i += 1

    print("class ind shape: ", np.array(class_ind).shape)
    for ind_i in class_ind:
        print("class ind i shape: ", ind_i.shape)
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for i in range(len(classes)):

        split_i = int(len(class_ind[i])*(1-split))
        print("split_i: ", split_i, ", len: ", len(class_ind[i]), " - ", len(class_ind[i])*(1-split))
        # if i == 0:
        #     train_x = image_data[:split_i]
        #     train_y = class_data[:split_i]
        #     test_x = image_data[split_i:]
        #     test_y = class_data[split_i:]
        # else:
        train_x += list(image_data[class_ind[i][:split_i]])
        train_y += list(class_data[class_ind[i][:split_i]])
        test_x += list(image_data[class_ind[i][split_i:]])
        test_y += list(class_data[class_ind[i][split_i:]])
    
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
            
            
        


