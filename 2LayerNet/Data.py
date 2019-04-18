import numpy as np

def data_generate(d_size = 128, seed = 0):
    np.random.seed(seed)
    x_train = np.array([[np.random.uniform(-1,1),np.random.uniform(-1,1)]])
    if x_train[0][0]*x_train[0][0] < x_train[0][1]:
        label = np.array([[0]])
    else:
        label = np.array([[1]])
    labels = label
    for i in range(d_size-1):
        data = np.array([[np.random.uniform(-1,1),np.random.uniform(-1,1)]])
        x_train = np.concatenate((x_train,data))
        if data[0][0]*data[0][0] < data[0][1]:
            label = np.array([[0]])
        else:
            label = np.array([[1]])
        labels = np.concatenate((labels,label))
    return x_train,labels
