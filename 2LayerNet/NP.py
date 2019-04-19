import numpy as np
from Data import data_generate

def sigmoid(x):
    return 1/(1+np.exp(-x))

# two layer Network
class TwoLayerNet():
    def __init__(self,seed = 0):
        np.random.seed(seed)
        self.params = {}
        self.params['w1'] = np.random.randn(2, 2)# * 1/np.sqrt(2)# input - 2 , hidden 2
        self.params['b1'] = np.zeros((2,1))
        self.params['w2'] = np.random.randn(1,2)# * 1/np.sqrt(2) # hidden2 , output - 1
        self.params['b2'] = np.zeros((1,1))

    def print_param(self):
        for param in self.params.items():
            print(param)

    def propagation(self, x):
        z1 = np.dot(self.params["w1"],x) + self.params["b1"]
        a1 = sigmoid(z1)
        z2 = np.dot(self.params["w2"],a1) + self.params["b2"]
        a2 = sigmoid(z2)
        #print(a2)
        return a1,a2

    def cost(self,a2,y):
        m = y.shape[1]

        cross = np.multiply(np.log(a2), y) + np.multiply(np.log(1-a2) , (1-y))
        cost = - np.sum(cross) / m
        cost = np.squeeze(cost)
        return cost

    def back_propagation(self, a1, a2, x, y):
        m = y.shape[1]
        dz2 = a2 - y
        dw2 = np.dot(dz2, a1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        d_gz = a1 * (1 - a1)
        dz1 = np.dot(self.params["w2"].T, dz2) * d_gz
        dw1 = np.dot(dz1, x.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        grads = {"w1": dw1,
                 "w2": dw2,
                 "b2": db2,
                 "b1": db1}
        return grads

    def update_parameters(self, grads, lr=0.1):
        for k in self.params.keys():
            self.params[k] -= lr * grads[k]

    def training(self, x, y, num_iter,lr=0.1):
        for i in range(num_iter):
            # propagation
            a1, a2 = self.propagation(x)

            # back_propagation
            grads = self.back_propagation(a1, a2, x, y)

            self.update_parameters(grads, lr=lr)
            #if (i % 1000 == 0):
            #    print(i,self.cost(a2,y) ,self.predict(x,y))

    def predict(self, x, y):
        a1, a2 = self.propagation(x)
        predictions = abs(a2 - y) < 0.5
        # print(predictions)

        return np.mean(predictions)

#learning rate list
lrs = [0.2 * i for i in range(1,2)]
print("================Numpy====================")
for lr in lrs:
    print("======== learning rate : ",lr,"===========")
    model = TwoLayerNet(seed = 0)
    sum = 0
    for i in range(10):
        x_train, y_train = data_generate(seed =i)
        model.training(x_train.T,y_train.T,num_iter=5000,lr =lr)
        x_valid, y_valid = data_generate(seed = 11)
        acc = model.predict(x_valid.T,y_valid.T)
        #acc = model.predict(x_train.T, y_train.T)
        sum += acc
        print(i ,"iteration, accuracy : ",acc)
    print("====","lr : ",lr,",mean accuracy : ", sum/10)
    model.print_param()

