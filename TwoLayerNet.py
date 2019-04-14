import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))


print(sigmoid(0))
d_size = 128
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

print(x_train.shape)



# two layer Network
class TwoLayerNet():
    def __init__(self):
        self.params = {}
        self.params['w1'] = np.random.randn(2, 2)# input - 2 , hidden 2
        self.params['b1'] = np.zeros((2,1))
        self.params['w2'] = np.random.randn(1,2) # hidden2 , output - 1
        self.params['b2'] = np.zeros((1,1))

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

        dz2 = a2 - y
        dw2 = np.dot(dz2, a1.T) / d_size
        db2 = np.sum(dz2, axis=1, keepdims=True) / d_size
        d_gz = 1 - np.power(a1, 2)
        dz1 = np.dot(self.params["w2"].T, dz2) * d_gz
        dw1 = np.dot(dz1, x.T) / d_size
        db1 = np.sum(dz1, axis=1, keepdims=True) / d_size

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
            if (i % 100 == 0):
                print(i,self.cost(a2,y) ,self.predict(x))

    def predict(self, x):
        a1, a2 = self.propagation(x)
        predictions = a2 > 0.5

        return np.mean(predictions)

model = TwoLayerNet()
model.training(x_train.T,labels.T,1000,0.005)