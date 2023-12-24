import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from pandas import DataFrame
import matplotlib.pyplot as plt

class Neural_Network:
    def __init__(self, hidden_layer_sizes):
        #aka numbers of features
        self.hidden_layer_sizes = np.asarray(hidden_layer_sizes)
    def fit(self, input_data, labeled, iter_, learning_rate, visualize = None):
        vis_ip = input_data.copy()
        learning_rate = learning_rate * 1/len(input_data)
        #labeled data -> matrix form
        self.expected_out = np.zeros((len(input_data), self.hidden_layer_sizes[-1]))
        for i in range(len(labeled)):
            self.expected_out[i][int(labeled[i])] = 1
        output_data = self.__forward_propagation(input_data)
        self.init_cost = self.__cost_function(output_data) #update cost
        #Optimizing
        self.opt = output_data.copy()
        #Visualize Init
        colors = {0:'ro', 1:'bo', 2:'go', 3: 'yo'}
        x_plot, y_plot = np.transpose(vis_ip)
        fig, ax = plt.subplots()
        plt.ion()
        self.cap = iter_/20
        for i in range(iter_):
            self.__apply_gradient(learning_rate) #gradient + back prop then update weights and biases
            self.opt = self.__re_feed(input_data) #update neurons
            cost = self.__cost_function(self.opt) #update cost
            print('Cost:', (sum(sum(cost))/2).round(2), '\tIter:', i) #print cost
            if visualize == True and i > self.cap or i == iter_-1:
                self.cap += iter_/20
                r_pred = self.predict(vis_ip)
                f_best = []
                for r in r_pred:
                    f_best.append(np.argmax(r))
                for i in range(len(f_best)):
                    ax.plot(x_plot[i], y_plot[i], colors[f_best[i]])
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.show()
        return self.opt
    def __forward_propagation(self, input_data):
        self.saved_weights = []
        self.saved_bias = []
        self.saved_raw_layer = []
        for feature in self.hidden_layer_sizes:
            #Init random weights and biases
            raw_in = input_data
            bias = self.__get_bias(input_data)
            input_data = np.c_[input_data, bias]
            weights = self.__get_weights(input_data, feature)
            #Perform matrix mul
            raw_out = input_data = input_data @ weights
            input_data = self.__activation_function(input_data)
            #Save weights, biases and raw layer (haven't pass thr activation and no biases)
            self.saved_bias.append(bias)
            self.saved_weights.append(weights)
            self.saved_raw_layer.append(raw_in)
        self.saved_raw_layer.append(raw_out)
        output_data = input_data
        return output_data
    def predict(self, input_data):
        for i in range(len(self.hidden_layer_sizes)):
            bias = self.saved_bias[i]
            weights = self.saved_weights[i]
            if bias.shape[0] > input_data.shape[0]:
                bias = np.resize(bias,(input_data.shape[0], bias.shape[1]))
            input_data = np.c_[input_data, bias]
            input_data = input_data @ weights
            input_data = self.__activation_function(input_data)
        output_data = input_data
        return output_data
    def __get_bias(self, layer):
        num_layer_rec = layer.shape[0]
        bias = np.zeros((num_layer_rec, 1))
        return bias
    def __get_weights(self, layer, num_nlayer_ft):
        num_layer_ft = layer.shape[1]
        weights = np.random.uniform(-0.25, 0.25, (num_layer_ft, num_nlayer_ft))
        return weights
    def __sigmoid(self, layer):
        return 1.0/(1.0+np.exp(-layer))
    def __activation_function(self, layer):
        ReLU = np.zeros(layer.shape, dtype=float)
        ReLU[layer>0] = layer[layer>0]
        return ReLU
    def __cost_function(self, output_data):
        #Calculate error and square error
        #Formula 1/2 sum squared error
        error = output_data - self.expected_out
        SE = np.square(error)
        #MSE
        cost = SE * 0.5
        return cost
    #Derivative
    #respect to Z (raw layer)
    def __activation_derivative_z(self, layer):
        dReLU = layer
        dReLU[dReLU>=0] = 1
        dReLU[dReLU<0] = 0
        return dReLU
    def __cost_derivative_z_L(self): #BP1 #line 30
        layer = self.saved_raw_layer[-1]
        ReLU = self.__activation_function(layer)
        dReLU = self.__activation_derivative_z(layer) #Maybe this is the fuking thing that cause the problems. Check latter
        del_L = (ReLU - self.expected_out) * dReLU
        self.del_lp1 = del_L
        return del_L
    def __cost_derivative_z_l(self, index_layer):#BP2
        del_l = ( 
                 (self.del_lp1 @ np.transpose(np.delete(self.saved_weights[index_layer],(-1),axis=0))) #delete biases's weights
                * self.__activation_derivative_z(self.saved_raw_layer[index_layer])
                )
        #update delta l+1
        self.del_lp1 = del_l
        return del_l
    #gradient descent
    def __back_propagation(self):
        network_del_l = []
        network_del_l.append(self.__cost_derivative_z_L())
        for i in reversed(range(1, len(self.saved_raw_layer)-1)):
            network_del_l.insert(0, self.__cost_derivative_z_l(i))
        return network_del_l
    def __gradient_descent(self):
        network_del_l = self.__back_propagation()
        grad_w = []
        grad_b = []
        for i in range(len(network_del_l)):
            if i == 0: aT = np.transpose(self.saved_raw_layer[i])
            else: aT = np.transpose(self.__activation_function(self.saved_raw_layer[i]))
            grad_w.append(aT@network_del_l[i])
            grad_b.append(np.mean(network_del_l[i], axis=1).reshape(-1,1))
        return grad_w, grad_b
    def __apply_gradient(self, learning_rate):
        grad_w, grad_b = self.__gradient_descent()
        for i in range(len(grad_w)):
            self.saved_weights[i][:-1,:] -= learning_rate * grad_w[i]
            self.saved_bias[i] -= learning_rate * grad_b[i]
    def __re_feed(self, input_data):
        #re-feed to update hidden and output layer
        updated_layer = []
        for i in range(len(self.hidden_layer_sizes)):
            input_data = np.c_[input_data, self.saved_bias[i]]
            input_data = input_data @ self.saved_weights[i]
            input_data = self.__activation_function(input_data)
            updated_layer.append(input_data) #append update
        self.saved_raw_layer[1:] = updated_layer #update data, exclude input layer
        return self.saved_raw_layer[-1] #return output layer

#Make data
X, y = make_blobs(n_samples=2000,centers=3, n_features=2)
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
data = df.to_numpy()
#Split
split = 0.5
TRAIN_SIZE = int(len(data)*split)
TEST_SIZE = len(data) - TRAIN_SIZE
np.random.shuffle(data)
training, test = data[:TRAIN_SIZE,:], data[TEST_SIZE:,:]
X, X_test = training[:,:2], test[:,:2]
y, y_test = training[:,2:].reshape(-1), test[:,2:].reshape(-1)
#Init model
nw = Neural_Network([10,5,5, 5])
nw.fit(X, y, 40000, 0.07, True)