'''
@author:OUBAHA Rachid & EL MAKHROUBI Mohammed
@date:28/06/2020
@title:Autoencoders with python

Machine & Deep Learning | Master SIM 

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

#It is useful to get (local min , global min ) in gradient descent 
rate_learning=2 
neuron_hidden_layer = 48
loss_list=[]
epochs=20

def sigmoid(x):
   return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
   return x * (1.0 - x)

def show_loss(ep,l):
  ep = [i for i in range(0,ep)]

  plt.style.use ('ggplot')
  ax = plt.axes()
  ax.set(xlabel='epoch', ylabel='square loss' )
  ax.set_title('epoch/square loss')

  plt.plot(ep,l,color='blue',marker='o',markersize=11)

def plot_data(DigitsData):
    
  plt.gray()

  for g in range(10):
    print('Images of Number : ' , g)
    plt.matshow(DigitsData.images[g])
    print('------------------------------')
    plt.show()


class NeuralNetwork:
   def __init__(self, x):

      #inputs
       self.input      = x
      #weights between inputs and a hidden layer
       self.weights1   = np.random.rand(self.input.shape[1],neuron_hidden_layer) 
      #weights between a hidden layer and  outputs
       self.weights2   = np.random.rand(neuron_hidden_layer,self.input.shape[1])                 

       self.y          = x[0:1]

       self.output     = np.zeros(self.input.shape[1]) # y hat

       self.loss     = 0 # y hat

       
   def feedforward(self):

      #ENCODER: f(x)=h
       self.layer1 = sigmoid(np.dot(self.input, self.weights1))     
      #DECODER: g(h)=r  
       self.output = sigmoid(np.dot(self.layer1, self.weights2))

       # Loss function = input vector — output vector
       self.loss=(np.sum(self.y - self.output)/X.shape[1])**2
       loss_list.append(self.loss)
       
   def backprop(self):
       
       #DECODER: we use here g'() ==> derivative
       d_weights2 = np.dot(self.layer1.T, (rate_learning*(self.y - self.output) * sigmoid_derivative(self.output)))
       
       #ENCODER: we use here f'() ==> derivative
       d_weights1 = np.dot(self.input.T,
                           (np.dot(rate_learning*(self.y - self.output) * sigmoid_derivative(self.output),
                                   self.weights2.T) * sigmoid_derivative(self.layer1)))
      #UPDATE weights
       self.weights1 += d_weights1
       self.weights2 += d_weights2





#load digits data
DigitsData = load_digits()

# #X Data
X = np.array(DigitsData.data)/255
print('X shape is ' , X.shape)

#show data
plot_data(DigitsData)


# Create the network  
nn = NeuralNetwork(X)


#feedforward ----> & backprop <----
for i in range(epochs):
   nn.feedforward()
   nn.backprop()
   
#plot square loss
show_loss(epochs ,loss_list)

