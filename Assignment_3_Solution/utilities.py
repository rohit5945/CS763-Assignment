import numpy as np

def initialization(layer_dims):
    no_of_layers = len(layer_dims)
    parameters ={}
    for l in range(no_of_layers-1):
        parameters["W"+str(l+1)]=np.zeros(shape=(layer_dims[l+1],layer_dims[l]))
        parameters["b"+str(l+1)]=np.zeros(shape=(layer_dims[l+1],1))

        #print("Layer {0} parameters shape : \n Weights :{1} \n bias : {2}".format(l+1,parameters["W"+str(l+1)].shape,parameters["b"+str(l+1)].shape))
#initialization([2,3,4,6,7])
     return parameters

def forward_pass(parameters,activation,layer_dims):
    pass

def backward_pass(forward_params):
    pass

def relu_forward(Z):
    pass

def relu_backward(Z):
    pass

def softmax_forward(Z):
    pass

def softmax_backward(Z):
    pass


