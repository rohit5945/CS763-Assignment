#!/usr/bin/env python
# coding: utf-8

# ### Import relevant packages

# In[392]:


import numpy as np
import cv2
#from utilities import *
import torchfile as tr
import matplotlib.pyplot as plt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load training and test data

# In[393]:


train_data = tr.load('../data.bin')
test_data = tr.load('../test.bin')
train_labels=tr.load('../labels.bin')


# In[394]:


print("Training data shape :{}".format(train_data.shape))
print("Testing data shape :{}".format(test_data.shape))
print("Training label data shape :{}".format(train_labels.shape))


# ### Split train data to train and val data

# In[395]:


from sklearn.model_selection import train_test_split
test_x,valid_x,test_y,valid_y=train_test_split(train_data,train_labels,test_size=0.2,random_state=0)


# ### Image plot for data Visualization

# In[396]:


index=14
img=train_data[index]


# In[397]:


plt.imshow(img,cmap='gray')


# ### Reshaping training and test data

# In[398]:


#Reshaping training data to (108*108,29160)
train_data=test_x.reshape(test_x.shape[0],-1).T
validation_data=valid_x.reshape(valid_x.shape[0],-1).T


# In[399]:


#Reshaping test data to shape (108*108,29160)
test_data=test_data.reshape(test_data.shape[0],-1).T


# In[400]:


#Reshaping train labels to (29160,1)
train_labels=test_y.reshape(test_y.shape[0],-1).T
validation_labels=valid_y.reshape(valid_y.shape[0],-1).T


# In[401]:


train_labels_acc=np.copy(train_labels)


# ### Convert Lables to one hot vectors

# In[402]:


train_labels=np.squeeze(np.eye(6)[train_labels]).T
#validation_labels=np.squeeze(np.eye(6)[validation_labels]).T


# In[403]:


print("Training data shape after Reshape:{}".format(train_data.shape))
print("Training label data shape after Reshape:{}".format(train_labels.shape))
print("Testing data shape after Reshape:{}".format(test_data.shape))


# ## Defining Network Architecture

# ### Weight Initialization

# In[404]:


def initialization(layer_dims):
    np.random.seed(3)
    no_of_layers = len(layer_dims)-1
    parameters ={}
    for l in range(1, no_of_layers + 1):
        parameters["W"+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters["b"+str(l)]=np.zeros(shape=(layer_dims[l],1))
    return parameters


# ### Utility Functions

# In[405]:


def relu_forward(Z):
    A=np.maximum(0,Z)
    return A,Z


# In[406]:


def relu_backward(prev_der,saved_data):
    this_layer_activation=saved_data  #saved_data=Z of this layer
    d_this_layer=np.array(prev_der, copy=True)
    #print("d of thsi layer===",d_this_layer)
    d_this_layer[this_layer_activation <=0 ]=0
    #print("d of thsi layer===",d_this_layer)
    return d_this_layer


# In[407]:


def softmax_forward(Z):
    exp=np.exp(Z)
    sum_exp=np.sum(exp,axis=0,keepdims=True)
    softmax=exp/sum_exp
    #Z -= np.max(Z)
    #softmax = (np.exp(Z).T / np.sum(np.exp(Z),axis=1)).T
    return softmax,Z


# ### Forward Propogation

# In[408]:


def linear_plus_activation(X,W,b,activation):
    Z = np.dot(W,X)+b
    
    if activation=="relu":
        A,activation_save_data=relu_forward(Z)
    else:
        A,activation_save_data=softmax_forward(Z)
    linear_save_data=(X,W,b)
    save_data=(linear_save_data,activation_save_data)
    return A,save_data


# In[409]:


def forward_prop(X,parameters):    
    A=X
    cache_data=[]
    #no of layers
    L=len(parameters)//2  #W and b so //2
    for l in range(1,L):
        A_prev=A
        A ,relu_save_data= linear_plus_activation(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
        cache_data.append(relu_save_data)
    A_final ,softmax_save_data= linear_plus_activation(A,parameters["W"+str(L)],parameters["b"+str(L)],"softmax")
    cache_data.append(softmax_save_data)
    return A_final,cache_data
        


# ### Backward propogation

# In[410]:


def backward_gradient_compute_linear(grad_from_top_layer,save_data,activation):
    linear_saved_data,activation_saved_data=save_data
    total_images=grad_from_top_layer.shape[1]
    if activation=="relu":
        grad_from_top=relu_backward(grad_from_top_layer,activation_saved_data)
    prev_activation,W,b=linear_saved_data
    #print("top grad==",grad_from_top)
    dW=np.dot(grad_from_top,prev_activation.T)/total_images
    db=np.sum(grad_from_top,axis=1,keepdims=True)/total_images
    d_pass_to_prev_layer=np.dot(W.T,grad_from_top)
    #print("dw==",dW)
    return d_pass_to_prev_layer,dW,db
    
    


# In[411]:


def backward_prop(output_from_softmax,Y,saved_data):
    gradients={}
    prev_derivate=output_from_softmax-Y
    
    number_of_layers=len(saved_data)
    total_images=Y.shape[1]
    final_layer_saved=saved_data[number_of_layers-1]
    linear,activation=final_layer_saved
    gradients['dW'+str(number_of_layers)]=np.dot(prev_derivate,linear[0].T)/total_images
    gradients['db'+str(number_of_layers)]=np.sum(prev_derivate,axis=1,keepdims=True)/total_images
    prev_derivate=np.dot(linear[1].T,prev_derivate)
    for l in reversed(range(number_of_layers-1)):
        current_layer_save_data=saved_data[l]
        prev_derivate,dW,db=backward_gradient_compute_linear(prev_derivate,current_layer_save_data,"relu")
        gradients['dW'+str(l+1)]=dW
        gradients['db'+str(l+1)]=db
        #print("gradient shapes of W{0}:{1} and b{2}:{3}".format(l,dW.shape,l,db.shape))
    #print("dW1===",gradients["dW1"])
    return gradients
        
        
    


# ### Compute Loss

# In[412]:


def cost(y_hat,y):
    #print("shape of y_hat {} and y {}".format(y_hat.shape,y.shape))
    total_images=y.shape[1]
    #cost=-np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))/total_images
    L_sum = np.sum(np.multiply(y, np.log(y_hat))+np.multiply((1-y), np.log(1-y_hat)))
    L = -(1./total_images) * L_sum

    L = np.squeeze(L) 
    return L


# In[413]:


# GRADED FUNCTION: compute_cost_with_regularization

def compute_cost_with_regularization(y_hat, y, parameters, lambd):
    total_images = y.shape[1]
    number_of_layers = len(parameters)//2
        
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = cost(y_hat, y) # This gives you the cross-entropy part of the cost
    
    ### START CODE HERE ### (approx. 1 line)
    L2_regularization_cost = lambd * (np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))/(2*total_images)
    ### END CODER HERE ###
    
    total_cost = cross_entropy_cost + L2_regularization_cost
    
    return total_cost


# ### Update parameters with gradient descent

# In[414]:


def update(parameters,gradients,learning_rate):
    number_of_layers=len(parameters)//2  # sice W and B in 1 params so /2
    #print("prev weight ",parameters["W1"])
    #print("prev grad ",gradients["dW1"])
    for l in range(number_of_layers):
        parameters["W"+str(l+1)]-=learning_rate*gradients["dW"+str(l+1)]
        parameters["b"+str(l+1)]-=learning_rate*gradients["db"+str(l+1)]
    #print("new weight ",parameters["W1"])
    return parameters


# ### Update parameters with gradient descent with momentum

# In[415]:


def update_with_momentum(parameters,gradients,learning_rate,beta,beta2):
    number_of_layers=len(parameters)//2  # sice W and B in 1 params so /2
    #print("prev weight ",parameters["W1"])
    #print("prev grad ",gradients["dW1"])
    # Initialize velocity
    v={}
    s={}
    epsilon=1e-8
    for l in range(number_of_layers):
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)

    for l in range(number_of_layers):
        # compute velocities
        v["dW" + str(l+1)] = beta*v["dW" + str(l+1)] + (1-beta)*gradients['dW' + str(l+1)]
        v["db" + str(l+1)] = beta*v["db" + str(l+1)] + (1-beta)*gradients['db' + str(l+1)]
        
        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        
        s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] + (1-beta2)*(gradients["dW" + str(l+1)]**2)
        s["db" + str(l+1)] = beta2*s["db" + str(l+1)] + (1-beta2)*(gradients["db" + str(l+1)]**2)
        
        # update parameters
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v["dW" + str(l+1)]                                                                    /(s["dW" + str(l+1)]**0.5+epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v["db" + str(l+1)]                                                                    /(s["db" + str(l+1)]**0.5+epsilon)
    #print("new weight ",parameters["W1"])
    return parameters


# ### Define complete Pipeline

# In[416]:


### CONSTANTS ###
layers_dims = [11664,32, 10, 6] #  3-layer model
#layers_dims = [11664,10,6]
learning_rate = 1e-3
num_iterations = 5000
L2_regularizer_lambd=0.8
momentum_beta=0.999
momentum_beta2=0.9


# In[417]:


costs = []  
def model(X,Y,layer_dims,learning_rate , num_iterations):
    
    parameters=initialization(layer_dims)
    #print("para   ",parameters)
    for i in range(0,num_iterations):
        #forward propogation
        y_hat,cache_data = forward_prop(X,parameters)
        #print("output===",y_hat)
        iter_cost=cost(y_hat,Y) #non-regularized cost
        #iter_cost = compute_cost_with_regularization(y_hat,Y,paramerters,L2_regularizer_lambd)
        # Print the cost every 100 training example
        if i % 1 == 0:
            print("Cost after iteration {}: {}".format(i+1, np.squeeze(iter_cost)))
            costs.append(iter_cost)
        gradients=backward_prop(y_hat,Y,cache_data)
        #print("total grads  ",len(gradients))
        parameters=update_with_momentum(parameters,gradients,learning_rate,momentum_beta,momentum_beta2)
    return parameters,costs


# ### Plot cost for Visualization

# In[418]:


def plot_cost(costs):
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundred)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


# ### Model Prediction

# In[419]:


def predict(X, y, parameters,acc_type):
    
    total_images = y.shape[1]
    p = np.zeros((1,total_images), dtype = np.int)
    
    # Forward propagation
    y_hat,cache_data = forward_prop(X,parameters)
    
    prediction=np.argmax(y_hat,axis=0)
    prediction=np.squeeze(prediction.reshape(y_hat.shape[1],1))
    actual_label=np.squeeze(y)
    accuracy = sum(prediction == actual_label)/(float(len(actual_label)))
    
    print(" {} Accuracy: {}".format(acc_type,accuracy))


# ### Train Model

# In[420]:


paramerters,costs = model(train_data,train_labels,layers_dims,learning_rate,num_iterations)
#plot cost
plot_cost(costs)


# ### Model Accuracy

# In[421]:


# Train accuracy
predict(train_data,train_labels_acc,paramerters,"Training")


# In[422]:


# Validation accuracy
predict(validation_data,validation_labels,paramerters,"Validation")


# In[ ]:




