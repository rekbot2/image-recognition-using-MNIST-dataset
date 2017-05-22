import numpy as np
import time

#load MNIST data
import h5py
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0]  ) )

MNIST_data.close()

#Find P[Y=k]
def prob_y_func():
    klist = np.zeros(10)

    for i in y_train:
        klist[i] += 1

    klist = np.divide(klist,len(y_train))
    return klist

#Find mu for all k values
def mu_func():
    y_train_list = np.reshape(y_train,len(y_train)).tolist()
    mu = {}
    for i in range(10):
        index = [j for j, val in enumerate(y_train_list) if val == i]
        x_train_i = x_train[index,:]
        mu[i] = np.mean(x_train_i,axis=0)   
    return mu

#Find sigma for all k values
def sigma_func():
    lam = 0.1
    y_train_list = np.reshape(y_train,len(y_train)).tolist()
    sigma = {}
    for i in range(10):
        index = [j for j, val in enumerate(y_train_list) if val == i]
        x_train_i = x_train[index,:]
        sigma[i] = np.cov(x_train_i.T)
        sigma[i] = sigma[i] + lam*np.eye(784) #Regularize the array to account for noise
    return sigma

#Run functions to have values in variables
prob_y = prob_y_func()
mu = mu_func()
sigma = sigma_func()

#Put found values into log liklihood function
def log_likelihood_func(x,mu,sigma,p):
    log_likelihood = np.zeros(10)
    
    for i in range(10):
        log_det_sigma = np.linalg.slogdet(sigma[i])[1]
        inverse_sigma = np.linalg.inv(sigma[i])
        log_likelihood[i] = np.log(p[i])+(-0.5*log_det_sigma-0.5*np.dot(x-mu[i],np.dot(inverse_sigma,x-mu[i])))     
    return log_likelihood

#To get solutions use np.argmax() on the result of the log_liklihood for desired image.

#Accuracy function for the first Q values (Using entire data set in this case)
def accuracy_func(Q):
    inc = 0
    for i in range(Q):
        if np.argmax(log_likelihood_func(x_test[i],mu,sigma,prob_y)) == y_test[i]:
            inc += 1
    return(inc/Q)    
Q = 1 #change Q to check accuracy for larger sample sizes           
print('Overall accuracy = %f' % (accuracy_func(Q)*100) + '%')

#Plot images with predictions
def PlotImage(vector_in, time_to_wait):
    array0 = 255.0*np.reshape(vector_in, (28,28) )
    from matplotlib import pyplot as plt
    plt.ion()
    plt.show(block=False)
    plt.figure(figsize= (2,2) )
    plt.imshow(array0, cmap='Greys_r')
    plt.draw()
    plt.show()
    time.sleep(time_to_wait)
    plt.close('all')

#time (in seconds) between each image showing up on the screen
time_to_wait = .1
#number of images from the dataset that will be plotted
N = 5

#Plot the images
for i in range(N):
    PlotImage(x_test[i], time_to_wait)
    solution = (np.argmax(log_likelihood_func(x_test[i],mu,sigma,prob_y)),y_test[i])
    print("Prediction: %i || Actual Value: %i" % solution)