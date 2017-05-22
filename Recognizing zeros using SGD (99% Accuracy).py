import numpy as np

#load MNIST data
import h5py
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0]  ) )

MNIST_data.close()

#initialize variables
theta = np.zeros((2,784)) #two rows, one representing theta_0 and other representing theta_1
alpha = .01 #found using trial and error
N = int(4e4) #number of trials
correct = 0 #variable incremented for accuracy calculations


#Define g(x) function to simplify math
def g(x):
    return 1/(1+np.exp(-x))

#Use Stochastic Gradient Descent to create theta predictions
for i in range(N): #for theta_0
    choice = np.random.randint(len(y_train))
    x = x_train[choice]
    if y_train[choice] == 0:
        y = 1
    else:
        y = 0
    t = np.transpose(theta[0])
    theta[0] = theta[0] + (alpha*(y-g(np.matmul(t,x))))*x
    
for i in range(N): #for theta_1
    choice = np.random.randint(len(y_train))
    x = x_train[choice]
    if y_train[choice] != 0:
        y = 1
    else:
        y = 0
    t = np.transpose(theta[1])
    theta[1] = theta[1] + (alpha*(y-g(np.matmul(t,x))))*x

#Define to simplify math for prediction loop
def sum_denom(x):
    total = 0
    for k in range(2):
        exponent = np.matmul(theta[k].T,x)
        total += np.exp(exponent)
    return total

#Prediction loop
for i in range(len(y_test)):
    p = [[np.zeros(784)],[np.zeros(784)]] #prediction array
    x = x_test[i] #define x for iteration of loop    
    if y_test[i] != 0: #define y for iteration of loop
        y = 1
    else: y = 0
    for j in range(2): #loop to get make prediction for y being either 0 or 1
        t = np.transpose(theta[j])
        temp = np.matmul(t,x)
        p[j] = np.exp(temp)/(sum_denom(x))
    guess = np.argmax(p)
    if guess == y:
        correct += 1

#Calculate accuracy
print('Overall Accuracy: %f' % ((correct/len(y_test))*100))