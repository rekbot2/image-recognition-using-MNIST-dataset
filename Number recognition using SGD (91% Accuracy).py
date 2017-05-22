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
theta = np.zeros((10,784)) #10 rows representing each possible y value
alpha = .01 #found using trial and error
N = int(1e5) #number of trials
correct = 0 #variable incremented for accuracy calculations


#Define g(x) function to simplify math
def g(x):
    return 1/(1+np.exp(-x))

#define SGD function to create theta predictions
def SGD(q):
    for i in range(N):
        choice = np.random.randint(len(y_train))
        x = x_train[choice]
        if y_train[choice] == q:
            y = 1
        else:
            y = 0
        t = np.transpose(theta[q])
        theta[q] = theta[q] + (alpha*(y-g(np.matmul(t,x))))*x
    return theta[q]

#theta defining loop
for i in range(len(theta)):
    theta[i] = SGD(i)

#Define to simplify math for prediction loop
def sum_denom(x):
    total = 0
    for k in range(len(theta)):
        t = np.transpose(theta[k])
        exponent = np.matmul(t,x)
        total += np.exp(exponent)
    return total

#Prediction loop
for i in range(len(y_test)):
    p = [[np.zeros(784)]]*10 #prediction array
    x = x_test[i] #define x for iteration of loop    
    y = y_test[i] #define y for iteration of loop
    for j in range(len(theta)): #loop to get make prediction for y being either 0 or 1
        t = np.transpose(theta[j])
        temp = np.matmul(t,x)
        p[j] = np.exp(temp)/(sum_denom(x))
    guess = np.argmax(p)
    if guess == y:
        correct += 1

#Calculate accuracy
print('Overall Accuracy: %f' % ((correct/len(y_test))*100))