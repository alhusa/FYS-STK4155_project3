import time
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import metrics
import xlrd
from scipy import stats

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import callbacks




class mlp:
    """
    Class to store the neural network
    """
    def __init__(self):
        """
        function to initialize the network
        """
        #set if the network prints the progress
        self.verbose = True
        #toggle the keras nettwork
        self.run_keras = True
        #learning rate
        self.eta = 0.1
        #momentum factor
        self.momentum = 0.1
        #number of hidden nodes
        self.hidden1 = 50
        #number of hidden nodes
        self.hidden2 = 50
        #number of inputs
        self.ninput = 8
        #number of outputs
        self.noutput = 2
        #size of minibatch
        self.minibatch = 19
        #number of epochs between checking earlystopping
        self.early = 10
        #weights for hidden layer1
        #self.v = (2/np.sqrt(self.ninput + 1)) * np.random.random_sample((self.ninput + 1 ,self.hidden1)) -1/np.sqrt(self.ninput + 1)
        self.v = np.random.randn(self.ninput + 1 ,self.hidden1) * np.sqrt(2/(self.ninput + 1))
        #store previous deltas for momentum
        #self.v = np.zeros((self.ninput + 1 ,self.hidden))
        self.vprev = np.zeros((self.ninput + 1 ,self.hidden1))
        #weights for hidden layer2
        #self.u = (2/np.sqrt(self.hidden1 + 1)) * np.random.random_sample((self.hidden1 + 1 ,self.hidden2)) -1/np.sqrt(self.hidden1 + 1)
        self.u = np.random.randn(self.hidden1 + 1 ,self.hidden2) * np.sqrt(2/(self.hidden1 + 1))
        #store previous deltas for momentum
        #self.v = np.zeros((self.ninput + 1 ,self.hidden))
        self.uprev = np.zeros((self.hidden1 + 1 ,self.hidden2))
        #weights for output layer
        #self.w = (2/np.sqrt(self.hidden2 + 1))* np.random.random_sample((self.hidden2 + 1 ,self.noutput)) -1/np.sqrt(self.hidden2 + 1)
        self.w = np.random.randn(self.hidden2 + 1 ,self.noutput) * np.sqrt(2/(self.hidden2 + 1))
        #store previous deltas for momentum
        #self.w = np.zeros((self.hidden + 1 ,self.noutput))
        self.wprev = np.zeros((self.hidden2 + 1 ,self.noutput))


    def earlystopping(self, inputs, targets, valid, validtargets):
        """
        The earlystopping function runs the training function a number of epoch and evaluates the nn. The training
        is stopped if the network show no sign of improvement. A validation set is used to assess the model.
        Inputs: training data, training labels, validation data, validation labels.
        """

        #creats arrays to store MSE
        last_ac = np.zeros(10)
        timestart = 0
        overfit = 0
        for k in range(0,1000):

            #trains the MLP using all the test data
            for i in range(0,inputs.shape[0],self.minibatch):
                self.train(inputs[i:i+self.minibatch,:],targets[i:i+self.minibatch,:])


            #reorder the data so the program is not trained in the same way every epoch
            order = list(range(np.shape(inputs)[0]))
            np.random.shuffle(order)
            inputs = inputs[order,:]
            targets = targets[order,:]


            #checks the model with the validation set every 100 epochs
            if k%self.early == 0:
                timend = time.time() - timestart

                #get accuracy of the model using the validation set
                ac, conf =  mlp1.confusion(valid,validtargets)

                #stores the last 10 ac scores
                last_ac[1:] = last_ac[:9]
                last_ac[0] = ac
                diff = ac - np.mean(last_ac)

                if self.verbose:
                    #get accuracy of the model using the traing set
                    ac_train, conf_train =  mlp1.confusion(inputs,targets)
                    print("--------------------")
                    print("After %d epochs" %k)
                    print("Validation set accuracy: %.5f" %ac)
                    print("Training set accuracy: %.5f" %ac_train)
                    print("Diff: %f" %diff)


                # #cheks if no improvements are found and increments overfit variable
                if diff <= 1e-5: overfit += 1
                #overfit varibale is reset if an improvement is found
                else:
                    overfit = 0
                    best_v = self.v
                    best_u = self.u
                    best_w = self.w

                #stops if there is little change in accuracy or
                if overfit > 3 or ac == 100 :
                    self.v = best_v
                    self.u = best_u
                    self.w = best_w
                    break
        return k

    def train(self, inputs, targets):
        """
        Trains the network by running it forwards and then using backpropagation to adjust the weights.
        Inputs: training data, training labels.
        Outputs: Number of runs before earlystopping and array of accuracy scores.
        """


        #runs the program forward
        outputH1,outputH2, outputO = self.forward(inputs)

        #calculate output error
        errOm = (outputO.T - targets) * outputO.T * np.subtract(1, outputO.T)

        #calculate hidden layer2 error
        errHm2 =  (outputH2 * np.subtract(1, outputH2))  * (self.w[1:,:] @ errOm.T)

        #calculate hidden layer1 error
        errHm1 =  (outputH1 * np.subtract(1, outputH1))  * (self.u[1:,:] @ errHm2)

        #put the bias in the hidden layer output
        outputHmi2 = np.zeros((self.hidden2 + 1,self.minibatch))
        outputHmi2[0,:] = -1
        outputHmi2[1:,:] = outputH2

        #adjusting weight for the output
        deltaw = self.eta * outputHmi2 @ (errOm/self.minibatch)
        self.w = self.w -  deltaw + self.momentum * self.wprev
        self.wprev = deltaw

        #put the bias in the hidden layer output
        outputHmi1 = np.zeros((self.hidden1 + 1,self.minibatch))
        outputHmi1[0,:] = -1
        outputHmi1[1:,:] = outputH1

        #adjusting wheight for hidden layer
        deltau = self.eta * outputHmi1 @ (errHm2/self.minibatch).T
        self.u = self.u - deltau + self.momentum * self.uprev
        self.uprev = deltau

        #put the bias in the inputs
        inputsi = np.zeros((inputs.shape[1] + 1,self.minibatch))
        inputsi[0,:] = -1
        inputsi[1:,:] = inputs.T

        #adjusting wheight for hidden layer
        deltav = self.eta * inputsi @ (errHm1/self.minibatch).T

        self.v = self.v - deltav + self.momentum * self.vprev
        self.vprev = deltav

    #function to run the MPL forward
    def forward(self, inputs):
        """
        Runs the network forwards.
        Inputs: training data.
        Outputs: Hiddel layer outputs and network output.
        """

        #puts bias in the inputs
        inputss = np.zeros((inputs.shape[0],inputs.shape[1] + 1))
        inputss[:,0] = -1
        inputss[:,1:] = inputs

        #caculate the hidden layer output
        outputhm1 = self.v.T @ inputss.T

        outputhm1 = self.activate(outputhm1)

        #puts bias in the inputs
        outputhm1i = np.zeros((self.hidden1 + 1,inputs.shape[0] ))
        outputhm1i[0,:] = -1
        outputhm1i[1:,:] = outputhm1

        #caculate the hidden layer output
        outputhm2 = self.u.T @ outputhm1i

        outputhm2 = self.activate(outputhm2)

        #puts bias in the hiddel layer outputs
        outputhm2i = np.zeros((self.hidden2 + 1,inputs.shape[0]))
        outputhm2i[0,:] = -1
        outputhm2i[1:,:] = outputhm2

        #calculate the outputs of the MLP
        outputom = self.w.T @ outputhm2i
        outputom = self.activate(outputom)

        #returns the hidden layer output and the output output
        return outputhm1,outputhm2, outputom

    def confusion(self, inputs, targets):
        """
        Calculates the accuracy score and confusion matrix for the network.
        Inputs: Data and labels that are going to be assessed.
        Outputs: accuracy score and confusion matrix.
        """
        #create confuison matrix
        conf = np.zeros((self.noutput,self.noutput))

        #creates a value to store the number of correct classifications
        correct = 0


        #runs the moodel forwards
        outputHi1,outputH2, outputOu = self.forward(inputs)

        #finds the target result and the estimated resilts
        tarind = np.argmax(targets,axis=1)
        estind = np.argmax(outputOu,axis=0)

        #for loop runs through the test input
        for i in range(0,inputs.shape[0]):
            #puts increments the values in the confusion matrix
            #based on the results
            conf[tarind[i]][estind[i]] = conf[tarind[i]][estind[i]] + 1

            #if a value is placed in the diag then classification is correct
            if tarind[i] == estind[i]:
                correct = correct + 1

        #gets the percentage of correct classifications
        percor = correct/inputs.shape[0]
        return percor * 100, conf


    def activate(self, inputs):
        """
        Calculates the activation function for an output.
        Inputs: outputs of a layer in the network
        Outputs: the activation function result.
        """

        return 1 / (1 + ( np.exp( - inputs)))


#Get the pulsar data
data = np.genfromtxt('pulsar_stars.csv', delimiter=',',skip_header=1)

#Get the labels of the data
labels = data[:,8]

#Subtract the mean for each inout
data[:,:8] = data[:,:8] - np.mean(data[:,:8], axis=0, keepdims=True)

#Divide the data by the max to reduce the size of the inputs
#max_abs_data = np.std(data[:,:23],axis=0,keepdims=True)
max_abs_data = np.max(data[:,:8], axis=0, keepdims=True)
data = data[:,:8]/max_abs_data

#generate onehot vector
target = np.zeros((data.shape[0],2));
for x in range(0,2):
    indices = np.where(labels==x)
    target[indices,x] = 1

# Randomly order the data
order = list(range(data.shape[0]))
np.random.shuffle(order)
data = data[order,:]
target = target[order,:]


# Split data into k sets
foldsm = []
foldst = []
folds = 6
for i in range(0,17898,2983):
    foldsm.append(data[i:i+2983,:])
    foldst.append(target[i:i+2983,:])


#arrays to store data
test_ac = np.zeros(folds)
train_ac = np.zeros(folds)

k = np.zeros(folds)
ac_test_keras = np.zeros(folds)
ac_train_keras = np.zeros(folds)



for i in range(0,folds):

    # Test data is used to evaluate how good the completely trained network is.
    test = foldsm[i]
    test_targets = foldst[i]
    sum = 0

    valind = np.random.randint(folds-1)
    if valind >= i:
        valind = valind + 1

    # Validation checks how well the network is performing and when to stop
    valid = foldsm[valind]
    valid_targets = foldst[valind]

    sumind = 0
    for j in range(0,folds):
        if j != i and j != valind: sumind = sumind + foldsm[j].shape[0]

    #Training data to train the network
    train = np.zeros((sumind,data.shape[1]))
    train_targets = np.zeros((sumind,2))
    placedind = 0
    for j in range(0,folds):
        if j != i and j != valind:
            train[placedind:placedind+foldsm[j].shape[0],:] = foldsm[j]
            train_targets[placedind:placedind+foldsm[j].shape[0],:] = foldst[j]
            placedind = placedind + foldsm[j].shape[0]


    # initialize the network
    mlp1 = mlp()


    #run the mlp
    k[i]=mlp1.earlystopping(train, train_targets, valid, valid_targets)


    #array to store accuracy scores
    test_ac[i], mat = mlp1.confusion(test,test_targets)
    train_ac[i], train_mat = mlp1.confusion(train,train_targets)



    if mlp1.run_keras:
        if mlp1.verbose: verb = 1
        else: verb = 0
        #using kera for comparison
        model = Sequential()
        model.add(Dense(mlp1.hidden1, input_dim=mlp1.ninput, activation='relu'))
        model.add(Dense(mlp1.hidden2, input_dim=mlp1.hidden1, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        earlystop = callbacks.EarlyStopping(monitor='val_acc', min_delta=1e-7, patience=5, verbose=0, mode='auto')
        model.fit(train, train_targets, epochs=100, verbose=verb, batch_size=20, validation_data=(valid,valid_targets), callbacks=[earlystop])
        keras_pred_test = model.predict(test)
        keras_pred_train = model.predict(train)




        #define values to be one or the other class
        keras_pred_test[keras_pred_test < 0.5] = 0
        keras_pred_test[keras_pred_test >= 0.5] = 1

        keras_pred_train[keras_pred_train < 0.5] = 0
        keras_pred_train[keras_pred_train >= 0.5] = 1


        #Get the accuracy of the keras model
        ac_test_keras[i] = metrics.accuracy_score(test_targets,keras_pred_test) * 100
        ac_train_keras[i] = metrics.accuracy_score(train_targets,keras_pred_train) * 100



    #print results for each fold
    print("--------------------------------------------------------\n")
    print("Own MLP:")
    print("Train set accuracy: %.4f%% Test set accuracy: %.4f%% \n" %(train_ac[i], test_ac[i]))
    print("Confusion matrix:")
    print(mat)

    if mlp1.run_keras:
        print("keras MLP:")
        print("Train set accuracy: %.4f%% Test set accuracy: %.4f%% \n" %(ac_train_keras[i], ac_test_keras[i]))

    print("--------------------------------------------------------")

#print results
print("Created MLP:")
print("Average accuracy train data: %.2f%%. Min: %.2f%%. Max: %.2f%%. Train std: %.2f%%"
%(np.mean(train_ac), np.min(train_ac), np.max(train_ac), np.std(train_ac)))
print("Average accuracy test data: %.2f%%. Min: %.2f%%. Max: %.2f%%. Test std: %.2f%%"
%(np.mean(test_ac), np.min(test_ac), np.max(test_ac), np.std(test_ac)))



if mlp1.run_keras:
    print("keras MLP:")
    print("Average accuracy train data: %.2f%%. Min: %.2f%%. Max: %.2f%%. Train std: %.2f%%"
    %(np.mean(ac_train_keras), np.min(ac_train_keras), np.max(ac_train_keras), np.std(ac_train_keras)))
    print("Average accuracy test data: %.2f%%. Min: %.2f%%. Max: %.2f%%. Test std: %.2f%%"
    %(np.mean(ac_test_keras), np.min(ac_test_keras), np.max(ac_test_keras), np.std(ac_test_keras)))



#plot the accuracy scores
plt.plot(train_ac, 'b',label='Created MLP train')
plt.plot(test_ac,'--b',label='Created MLP test')

if mlp1.run_keras:
    plt.plot(ac_train_keras,'r',label='keras MLP train')
    plt.plot(ac_test_keras,'--r',label='keras MLP test')



if mlp1.run_keras:
    plt.title("Accuracy scores for test and training data for the created and the keras MLP during a 6-fold", fontsize = 16)
else:
    plt.title("Accuracy scores for test and training data for the created MLP during a 6-fold", fontsize = 16)
plt.legend(fontsize=16)
plt.xlabel('Fold number',fontsize=15)
plt.ylabel('Accuracy score [%]',fontsize=15)
plt.tick_params(labelsize=15)


plt.show()
