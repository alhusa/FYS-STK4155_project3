
import numpy as np
from sklearn import metrics
from sklearn import linear_model
import sklearn
import random
import time as tm
import xlrd


from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def predict(x,beta):
    """
    Function that predicts labels based on the beta parameters that has been
    calculatted.
    Inputs: datapoints and beta parameters
    Output: predicted label
    """
    ypred = x @ beta
    ypred = 1 / (1 + np.exp(-ypred))
    if np.isnan(np.sum(ypred)):
        ypred_0 = np.zeros((ypred.shape))
        return ypred_0
    ypred[ypred < 0.5] = 0
    ypred[ypred >= 0.5] = 1
    return ypred


def beta_update(lr,x,y,beta):
    """
    Function to calculate the new beta parameters without regularization.
    Inputs: data, labels, beta parameters
    Output: updated beta
    """
    output = 1 / (1 + np.exp(-(x @ beta)))
    delta = x.T @ (output - y.reshape((len(y),1)))
    new_beta = beta - lr * delta/len(y)
    return new_beta

def beta_update_l2(lr,x,y,beta,lamb):
    """
    Function to calculate the new beta parameters with l1 regularization.
    Inputs: data, labels, beta parameters, lambda
    Output: updated beta
    """
    output = 1 / (1 + np.exp(-(x @ beta)))
    delta = x.T @ (output - y.reshape((len(y),1)))
    new_beta = beta - lr * (delta/len(y) + lamb * beta)
    return new_beta

def confusion(targets, pred_targets):
    """
    Calculates the onfusion matrix for the SVM.
    Inputs: Predicted and correct labels that are going to be assessed.
    Outputs: confusion matrix.
    """
    #create confuison matrix
    conf = np.zeros((2,2))


    #Remove redundant dimensions
    sq_targets = np.squeeze(targets).astype(int)
    sq_pred_targets = np.squeeze(pred_targets).astype(int)


    #for loop runs through the test input
    for i in range(0,targets.shape[0]):


        #Increments the values in the confusion matrix
        #based on the results
        conf[sq_targets[i]][sq_pred_targets[i]] = conf[sq_targets[i]][sq_pred_targets[i]] + 1


    return conf

#adjustable parameters
trainp = 0.5 #percentage of the data to be used for training
minibatch = 10 #minibatch size
lr = 0.2 #learning rate
lamb = [0.00001, 0.0001,0.001, 0.01, 1, 10,100] #value of lambda
#lamb = [0.00001, 0.0001,0.001, 0.01, 1,10,100,1000,10000] #a larger lambda
boot_runs = 100 #number of bootstrap runs

#Get data
data = np.genfromtxt('pulsar_stars.csv', delimiter=',',skip_header=1)
target = data[:,8]


#Subtract the mean for each inout
data[:,:8] = data[:,:8] - np.mean(data[:,:8], axis=0, keepdims=True)

#Divide the data by the max to reduce the size of the inputs
#max_abs_data = np.std(data[:,:23],axis=0,keepdims=True)
max_abs_data = np.max(data[:,:8], axis=0, keepdims=True)
data = data[:,:8]/max_abs_data


# Randomly order the data
order = list(range(data.shape[0]))
np.random.shuffle(order)
data = data[order,:]
target = target[order]

#find total samples and calculate the number of training data
sampn = len(target)
trainn = int(sampn*trainp)


#split the data into training and test sets
xt = np.zeros((trainn,data.shape[1]+1))
xt[:,0] = -1
xt[:,1:] = data[:trainn,:]
yt = target[:trainn]

xte = np.zeros((sampn-trainn,data.shape[1]+1))
xte[:,0] = -1
xte[:,1:] = data[trainn:,:]
yte = target[trainn:]



#array to store the ac scores
train_ac = np.zeros(len(lamb))
test_ac = np.zeros(len(lamb))
train_ac_l2 = np.zeros(len(lamb))
test_ac_l2 = np.zeros(len(lamb))
train_ac_sci = np.zeros(len(lamb))
test_ac_sci = np.zeros(len(lamb))




count = 0
for l in lamb:
    #array to store ac score in bootruns
    b_train_ac = np.zeros(boot_runs)
    b_test_ac = np.zeros(boot_runs)
    b_train_ac_l2 = np.zeros(boot_runs)
    b_test_ac_l2 = np.zeros(boot_runs)
    b_train_ac_sci = np.zeros(boot_runs)
    b_test_ac_sci = np.zeros(boot_runs)
    for j in range(boot_runs):

        #use scikit to take randoms samples with replacement
        x_boot, y_boot = sklearn.utils.resample(xt,yt)

        #initialize the beta parameters
        beta = (2/np.sqrt(data.shape[1]+1)) * np.random.random_sample((data.shape[1]+1,1)) -1/np.sqrt(data.shape[1]+1)
        beta_l2 = (2/np.sqrt(data.shape[1]+1)) * np.random.random_sample((data.shape[1]+1,1)) -1/np.sqrt(data.shape[1]+1)

        #variable to store the best score
        best_score = 0
        best_score_l2 = 0
        for k in range(0,50):

            for i in range(0,trainn,minibatch):

                #update beta parameters
                beta = beta_update(lr,x_boot[i:i+minibatch,:],y_boot[i:i+minibatch],beta)
                beta_l2 = beta_update_l2(lr,x_boot[i:i+minibatch,:],y_boot[i:i+minibatch],beta_l2,l)

            #checks the score of the model and stores the best parameters
            temp_pred = predict(x_boot,beta)
            temp_score = np.sum(y_boot.reshape((trainn,1)) == temp_pred) / len(y_boot)
            if temp_score > best_score:
                best_beta = beta
                best_score = temp_score

            temp_pred = predict(x_boot,beta_l2)
            temp_score = np.sum(y_boot.reshape((trainn,1)) == temp_pred) / len(y_boot)
            if temp_score > best_score_l2:
                best_beta_l2 = beta_l2
                best_score_l2 = temp_score


            #reshuffle the data so the model does not train the same way
            order = list(range(np.shape(x_boot)[0]))
            np.random.shuffle(order)
            x_boot = x_boot[order,:]
            y_boot = y_boot[order]

        #predicts the labels using beta
        ypred = predict(xte,best_beta)
        ypred_train = predict(x_boot,best_beta)


        #predicts the labels using beta_l2
        ypred_l2 = predict(xte,best_beta_l2)
        ypred_train_l2 = predict(x_boot,best_beta_l2)


        #fiting a scikit model
        scilearn = linear_model.LogisticRegression(penalty='l2',C=1/l).fit(x_boot, y_boot)


        #calualte the score of the predicted labels
        b_test_ac[j]= (np.sum(yte.reshape((sampn - trainn,1)) == ypred) / len(yte))*100
        b_train_ac[j]= (np.sum(y_boot.reshape((trainn,1)) == ypred_train) / len(y_boot))*100


        b_test_ac_l2[j]= (np.sum(yte.reshape((sampn - trainn,1)) == ypred_l2) / len(yte))*100
        b_train_ac_l2[j] = (np.sum(y_boot.reshape((trainn,1)) == ypred_train_l2) / len(y_boot))*100


        #calualte the score of the scikit models
        b_test_ac_sci[j] = scilearn.score(xte,yte) * 100
        b_train_ac_sci[j] = scilearn.score(x_boot,y_boot) * 100


    #Stor the mean of the accuracy of each run in the bootstrap
    train_ac[count] = np.mean(b_train_ac)
    test_ac[count] = np.mean(b_test_ac)
    train_ac_l2[count] = np.mean(b_train_ac_l2)
    test_ac_l2[count] = np.mean(b_test_ac_l2)
    train_ac_sci[count]= np.mean(b_train_ac_sci)
    test_ac_sci[count] = np.mean(b_test_ac_sci)
    
    #print results
    print("Created minibatch method:")
    print("Train score: %.4f" %train_ac[count])
    print("Test score: %.4f" %test_ac[count])
    print("Test: Max score: %f Min score: %f\n" %(np.max(b_test_ac),np.min(b_test_ac)))

    print("Created minibatch with L2 regularization lambda = %.5f:" %l)
    print("Train score: %.4f" %train_ac_l2[count])
    print("Test score: %.4f" %test_ac_l2[count])
    print("Test: Max score: %f Min score: %f\n" %(np.max(b_test_ac_l2),np.min(b_test_ac_l2)))

    print("Scikit learn method lambda = %.5f:" %l)
    print("Train score: %.4f" %train_ac_sci[count])
    print("Test score: %.4f" %test_ac_sci[count])
    print("Test: Max score: %f Min score: %f\n" %(np.max(b_test_ac_sci),np.min(b_test_ac_sci)))


    print("--------------\n")



    count += 1
plt.figure(1)
# Plot our performance on both the training and test data
plt.semilogx(lamb, train_ac, 'b',label='Created method train')
plt.semilogx(lamb, test_ac,'--b',label='Created method test')
plt.semilogx(lamb, train_ac_l2,'r',label='Created L2 train',linewidth=1)
plt.semilogx(lamb, test_ac_l2,'--r',label='Created L2 test',linewidth=1)
plt.semilogx(lamb, train_ac_sci, 'g',label='Scikit train')
plt.semilogx(lamb, test_ac_sci, '--g',label='Scikit test')



plt.title("Accuracy scores for test and training data with different values for lambda", fontsize = 16)
plt.legend(loc='lower left',fontsize=16)
plt.xlim([min(lamb), max(lamb)])
plt.xlabel('Lambda',fontsize=15)
plt.ylabel('Accuracy score [%]',fontsize=15)
plt.tick_params(labelsize=15)

plt.show()
