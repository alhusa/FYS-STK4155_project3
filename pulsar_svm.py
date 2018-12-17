

import numpy as np
import random as rand
from sklearn import svm
import time

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import cvxopt as cv
from cvxopt import matrix as ma



def my_SVM(x,target,data_test,target_test,kernel,C,gamma,tolerance,poly_deg,verbose):
    """
    Uses train data and test data to train and test the SVM. Also needs parameters for the SVM.
    Inputs: train data, train target, test data, test targets, kernel type, C-parameter, gamma-parameter,
    tolerance for suppoer vectors and degree of the polynomial kernel.
    Output: Predicted class of the test data
    """



    n = x.shape[0]
    #Finding K
    K = x @ x.T

    #Checks the kernel
    if kernel == 'poly':
        K = (1 + (gamma * K))**poly_deg

    elif kernel == 'rbf':

        #gets the x squared from the diagonal of K
        x_square = np.diag(K).reshape((n,1))

        #Sums up the squares
        x_mark = x_square @ np.ones((1,n))

        #Subtract the squares from K
        K = K - 0.5*(x_mark + x_mark.T)

        #Caclute let the final kernel using the exp
        K = np.exp(K*gamma)
    else:
        print("Kernel not recognised")
        return 0,0,0

    #Sers a minimum rank for the P matrix so the solver does not get
    #an error when different values are tested
    if poly_deg == 10: min_rank = 48
    elif poly_deg == 9: min_rank = 35
    elif poly_deg == 8: min_rank = 32
    elif poly_deg == 7: min_rank = 29
    elif poly_deg == 6: min_rank = 26
    elif poly_deg == 5: min_rank = 19
    elif poly_deg == 4: min_rank = 13
    elif poly_deg == 3: min_rank = 10
    else: min_rank = 0


    #Making matrices to fit the solver
    P = ma(target * target.T * K)
    q = ma(-np.ones((n,1)))
    G = ma(np.concatenate((np.eye(n),-np.eye(n))))
    h = ma(np.concatenate((C * np.ones((n,1)),np.zeros((n,1)))))
    A = ma(target.reshape(1,n),(1,n),'d')
    b = ma(0.)




    #Checks the rank of the P matrix and skips the solver if it is too small
    if np.linalg.matrix_rank(P) <= min_rank and kernel == 'poly' :
        print("Error P-matrix rank too low")
        print(np.linalg.matrix_rank(P))
        return 0,0,0

    #Use the cvxopt qp solver
    sol_min = cv.solvers.qp(P,q,G,h,A,b)


    #Get the lambdas from the solutions
    lambdas = np.array(sol_min['x'])

    #Get the vector within the margin
    s_vector = np.where(lambdas>tolerance)[0]

    #Skips the calculations if no support vectors are found
    if len(s_vector) == 0:
        return 0,0,0

    #Get the data and target of our support vectors
    s_x = x[s_vector,:]
    s_lambda = lambdas[s_vector]
    s_target = target[s_vector]
    if verbose: print("Number of support vectors %d" %len(s_x))
    #Get the bias
    b_best = 1/len(s_vector) * (np.sum(s_target - np.sum(s_lambda * s_target * K[s_vector.reshape((len(s_vector),1)),s_vector.reshape((1,len(s_vector)))],axis=1,keepdims=True)))


    #Gets prediction of the test data
    if kernel == 'poly': K_pred_test = (1. + gamma*(data_test @ s_x.T))**poly_deg
    elif kernel == 'rbf':

        #Get K matrix for the test data using the same technique as when
        #the first K matrix was created
        K_pred_test = data_test @ s_x.T
        y = np.sum(data_test**2, axis=1, keepdims=True) @ np.ones((1,K_pred_test.shape[1]))
        xx = x_square[s_vector] @ np.ones((1,K_pred_test.shape[0]))
        K_pred_test -= 0.5*(y + xx.T)
        K_pred_test = np.exp(K_pred_test * gamma)

    #Get the predicted classes using the K matrix
    class_pred_test = s_lambda.T * s_target.reshape(1,len(s_target)) @ K_pred_test.T + b_best

    #Gets the sign of the classification to determine the class
    class_pred_test = np.sign(class_pred_test)

    #Gets tge number of correct classifications
    score_test = float(np.sum(target_test.T== class_pred_test)) /len(target_test)



    #Gets prediction of the train data
    if kernel == 'poly': K_pred_train = (1. + gamma  *(x @ s_x.T))**poly_deg

    elif kernel == 'rbf':

        #Get K matrix for the test data using the same technique as when
        #the first K matrix was created
        K_pred_train = x @ s_x.T
        y = np.sum(x**2, axis=1, keepdims=True) @ np.ones((1,K_pred_train.shape[1]))
        xx = x_square[s_vector] @ np.ones((1,K_pred_train.shape[0]))
        K_pred_train -= 0.5*(y + xx.T)
        K_pred_train = np.exp(K_pred_train * gamma)

    #Get the predicted classes using the K matrix
    class_pred_train = s_lambda.T * s_target.reshape(1,len(s_target)) @ K_pred_train.T + b_best

    #Gets the sign of the classification to determine the class
    class_pred_train = np.sign(class_pred_train)

    #Gets tge number of correct classifications
    score_train = float(np.sum(target.T== class_pred_train)) /len(target)

    #Get the confusion matrix for the test data
    confuison_matrix = confusion(target_test,class_pred_test)


    return score_test*100, score_train*100, confuison_matrix

def confusion(targets, pred_targets):
    """
    Calculates the onfusion matrix for the SVM.
    Inputs: Predicted and correct labels that are going to be assessed.
    Outputs: confusion matrix.
    """
    #create confuison matrix
    conf = np.zeros((2,2))


    #Remove redundant dimensions
    sq_targets = np.squeeze(targets)
    sq_pred_targets = np.squeeze(pred_targets)


    #for loop runs through the test input
    for i in range(0,targets.shape[0]):

        if sq_targets[i] == -1: tarind = 0
        elif sq_targets[i] == 1: tarind = 1
        else: print("Error target")

        if sq_pred_targets[i] == -1: predind = 0
        elif sq_pred_targets[i] == 1: predind = 1
        else: print("Error perdiction")

        #Increments the values in the confusion matrix
        #based on the results
        conf[tarind][predind] = conf[tarind][predind] + 1


    return conf


#Adjustable parameters
kernel = 'rbf' #can be 'poly' of 'rbf'
train_n = 1 #number of traning fold (max: 5)
poly_deg = 10 #polynomial degree (only tested to 10)
tolerance = 1e-5 #tolerance for the choosing of support vectors
C = 1 #adjustable parameter that trades of margin size with errors
gamma = 9 #adjustable parameter for the SVM
skikit_SVM = False #Whether to run the scikit SVM or not
own_svm = True #Whether to run the created SVM or not
verbose = False #Adjust wether the functions are verbose or not

#Import the data
data = np.genfromtxt('pulsar_stars.csv', delimiter=',',skip_header=1)

#Sets the targets
target = data[:,8]


#Subtract the mean for each inout
data[:,:8] = data[:,:8] - np.mean(data[:,:8], axis=0, keepdims=True)

#Divide the data by the max to reduce the size of the inputs
#max_abs_data = np.std(data[:,:23],axis=0,keepdims=True)
max_abs_data = np.max(data[:,:8], axis=0, keepdims=True)
data = data[:,:8]/max_abs_data

#Sets the targets to -1 instead of 0
target[target==0] = -1
target = target.reshape(len(target),1)

# Randomly order the data
order = list(range(data.shape[0]))
np.random.shuffle(order)
data = data[order,:]
target = target[order,:]


# Split data into k sets
foldsm = []
foldst = []
folds = 6
fold_size = 2983
for i in range(0,17898,fold_size):
    foldsm.append(data[i:i+fold_size,:])
    foldst.append(target[i:i+fold_size,:])


#arrays to store data
test_ac = np.zeros(folds)
train_ac = np.zeros(folds)
ski_test_score = np.zeros(folds)
ski_train_score = np.zeros(folds)



for i in range(0,folds):


    # Test data is used to evaluate how good the completely trained network is.
    test = foldsm[i]
    test_targets = foldst[i]

    #Use random traing data
    train_ind = np.array(rand.sample(range(folds-1),train_n))
    train_ind[train_ind >= i] += 1

    #Training data to train the network
    train = np.zeros((fold_size*train_n,data.shape[1]))
    train_targets = np.zeros((fold_size*train_n,1))
    placedind = 0
    for j in range(0,train_n):

        train[placedind:placedind+foldsm[train_ind[j]].shape[0],:] = foldsm[train_ind[j]]
        train_targets[placedind:placedind+foldsm[train_ind[j]].shape[0],:] = foldst[train_ind[j]]
        placedind = placedind + foldsm[train_ind[j]].shape[0]

    if own_svm: test_ac[i], train_ac[i], mat = my_SVM(train,train_targets,test,test_targets,kernel,C,gamma,tolerance,poly_deg,verbose)

    if skikit_SVM:
        clf = svm.SVC(C=C,kernel=kernel,degree=poly_deg,gamma=gamma,tol=tolerance,verbose=verbose)
        clf.fit(train,train_targets)
        ski_test_score[i] = clf.score(test,test_targets) * 100
        ski_train_score[i] = clf.score(train,train_targets) * 100
        ski_mat = confusion(test_targets, clf.predict(test))

    #print results for each fold
    if verbose:
        print("--------------------------------------------------------\n")
        if own_svm:
            print("Own SVM:")
            print("C: %f  gamma: %f" %(C,gamma))
            print("Train set accuracy: %.4f%% Test set accuracy: %.4f%% \n" %(train_ac[i], test_ac[i]))
            print("Confusion matrix for test data:")
            print(mat)

        if skikit_SVM:
            print("Scikit SVM:")
            print("C: %f  gamma: %f" %(C,gamma))
            print("Train set accuracy: %.4f%% Test set accuracy: %.4f%% \n" %(ski_train_score[i], ski_test_score[i]))
            print("Confusion matrix for test data:")
            print(ski_mat)

#print results
if own_svm:
    print("Created SVM:")
    print("Average accuracy train data: %.2f%%. Min: %.2f%%. Max: %.2f%%. Train std: %.2f%%"
    %(np.mean(train_ac), np.min(train_ac), np.max(train_ac), np.std(train_ac)))
    print("Average accuracy test data: %.2f%%. Min: %.2f%%. Max: %.2f%%. Test std: %.2f%%"
    %(np.mean(test_ac), np.min(test_ac), np.max(test_ac), np.std(test_ac)))



if skikit_SVM:
    print("Scikit SVM:")
    print("Average accuracy train data: %.2f%%. Min: %.2f%%. Max: %.2f%%. Train std: %.2f%%"
    %(np.mean(ski_train_score), np.min(ski_train_score), np.max(ski_train_score), np.std(ski_train_score)))
    print("Average accuracy test data: %.2f%%. Min: %.2f%%. Max: %.2f%%. Test std: %.2f%%"
    %(np.mean(ski_test_score), np.min(ski_test_score), np.max(ski_test_score), np.std(ski_test_score)))




#plot the accuracy scores
if own_svm:
    plt.plot(train_ac, 'b',label='Created SVM train')
    plt.plot(test_ac,'--b',label='Created SVM test')

if skikit_SVM:
    plt.plot(ski_train_score,'r',label='Scikit SVM train')
    plt.plot(ski_test_score,'--r',label='Scikit SVM test')



if skikit_SVM and own_svm:
    plt.title("Accuracy scores for test and training data for the created and the Scikit SVM during a 6-fold", fontsize = 16)
elif own_svm:
    plt.title("Accuracy scores for test and training data for the created SVM during a 6-fold", fontsize = 16)
else:
    plt.title("Accuracy scores for test and training data for the Scikit SVM during a 6-fold", fontsize = 16)
plt.legend(fontsize=16)
plt.xlabel('Fold number',fontsize=15)
plt.ylabel('Accuracy score [%]',fontsize=15)
plt.tick_params(labelsize=15)


plt.show()
