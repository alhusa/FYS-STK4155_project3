# FYS-STK4155_project3

## Running the programs
All the programs can be run as normal in the command line. Some of the programs have parameters that can be adjusted as decribed bellow.


#### Logistic Regression
The parameters that can be adjusted can be found bellow the functions. You can adjust:
- Percentage of data to be used for training
- Minibatch size
- Learning rate
- Lambda values
- Number of runs in the bootstrap

#### MLP
The parameters that can be adjusted in the MLP can be in initializaion of the MLP class. You can adjust:
- Verbose (to print results during training or not)
- To run the keras network or not
- Learning rate
- Momentum
- Number of hidden nodes in hidden layer 1
- Number of hidden nodes in hidden layer 2
- Minibatch size
- Number of epochs between checking early stopping

#### SVM
The parameters that can be adjusted can be found bellow the functions. You can adjust:
- Kernel chooses the type of kernel. It can be 'poly' of 'rbf'
- Number of folds to be used for training (max: 5)
- The degree of kernal polynomial (tested up to 10)
- Tolerance for choosing the support vectors
- The C-parameter that trades of margin size and number of errors
- The gamma-parameter for the SVM
- Wether to run the scikit SVM or not
- Wether to run the created SVM or not
- Wether the functions are verbose or not


### Data Set
The data set is obtained from Kaggle, where it was posted by Pavan Raj.
URL: https://www.kaggle.com/pavanraj159/predicting-pulsar-star-in-the-universe/notebook?scriptVersionId=4487650

### Results
The results contains outputs of runs with the parameters that we found to work the best. The files shows outputs from the methods created for these projects and compares them to methods from the Scikit/Keras packages. The only exeption is the SVM runs with 5 folds of training data. These results only contain output from the Scikit SVM. It is therefore 4 output files for the SVM with different numbers of fold for training and with a polynomial and RBF kernel. 
