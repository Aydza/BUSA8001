
import sys
import numpy as np

class Perceptron:
    ''' Perceptron classifier
    
    Parameters: 
    ----------
    eta: learning Rate between 0.0 and 1.0
    n_iter: to go over the training set
    random_state: random number generator seed for random weight initialisation
    
    
    Attributes:
    -----------
    w_ : weights after fitting
    errors_ number of misclassifications in each epoch
    '''
    
    def __init__(self, eta=0.01, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        ''' Fit training data
    
        Paramters:
        ----------
        X: shape = [number_examples, number_features] - training data
        y: shape = [number_examples] - target values

        Returns:
        --------
        self: object
        '''

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # scale is standard deviation
        self.errors_ = []

        for _ in range(self.n_iter):
            print(f'---------------- Epoch:{_} -----------------')
            errors = 0
            for i, (xi, target) in enumerate(zip(X, y)): # for each row in zip(X,y)
                update = self.eta * (target - self.predict(xi)) # for each row xi
                errors += int(update != 0)
             #   print(f'observation: {i}, x: {xi}, w: {self.w_} y: {target}, prediction: {self.predict(xi)}, update factor: {update}, errors: {errors}')
                self.w_[1:] = self.w_[1:] + update*xi 
                self.w_[0] = self.w_[0] + update  
            self.errors_.append(errors)
        print(len(self.errors_))

        return self
    
    def net_input(self, X):   # X is a row of X
        ''' Calculate net input '''
        return np.dot(X, self.w_[1:]) + self.w_[0]  # np.dot function computes the vector dot (inner) product w'x
    
    def predict(self, X):  # X is a row of X
        ''' Return class label after unit step '''
        return np.where(self.net_input(X) >= 0, 1, 0)
    
    def print_weights(self):
        print(f'Optimized weights: {self.w_}')
