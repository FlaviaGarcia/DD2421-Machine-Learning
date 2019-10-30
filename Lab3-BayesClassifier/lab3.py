#!/usr/bin/python
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

# ## Jupyter notebooks
# 
# In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.
# 
# If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.
# 
# And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.

# ## Import the libraries
# 
# In Jupyter, select the cell below and press `ctrl + enter` to import the needed libraries.
# Check out `labfuns.py` if you are interested in the details.

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random


# ## Bayes classifier functions to implement
# 
# The lab descriptions state what each function should do.

"""
# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    for class_ in classes: 
        idx = np.where(labels == class_)[0]
        Nclass = len(idx)
        prior[class_] = Nclass/Npts
   
    return prior

"""

def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
        
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    for class_ in classes: 
        idx = np.where(labels == class_)[0]
        W_class = W[idx,:]
        prior[class_] = np.sum(W_class)/sum(W)
   
    return prior

"""
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts, Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses, Ndims, Ndims))

    for class_ in classes:
        idx = np.where(labels == class_)[0]
        X_class = X[idx, :]
        N_class = len(X_class)
        mu[class_] = np.sum(X_class, axis=0)/N_class
        for idx_dim in range(Ndims):
            # Just the elements of the diagonal will have a value != 0
            sigma[class_, idx_dim, idx_dim] = sum(np.power(X_class[:,idx_dim] - mu[class_,idx_dim], 2))/N_class
             
            
    return mu, sigma

"""



# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
#          W - N x 1 matrix of weights
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts, Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses, Ndims, Ndims))

    for class_ in classes:
        idx = np.where(labels == class_)[0]
        X_class = X[idx, :]
        W_class = W[idx,:]
        N_class = X_class.shape[0]
        mu[class_] = np.sum(W_class * X_class)/np.sum(W_class)
        for idx_dim in range(Ndims):
            sigma[class_, idx_dim, idx_dim] = 1/np.sum(W_class) * np.sum(W_class * np.power(X_class[:, idx_dim] - mu[class_,idx_dim], 2).reshape(N_class,1))
            
    return mu, sigma




# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses, Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    for class_idx in range(Nclasses):
        determ_sigma_class = np.linalg.det(sigma[class_idx])
        first_term_disc_eq = -1/2 * np.log(determ_sigma_class)
        third_term_disc_eq = np.log(prior[class_idx])

        for X_idx in range(Npts):
            second_term_disc_eq =   - 1/2 * \
                                    np.dot(
                                            np.dot(
                                                    X[X_idx] - mu[class_idx], 
                                                    np.linalg.inv(sigma[class_idx])
                                                    ), 
                                            np.transpose(X[X_idx] - mu[class_idx])
                                            )
            logProb_x_class =  first_term_disc_eq + second_term_disc_eq + third_term_disc_eq
            logProb[class_idx, X_idx] = logProb_x_class
    
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h



# The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:


# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# ## Test the Maximum Likelihood estimates
# 
# Call `genBlobs` and `plotGaussian` to verify your estimates.


X, labels = genBlobs(centers=5)
N = X.shape[0]

W = np.ones((N,1))/float(N)

#mu_prev, sigma_prev = mlParams_prev(X, labels)
# mu, sigma = mlParams(X,labels, W)

#plotGaussian(X,labels,mu,sigma)


# Call the `testClassifier` and `plotBoundary` functions for this part.

#testClassifier(BayesClassifier(), dataset='iris', split=0.7)

#testClassifier(BayesClassifier(), dataset='vowel', split=0.7)

#plotBoundary(BayesClassifier(), dataset='iris',split=0.7)


# ## Boosting functions to implement
# 
# The lab descriptions state what each function should do.






# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list (trained classifiers)
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # ==========================
        error_iter = get_classification_error(vote, labels, wCur)

        alpha = 1/2 * (np.log(1 - error_iter) - np.log(error_iter))
        alphas.append(alpha) # you will need to append the new alpha
        
        # update wCur
        wCur = update_weigths(wCur, alpha, vote, labels)
        # ==========================
        
    return classifiers, alphas




def get_classification_error(vote, labels, W):    
    """prediction_error = vote == labels
    error_classifier =  np.sum(W.reshape(len(W)) * (1 - prediction_error))
    """
    error_classifier = sum([W[i] for i in range(len(vote)) if vote[i]!=labels[i]])
    if error_classifier == 0:
        error_classifier = 0.000000000001
    return error_classifier
    


def update_weigths(W, alpha, vote, labels):
    W_updated = np.zeros((len(W),1))
    
    correct_prediction= vote == labels
    
    W_updated[correct_prediction] = W[correct_prediction] * np.exp(-alpha)
    W_updated[~correct_prediction] = W[~correct_prediction] * np.exp(alpha)
    
    normalizer = np.sum(W_updated)
    W_updated_normal = W_updated/normalizer
    
    return W_updated_normal



# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        
        votes = np.zeros((Npts, Nclasses))

        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        predictions = [classifier.classify(X) for classifier in classifiers] # length T Python list of the predicted class 
        #print(len(predictions))
        predictions_np = np.transpose(np.asarray(predictions)) # rows: each point, columns: classification output of each classifier
        #print("Clasificadores: " + str(len(classifiers)))
        #print("Puntos " + str(Npts))
        #print(predictions_np.shape)
        for idx_point in range(Npts):
            for class_ in range(Nclasses):
                idx_class = np.where(predictions_np[idx_point] == class_)
                votes[idx_point, class_] = np.sum(np.asarray(alphas)[idx_class])
        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
# 
# Call the `testClassifier` and `plotBoundary` functions for this part.


#testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)


testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)



#plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)


# Now repeat the steps with a decision tree classifier.


#testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)



# testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)



# testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)



# testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)



#plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)



#plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)


# ## Bonus: Visualize faces classified using boosted decision trees
# 
# Note that this part of the assignment is completely voluntary! First, let's check how a boosted decision tree classifier performs on the olivetti data. Note that we need to reduce the dimension a bit using PCA, as the original dimension of the image vectors is `64 x 64 = 4096` elements.


#testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare this with using pure decision trees or a boosted bayes classifier. Not too bad, now let's try and classify a face as belonging to one of 40 persons!


#X,y,pcadim = fetchDataset('olivetti') # fetch the olivetti data
#xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) # split into training and testing
#pca = decomposition.PCA(n_components=20) # use PCA to reduce the dimension to 20
#pca.fit(xTr) # use training data to fit the transform
#xTrpca = pca.transform(xTr) # apply on training data
#xTepca = pca.transform(xTe) # apply on test data
# use our pre-defined decision tree classifier together with the implemented
# boosting to classify data points in the training data
#classifier = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr)
#yPr = classifier.classify(xTepca)
# choose a test point to visualize
#testind = random.randint(0, xTe.shape[0]-1)
# visualize the test point together with the training points used to train
# the class that the test point was classified to belong to
#visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])

