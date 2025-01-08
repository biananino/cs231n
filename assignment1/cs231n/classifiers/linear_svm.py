from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
          if j == y[i]:
            continue
          margin = scores[j] - correct_class_score + 1 # note delta = 1
          if margin > 0:
            loss += margin
            dW[:,j] += X[i]
            dW[:,y[i]] -= X[i] 


    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train



    # Add regularization to the gradient
    dW = dW + 2*reg*W/num_train
    
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_inputs = y.shape[0]
    scores = np.dot(X,W)
    correct_scores = scores[range(num_inputs),y]
    margin = (scores - np.matrix(correct_scores).T + 1)

    losses = np.maximum(margin,np.zeros(margin.shape))
    loss = np.sum(losses)
    loss -= num_inputs
    #Regularization loss
    loss += reg*np.sum(W*W)   
    loss /= num_inputs
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #First i create a matrix with the derivative of the loss with respect to the scores
    margin_bool = np.where(margin > 0, 1, 0) #Matrix with 1s whenever the margin is positive
    over_margin_counter = np.sum(margin_bool, axis = 1) - 1 #Counts the number of positive margins in a row
    margin_bool[range(num_inputs),y] = - over_margin_counter 
    #Then i apply chain rule to get the gradient due to SVM loss
    dW = np.dot(X.T,margin_bool)
    #Add regularization
    dW += 2*reg*W    
    dW /= num_inputs
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
