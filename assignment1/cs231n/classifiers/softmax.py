from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
	"""
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	num_inputs = X.shape[0]
	num_classes = W.shape[1]
	for i in range(num_inputs):
		correct_class = y[i]
		score = np.dot(X[i],W)
		score -= np.max(score)
		exp_score = np.exp(score)
		exp_sum = np.sum(exp_score)
		probabilities = exp_score/exp_sum
		for category in range(num_classes):
			class_prob = probabilities[category]
			if category == correct_class:
				loss += -np.log(class_prob)
				dW[:,category] += X[i]*(class_prob - 1)
			else:
				dW[:,category] += X[i]*class_prob			
	#Add regularization loss
	loss += reg*np.sum(W*W)
	dW += 2*reg*W
	#Normalize loss by sample size
	loss /= num_inputs 
	dW /= num_inputs

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

	return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
	"""
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
	"""
    # Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	num_inputs = y.shape[0]
	scores = np.dot(X,W)
	scores -= np.max(scores, axis = 1, keepdims = True)
	exp_scores = np.exp(scores)
	exp_sums = np.sum(exp_scores, axis = 1, keepdims = True)
	probabilities = exp_scores/exp_sums
	correct_probabilities = probabilities[range(num_inputs),y]
	loss = np.sum(-np.log(correct_probabilities))

	#Construct matrix of the gradient of the loss with respect to the scores
	probabilities[range(num_inputs),y] = correct_probabilities - 1
	#Apply chain rule to get gradient with respect to weights
	dW = np.dot(X.T,probabilities)

	#Add regularization loss 
	loss += reg*np.sum(W*W)
	dW += 2*reg*W
	#Normalize loss by sample size
	loss /= num_inputs
	dW /= num_inputs

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

	return loss, dW
