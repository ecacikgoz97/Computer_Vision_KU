from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg, regtype='L2'):
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
    - regtype: Regularization type: L1 or L2

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
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_samples = X.shape[0]    
    num_classes = W.shape[1]
    
    # scores: shape (N, C)
    scores = np.dot(X, W)
    
    for sample in range(num_samples):
        # scaling the values for numeric stability, i.e. make highest number 0
        x = scores[sample] - np.max(scores[sample])
        
        # calculate loss
        softmax = np.exp(x) / np.sum(np.exp(x))
        loss = loss - np.log(softmax[y[sample]])
     
        # gradient operations
        for c in range(num_classes):
            dW[:,c] += X[sample] * softmax[c]
        dW[:,y[sample]] = dW[:,y[sample]] - X[sample]
        
    # regularization operation
    # regularization
    if regtype == 'L1':
        regularization = reg * np.sum(W)
    elif regtype == 'L2':
        regularization = reg * np.sum(W*W)
    else:
        raise TypeError("Only regtype='L1' and regtype='L2' are allowed ")
        
    loss = loss/num_samples + regularization 
    
    dW = dW / num_samples
    dW += 2*W*reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg, regtype='L2'):
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
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_samples = X.shape[0]    
    num_classes = W.shape[1]
    
    # scores: shape (N, C)
    scores = np.dot(X, W)
    scores = scores - np.max(scores)
    scores_exp = np.exp(scores)
    #softmax = scores_exp / np.sum(scores_exp, axis=1)
    softmax = scores_exp[range(num_samples), y] / np.sum(scores_exp, axis=1)
    
    # regularization
    if regtype == 'L1':
        regularization = reg * np.sum(W)
    elif regtype == 'L2':
        regularization = reg * np.sum(W*W)
    else:
        raise TypeError("Only regtype='L1' and regtype='L2' are allowed ")
    
    # loss
    loss = np.sum(-np.log(softmax))
    loss = loss/num_samples + regularization
    
    # gradient operations
    d = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
    d[np.arange(num_samples), y] = d[np.arange(num_samples), y] - 1 
    dW = np.dot(X.T, d) / num_samples
    dW = dW + 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
