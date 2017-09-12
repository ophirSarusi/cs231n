import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
      scores = X[i,:].dot(W) #shape (1, C)
      stability = scores.max()
      exp_scores_i = np.exp(scores-stability) #shape (1,C)
      numerator = np.exp(scores[y[i]]-stability)
      denom = np.sum(exp_scores_i)
      loss += -np.log(numerator/denom)
      
      for j in xrange(num_classes):
          if j == y[i]:
              dW[:,j] += -X[i,:].T + (exp_scores_i[j] / denom) * X[i,:].T
          else:
              dW[:,j] += (exp_scores_i[j] / denom) * X[i,:].T
  
  loss = loss / float(num_train) + 0.5 * reg * np.sum(W*W)
  dW = dW / float(num_train) + reg * W
      
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  scores = X.dot(W) #shape (N,C)
  stability = scores.max(axis=1) #shape (1,C)
  scores -= np.tile(stability,(num_classes,1)).T #shape (N,C)
  exp_scores = np.exp(scores) #shape (N,C)
  
  log_numerator = np.log(exp_scores[range(num_train),y])
  denom = np.sum(exp_scores, axis=1)
  log_denom = np.log(denom)
  
  loss = -np.mean(log_numerator-log_denom)
  
  
  p = exp_scores / np.tile(denom, (num_classes,1)).T
  ind = np.zeros(p.shape)
  ind[range(num_train),y] = 1
  dW = np.dot(X.T,(p-ind))

  
  #regularization
  loss += 0.5 * reg * np.sum(W*W)
  dW = dW / float(num_train) + reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

