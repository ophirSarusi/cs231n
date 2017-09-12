import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    
    num_positive_margin = 0.0 # Number of positive margin
    dLi = np.zeros(W.shape)
    
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dLi[:,j] = X[i,:]
        num_positive_margin += 1.0
        
    dLi[:,y[i]] = - num_positive_margin * X[i]
    dW += dLi
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW = (1.0/num_train) * dW + reg * W 

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

#compute all the scores
  scores = X.dot(W)
  
  marginVal = 1
  # Extract the scores of the correct class  
  #correct_class_score = np.choose(y,scores.T)
  num_train = X.shape[0]
  
  sequence =  np.arange(num_train)
  correct_class_score = scores[sequence, y]
    
  margins = np.maximum(0, scores.T - correct_class_score + marginVal)  
  margins[y, sequence] = 0
    
  # Sum over the classes
  loss = np.sum(margins)
  loss /= num_train
  loss += reg * 0.5 * np.sum(W * W)

  margins = np.where(margins > 0, 1, 0)
  margins[y, np.arange(0, scores.shape[0])] = -1 * np.sum(margins, 0)
  dW = np.dot(margins, X)

  dW /= num_train
  
  dW += reg * W.T
  dW = dW.T 

  return loss, dW
