# GGD

There lists three .R files, where 'sngd' refers to ggd-sgd algorithm writting in R script and 'ssvrg', 'snarah' refer to ggd-svrg and ggd-sarah+ algorithm respectively. All three programs are designed to fit a Logistic regression as illustration in section 4 of our work.

## Demonstration for arguments

The arguments used in all three programs includes

1. 'trainset' : data from training set, it does not include the label of data 
2. 'trainset.label' : label of training samples, they should be encoded to 0 and 1.
3. 'testset' : data from testing set without label.
4. 'testset.label' : label of testing samples, preprocessed as 'trainset.label'.
5. 'initial.beta' : starting point for optimization, it is set to be all zero vector whose dimension is equal to the number of features.
6. 'lambda' : the penalty constant in regularized logistic regression.
7. 'epoch' : the number of whole training set being passed through once.
8. 'record.time' : the number of train loss, test loss and gradnorm being recorded in a single epoch, it is set to be 17 in our experiments.
9. 'stepsize' : learning rate for stochastic optimization algorithm
10. 'gradient.based' : choose which resampling distribution to use in training process. 'True' refers to optimal resampling distribution and 'False' refers to the resampling distribution defined in terms of loss function values.
11. 'adaptive.stepsize' : only used in 'sngd', determines whether a diminishing stepsize sequence is used or not in training process.
12. 'b' : the batch size
13. 'update.frequency' : only used in 'ssvrg' and 'snarah', determines the maximum iteration number of inner loop.
14. 'eta' : only used in 'snarah', helps the user to choose the iteration number of inner loop adaptively.

