import numpy as np
import math
import random 


def random_rearrange (X_tr, y_tr, seed):
    np.random.seed(seed)
    np.random.shuffle(X_tr)
    np.random.seed(seed)
    np.random.shuffle(y_tr)

## Need regularization here!
def updateW(X_tr, y_tr, w, b, eps):
    no_data = X_tr.shape[0]
    brac_term = np.dot(X_tr, w) - y_tr + b
    grad_f = 1/no_data*(np.dot(X_tr.T, brac_term))
    w = w - eps*grad_f
    return w
    

def updateB(X_tr, y_tr, w, b, eps):
    brac_term = np.dot(X_tr, w) - y_tr + b
    grad_f = np.mean(brac_term)
    b = b - eps*grad_f
    return b
    

def training(X_tr, y_tr, w, b, n_squig, eps, alpha, epochs):
    no_data = X_tr.shape[0]
    for epoch in range(0, epochs):
        data_remain = True
        n_curr = 0
        n_next = n_squig
        while(data_remain):
            X_tr_temp = X_tr[n_curr:(min(n_next, no_data))]
            n_curr = n_next
            n_next += n_squig

            data_remain = True if n_next<no_data else False

            w = updateW(X_tr, y_tr, w, b, eps)
            b = updateB(X_tr, y_tr, w, b, eps)
        print(test_data(X_tr, y_tr, w, b))
    return w,b
        

def cross_validation(X_tr, y_tr, w, b, validation_perct):

    # 10 folds data logic here
    # Preferably separate function
    
    # for 10 folds:
        # w, b = training(X_tr_act, y_tr_act, w, b, n_squig, eps, alpha, epochs)
        # error += test_data(X_tr, y_tr, w, b)
    # error = error/10  #(10 folds)

    
    return error


def double_cross_validation(X_tr, y_tr, w, b, validation_perct):
    no_data = X_tr.shape[0]
    
    # Defining values:
    tr_perct = 1-validation_perct
    no_data_act = round(no_data*tr_perct)
    
    minibatch_perct = np.divide(np.array([1, 5, 10, 25]), 100)
    eps_set = np.array([0.1, 0.01, 0.001, 0.0001])    
    alpha_set = np.array([0.1, 0.01, 0.001, 0.0001])    
    epochs_set = np.array([100, 250, 500, 1000])
    
    h_star = [minibatch_perct[0], eps_set[0], alpha_set[0], epochs_set[0]]

    # 10 folds data logic here
    # Preferably separate function
    # for 10 folds:
    #     A = +math.inf
    #     for minibatch in minibatch_perct:
    #         for eps in eps_set:
    #             for alpha in alpha_set:
    #                 for epochs in epochs_set:
    #                     n_squig = round(no_data_act*minibatch_perct)
    #                     A_curr = cross_validation(X_fold, y_fold, w, b, validation_perct)
    #                     if(A_curr<A):
    #                         A = A_curr
    #                         h_star = [minibatch, eps, alpha, epochs]
    
    # return minibatch, eps, alpha, epochs
    
    
    
    return n_squig, eps, alpha, epochs


def stoch_grad_regression (X_tr, y_tr):

    no_data = X_tr.shape[0]
    no_features = X_tr.shape[1]

    # Step 1, random w and b generation
    w = np.random.rand(no_features)
    b = random.random()

    validation_perct = 20/100

    # Randomizing the data
    random_rearrange(X_tr, y_tr, 10) #seed can be any random number
    n_squig, eps, alpha, epochs = double_cross_validation(X_tr, y_tr, w, b, validation_perct)
    w,b = training(X_tr_act, y_tr_act, w, b, n_squig, eps, alpha, epochs)
    return w,b
    

def test_data(X_te, y_te, w, b):
    f = 0
    for i, x in enumerate(X_te):
        f += (np.dot(x,w) - y_te[i] + b)**2
    f = f/(X_te.shape[0])
    return f


    
def train_age_regressor ():# train_age_regressor()
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    y_tr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    y_te = np.load("age_regression_yte.npy")

    w,b = stoch_grad_regression(X_tr, y_tr)
    testing_age = test_data(X_te, y_te, w, b)
    print(test_data(X_te, y_te, w, b))

    

    # Report fMSE cost on the training and testing data (separately)
    # ...


if __name__ == '__main__':
    train_age_regressor()