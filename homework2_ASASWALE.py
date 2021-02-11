import numpy as np
import math
import random 
import copy


def random_rearrange (X_tr, y_tr, seed):
    np.random.seed(seed)
    np.random.shuffle(X_tr)
    np.random.seed(seed)
    np.random.shuffle(y_tr)


def updateW(X_tr, y_tr, w, b, eps, alpha):
    no_data = X_tr.shape[0]
    brac_term = np.dot(X_tr, w) - y_tr + b
    grad_f_mse = 1/no_data*(np.dot(X_tr.T, brac_term))
    regularisation = alpha/(no_data)*(w)
    grad_f = grad_f_mse + regularisation
    w -= eps*grad_f
    return w
    

def updateB(X_tr, y_tr, w, b, eps):
    brac_term = np.dot(X_tr, w) - y_tr + b
    grad_f = np.mean(brac_term)
    b -= eps*grad_f
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

            w = updateW(X_tr, y_tr, w, b, eps, alpha)
            b = updateB(X_tr, y_tr, w, b, eps)

    return w,b
        

def cross_validation(X_tr, y_tr, n_squig, eps, alpha, epochs, no_of_folds):
    no_data = X_tr.shape[0]
    no_features = X_tr.shape[1]
    
    w = np.random.rand(no_features)
    b = random.random()
    no_of_data_per_fold = no_data/no_of_folds   
    n_curr = 0
    n_next = no_of_data_per_fold
    acc = np.zeros([no_of_folds])
    
    for k in range(0,no_of_folds):
        X_tr_local = copy.deepcopy(X_tr)
        y_tr_local = copy.deepcopy(y_tr)


        X_tr_valid = X_tr_local[n_curr:(min(int(n_next), no_data))]
        y_tr_valid = y_tr_local[n_curr:(min(int(n_next), no_data))]

        X_tr_local = np.delete(X_tr_local, range(n_curr,(min(int(n_next), no_data))), axis = 0)
        y_tr_local = np.delete(y_tr_local, range(n_curr,(min(int(n_next), no_data))), axis = 0)

        n_curr = int(n_next)
        n_next += no_of_data_per_fold

        w, b = training(X_tr_local, y_tr_local, w, b, n_squig, eps, alpha, epochs)
        acc[k] = test_data(X_tr_valid, y_tr_valid, w, b)
        
    return np.mean(acc)

    
    # for 10 folds:
        # error += test_data(X_tr, y_tr, w, b)
    # error = error/10  #(10 folds)

    
    return error


def double_cross_validation(X_tr, y_tr):
    no_data = X_tr.shape[0]
    no_features = X_tr.shape[1]

    # Step 1, random w and b generation
    w = np.random.rand(no_features)
    b = random.random()
        
    n_squig_set = np.array([50, 100, 250, 500])
    eps_set = np.array([0.1, 0.01, 0.001, 0.0001])
    alpha_set = np.array([0.01, 0.1, 0.25, 0.5])
    epochs_set = np.array([10, 50, 100, 500])
    
    h_star = [n_squig_set[0], eps_set[0], alpha_set[0], epochs_set[0]]

    no_of_folds = 2
    no_of_data_per_fold = no_data/no_of_folds   
    n_curr = 0
    n_next = no_of_data_per_fold
    
    for k in range(0,no_of_folds):
        X_tr_local = copy.deepcopy(X_tr)
        y_tr_local = copy.deepcopy(y_tr)

        X_tr_valid = X_tr_local[int(n_curr):(min(int(n_next), no_data))]
        y_tr_valid = y_tr_local[int(n_curr):(min(int(n_next), no_data))]

        X_tr_local = np.delete(X_tr_local, range(int(n_curr),(min(int(n_next), no_data))), axis = 0)
        y_tr_local = np.delete(y_tr_local, range(int(n_curr),(min(int(n_next), no_data))), axis = 0)

        n_curr = n_next
        n_next += no_of_data_per_fold

        A = +math.inf
        no_data_act = X_tr_local.shape[0]
        for n_squig in n_squig_set:
            for eps in eps_set:
                for alpha in alpha_set:
                    for epochs in epochs_set:
                        A_curr = cross_validation(X_tr_local, y_tr_local, n_squig, eps, alpha, epochs, no_of_folds)
                        print("Error:", A_curr, "h_star:", n_squig, eps, alpha, epochs)
                        if(A_curr<A):
                            A = A_curr
                            h_star = [n_squig, eps, alpha, epochs]
                            print("## h_star updated ##")
    
    print("######################")
    print("BEST HYPERPARAMETERS")
    print(n_squig, eps, alpha, epochs)
    print("######################")
    return n_squig, eps, alpha, epochs
    

def stoch_grad_regression (X_tr, y_tr):

    no_data = X_tr.shape[0]
    no_features = X_tr.shape[1]

    # Step 1, random w and b generation

    validation_perct = 20/100

    # Randomizing the data
    random_rearrange(X_tr, y_tr, 10) #seed can be any random number
    # print("Enter 1 for hyperparameter tuning\nEnter 2 for training on the tuned hyperparameters")
    value = input("Enter 1 for hyperparameter tuning\nEnter 2 for training on the tuned hyperparameters\n")
    if value == 1:
        print("Tuning Hyperparameters!")
        n_squig, eps, alpha, epochs = double_cross_validation(X_tr, y_tr)
    else:
        print("Training using pretuned hyperparameters")
        n_squig, eps, alpha, epochs = 50, 0.001, 0.1, 500

    w = np.random.rand(no_features)
    b = random.random()
    w,b = training(X_tr, y_tr, w, b, n_squig, eps, alpha, epochs)
    return w,b
    

def test_data(X_te, y_te, w, b):
    f = np.square(np.dot(X_te, w)-y_te + b)
    err = np.mean(f)/2
    return err


    
def train_age_regressor ():# train_age_regressor()
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    y_tr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    y_te = np.load("age_regression_yte.npy")

    w,b = stoch_grad_regression(X_tr, y_tr)
    testing_age = test_data(X_te, y_te, w, b)
    print("################################")
    print("FINAL TESTING ERROR:", testing_age)
    print("################################")
    

    # Report fMSE cost on the training and testing data (separately)
    # ...


if __name__ == '__main__':
    train_age_regressor()