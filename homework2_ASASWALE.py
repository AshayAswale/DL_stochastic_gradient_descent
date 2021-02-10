import numpy as np
import random 


no_data = 0
no_features = 0
no_data_act = 0 #number of data points actually available for training


def random_rearrange (X_tr, y_tr, seed):
    np.random.seed(seed)
    np.random.shuffle(X_tr)
    np.random.seed(seed)
    np.random.shuffle(y_tr)

def updateW(X_tr, y_tr, w, b, eps):
    brac_term = np.dot(X_tr, w) - y_tr + b
    grad_f = 1/no_data_act*(np.dot(X_tr.T, brac_term))
    w = w - eps*grad_f
    return w
    

def updateB(X_tr, y_tr, w, b, eps):
    brac_term = np.dot(X_tr, w) - y_tr + b
    grad_f = np.mean(brac_term)
    b = b - eps*grad_f
    return b
    

def training(X_tr, y_tr, w, b, n_squig, eps, alpha, epochs):
    for epoch in range(0, epochs):
        data_remain = True
        n_curr = 0
        n_next = n_squig
        while(data_remain):
            X_tr_temp = X_tr[n_curr:(min(n_next, no_data_act))]
            n_curr = n_next
            n_next += n_squig

            data_remain = True if n_next<no_data_act else False

            w = updateW(X_tr, y_tr, w, b, eps)
            b = updateB(X_tr, y_tr, w, b, eps)
        print(test_data(X_tr, y_tr, w, b))
    return w,b
        


def stoch_grad_regression (X_tr, y_tr):
    global no_data
    global no_features
    global no_data_act

    no_data = X_tr.shape[0]
    no_features = X_tr.shape[1]

    # Step 1, random w and b generation
    w = np.random.rand(no_features)
    b = random.random()

    # Randomizing the data
    random_rearrange(X_tr, y_tr, 10) #seed can be any random number

    # Defining values:
    validation_perct = 20/100
    tr_perct = 1-validation_perct
    no_data_act = round(no_data*tr_perct)
    minibatch_perct = 10/100

    n_squig = round(no_data_act*minibatch_perct)
    eps = 0.001
    alpha = 0.1
    epochs = 100 #1000 works better, testing error 235

    X_tr_act = X_tr[0:no_data_act]
    y_tr_act = y_tr[0:no_data_act]
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