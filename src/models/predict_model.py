import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV

feature_col =['room_type', 'person_capacity', 'host_is_superhost', 'multi', 'guest_satisfaction_overall', 'dist', 'metro_dist', 'week_time',
                'city_athens', 'city_barcelona', 'city_berlin' , 'city_lisbon', 'city_london', 'city_paris', 'city_rome', 'city_vienna']
alpha = 10**np.linspace(4,-2,100)


#Function to call all regression modells
def predict(X_train,y_train):

    #Linear Regression 
    print('#'*50)
    print("Linear Regression...")
    reg_model = lin_reg(X_train,y_train)

    #Ridge Regression
    print('#'*50)
    print("Ridge Regression...")
    fit_rcv = ridge(X_train,y_train)

    #Lasso Regression
    print('#'*50)
    print("Lasso Regression...")
    fit_lcv = lasso(X_train,y_train)

    #Decision tree Regression
    print('#'*50)
    print("Decision Tree Regression...")
    fit_dt = dt(X_train,y_train)

    #Random Forest Regression
    print('#'*50)
    print("Random Forest Regression...")
    fit_rfr = rfr(X_train,y_train)

    return reg_model,fit_rcv,fit_lcv,fit_dt,fit_rfr


#Function to Instantiate and fit a linear regression
def lin_reg(X_train,y_train):
    #fit the model using X_train and y_train
    reg_model = LinearRegression() 
    reg_model.fit(X = X_train, y= y_train)

    return reg_model

#Function to Instantiate and fit a linear regression
def ridge(X_train,y_train):
    # define the model (use alphas=alpha, cv=10)  and fit the model using X_train and y_train
    fit_rcv = RidgeCV(alphas=alpha, cv=10)
    fit_rcv.fit(X_train, y_train)

    return fit_rcv

#Function to Instantiate and fit a lasso regression
def lasso(X_train,y_train):
    # define the model (use alphas=alpha, cv=10)  and fit the model using X_train and y_train
    fit_lcv = LassoCV(alphas=alpha, cv=10)
    fit_lcv.fit(X_train, y_train)

    return fit_lcv

#Function to Instantiate and fit a dt regression
def dt(X_train,y_train):
    #fit the model using X_train and y_train
    fit_dt = DecisionTreeRegressor()
    fit_dt.fit(X_train, y_train)

    return fit_dt

#Function to Instantiate and fit a rfr regression
def rfr(X_train,y_train):
    #fit the model using X_train and y_train
    fit_rfr = RandomForestRegressor(n_estimators = 60,max_depth=70)
    fit_rfr.fit(X_train,y_train)

    return fit_rfr

