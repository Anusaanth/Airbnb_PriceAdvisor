import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns
from sklearn.tree import export_text


feature_col =['room_type', 'person_capacity', 'host_is_superhost', 'multi', 'guest_satisfaction_overall', 'dist', 'metro_dist', 'week_time',
                'city_athens', 'city_barcelona', 'city_berlin' , 'city_lisbon', 'city_london', 'city_paris', 'city_rome', 'city_vienna']


def lin_reg_vis(reg_model,X_train, X_test, y_train, y_test):

    print(reg_model.intercept_)
    co = reg_model.coef_
    print()
    print("Coeffients:")
    for i in range(len(feature_col)):
        print(feature_col[i]+ " = " + str(co[i]))

    print()
    print(tuple(zip(feature_col, reg_model.coef_)))
    print()

    y_train_pred = reg_model.predict(X_train)
    y_test_pred = reg_model.predict(X_test)

    training_accuracy = r2_score(y_train, y_train_pred)
    test_accuracy = r2_score(y_test, y_test_pred)
    explained_variance = explained_variance_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print ('Regression: R^2 score on training set', training_accuracy)
    print ('Regression: R^2 score on test set', test_accuracy)
    print("Explained_variance",explained_variance)
    print("Mean absolute Error: ", mae) 
    print("Root Square Error: ", mse) 
    print("Root Mean Square Error: ", rmse)


def ridge_vis(fit_rcv,X_train, X_test, y_train, y_test):
   
    print()
    print(tuple(zip(feature_col, fit_rcv.coef_)))
    print()

    y_train_pred = fit_rcv.predict(X_train)
    y_test_pred = fit_rcv.predict(X_test)
    training_accuracy = r2_score(y_train, y_train_pred)
    test_accuracy = r2_score(y_test, y_test_pred)
    explained_variance = explained_variance_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print ('Regression: R^2 score on training set', training_accuracy)
    print ('Regression: R^2 score on test set', test_accuracy)
    print("Explained_variance",explained_variance)
    print("Mean absolute Error: ", mae) 
    print("Root Square Error: ", mse) 
    print("Root Mean Square Error: ", rmse)
    print("Value of alpha " + str(fit_rcv.alpha_))


def lasso_vis(fit_lcv,X_train, X_test, y_train, y_test):
    
    print()
    print(tuple(zip(feature_col, fit_lcv.coef_)))
    print()

    y_train_pred = fit_lcv.predict(X_train)
    y_test_pred = fit_lcv.predict(X_test)
    training_accuracy = r2_score(y_train, y_train_pred)
    test_accuracy = r2_score(y_test, y_test_pred)
    explained_variance = explained_variance_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print ('Regression: R^2 score on training set', training_accuracy)
    print ('Regression: R^2 score on test set', test_accuracy)
    print("Explained_variance",explained_variance)
    print("Mean absolute Error: ", mae) 
    print("Root Square Error: ", mse) 
    print("Root Mean Square Error: ", rmse)
    print("Value of alpha " + str(fit_lcv.alpha_))


def dt_vis(fit_dt,X_train, X_test, y_train, y_test):
    
    #tree_rules = export_text(fit_dt, feature_names=list(X_train.columns))
    y_train_pred = fit_dt.predict(X_train)
    y_test_pred = fit_dt.predict(X_test)
    training_accuracy = r2_score(y_train, y_train_pred)
    test_accuracy = r2_score(y_test, y_test_pred)
    explained_variance = explained_variance_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    

    print ('Regression: R^2 score on training set', training_accuracy)
    print ('Regression: R^2 score on test set', test_accuracy)
    print("Explained_variance",explained_variance)
    print("Mean absolute Error: ", mae) 
    print("Root Square Error: ", mse) 
    print("Root Mean Square Error: ", rmse)
    #print(tree_rules)
    #print()


def rfr_vis(fit_rfr,X_train, X_test, y_train, y_test,x):
  
    tree_rules = export_text(fit_rfr.estimators_[0], feature_names=list(X_train.columns))
    y_train_pred = fit_rfr.predict(X_train)
    y_test_pred = fit_rfr.predict(X_test)
    training_accuracy = r2_score(y_train, y_train_pred)
    test_accuracy = r2_score(y_test, y_test_pred)
    explained_variance = explained_variance_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    
    print ('Regression: R^2 score on training set', training_accuracy)
    print ('Regression: R^2 score on test set', test_accuracy)
    print("Explained_variance",explained_variance)
    print("Mean absolute Error: ", mae) 
    print("Root Square Error: ", mse) 
    print("Root Mean Square Error: ", rmse)
    print()

    #Print rfr rules, however it is very long so it is commented out
    '''
    for i in range(50):
        print(tree_rules[i])

    print()
    '''

    # Plot the decision tree, take long time to run so it is commented out
    '''
    estimator = fit_rfr.estimators_[0] 
    plt.figure(figsize=(12, 8)) 
    plot_tree(estimator, filled=True)
    plt.draw() 
    plt.pause(0.001)
    input("Press [enter] to continue.")
    '''

    #Scatter plot of predicted vs actual price
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.axis([0,100,0,100])
    plt.ylabel('Predicted Prices')
    plt.xlabel('Actual Prices')
    plt.title('Predicted Prices vs Actual Prices')
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    plt.close()
    
    #Corrleation of data
    corr = X_train.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool_), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True)
    #plt.show()
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    plt.close()

    #Plots a horizontal bar chart of the top (by default 10) important features in the random forest model.
    top = 10
    importances = fit_rfr.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12,8))
    plt.barh(range(top), importances[indices][:top], align="center")
    plt.yticks(range(top), [x.columns[i] for i in indices[:top]])
    plt.xlabel("Relative Importance")
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    plt.close()