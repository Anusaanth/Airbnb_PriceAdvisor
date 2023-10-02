from data.make_dataset import main

#
#To run project type "cd src" and then"python main.py" in terminal on Visual code studio
#

print("Data:")
print("...")
main()
print('-'*100)
print()


from features.pre_processing import pre_proc
print("Pre-processing:")
print()
pre_proc()
print('-'*100) 
print()


from models.predict_model import predict
from models.train_model import train
print("Models:")
print()
X_train, X_test, y_train, y_test,x = train()
reg_model,fit_rcv,fit_lcv,fit_dt,fit_rfr = predict(X_train,y_train)
print('-'*100)
print()


from visualization.visualize import lin_reg_vis,ridge_vis,lasso_vis,dt_vis,rfr_vis
print("Visualization:")
print()

print('#'*50)
print()
print("Linear Regression")
lin_reg_vis(reg_model,X_train, X_test, y_train, y_test)

print('#'*50)
print()
print("Ridge Regression")
ridge_vis(fit_rcv,X_train, X_test, y_train, y_test)

print('#'*50)
print()
print("Lasso Regression")
lasso_vis(fit_lcv,X_train, X_test, y_train, y_test)

print('#'*50)
print()
print("Decision Tree Regression")
dt_vis(fit_dt,X_train, X_test, y_train, y_test)

print('#'*50)
print()
print("Random Forest Regression")
rfr_vis(fit_rfr,X_train, X_test, y_train, y_test,x)
print()  