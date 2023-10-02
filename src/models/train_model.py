import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train():

    #Input file and read.csv
    input_file = '../data/processed/euro_data_proc.csv'
    euro_data = pd.read_csv(input_file)

    #Create dummy variable
    euro_data = pd.get_dummies(data=euro_data, columns=['city'])

    #Features
    feature_col =['room_type', 'person_capacity', 'host_is_superhost', 'multi', 'guest_satisfaction_overall', 'dist', 'metro_dist', 'week_time',
                'city_athens', 'city_barcelona', 'city_berlin' , 'city_lisbon', 'city_london', 'city_paris', 'city_rome', 'city_vienna']
    
    #x and y fir train test split
    x =  euro_data[feature_col]
    y = euro_data["realSum"] 

    #Train and split data
    X_train, X_test, y_train, y_test = train_test_split(x, np.sqrt(y),  test_size=0.15,random_state= 42)

    #Scaling numeric features using sklearn StandardScalar
    numeric=['person_capacity', 'guest_satisfaction_overall', 'dist','metro_dist']
    sc=StandardScaler()
    X_train[numeric]=sc.fit_transform(X_train[numeric])
    X_test[numeric]=sc.transform(X_test[numeric])

    return X_train, X_test, y_train, y_test, x


