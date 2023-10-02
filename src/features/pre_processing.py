from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)

def pre_proc():
    
    #Input and Output File
    input_file = '../data/raw/euro_data.csv'
    output_filepath = '../data/processed/euro_data_proc.csv'
    euro_data = pd.read_csv(input_file)
    
    #Print colums and head
    print(euro_data.columns)
    print(euro_data.head())

    #Remove column that is not needed
    euro_data.drop(columns=['Unnamed: 0'], inplace=True)

    #Replace Categorical variable with numbers
    euro_data.week_time = euro_data.week_time.apply(lambda string: 1 if string == 'weekdays' else 0)
    euro_data.room_type = euro_data.room_type.apply(lambda string: 1 if string == 'Entire home/apt' else 0)

    #Check for any null values
    print(euro_data.isna().sum())

    #Description of data
    print(euro_data.describe())

    #Number of data each city has
    print(euro_data['city'].value_counts())

    #See correlation
    corr = euro_data.corr(numeric_only = True)
    #sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(240,10,as_cmap=True),square=True)
    #plt.show(block=False)

    #Write database to output file    
    euro_data.to_csv(output_filepath, index=False)

    
    




