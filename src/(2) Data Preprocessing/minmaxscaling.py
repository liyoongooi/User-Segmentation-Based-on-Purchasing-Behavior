import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def minmaxscaling(data):
    categorical_cols = data.select_dtypes(include=['object']).columns
    numerical_cols = data.select_dtypes(exclude=['object']).columns
    
    scaler = MinMaxScaler()
    df_num = scaler.fit_transform(data[numerical_cols])
    df_num = pd.DataFrame(df_num, columns = numerical_cols, index = data.index)
    df_clean = pd.concat([data[categorical_cols], df_num], axis=1)
    
    return df_clean