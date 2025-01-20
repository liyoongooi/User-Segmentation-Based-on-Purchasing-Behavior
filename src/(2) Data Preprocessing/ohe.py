import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

def ohe(data):
    categorical_cols = data.select_dtypes(include=['object']).columns
    numerical_cols = data.select_dtypes(exclude=['object']).columns
    
    encoder = OneHotEncoder(sparse=False) 
    encoded_categories = encoder.fit_transform(data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(input_features=categorical_cols), index=data.index)
    df_clean = pd.concat([encoded_df, data[numerical_cols]], axis=1)
    
    return df_clean