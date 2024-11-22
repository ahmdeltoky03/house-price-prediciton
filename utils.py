## Main Libraries

import pandas as pd

## sklearn Modulos
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from xgboost import XGBRegressor

import os
import joblib


## read and overview Dataset
data_path = os.path.join(os.getcwd(), 'housing.csv')
df_housing = pd.read_csv(data_path) 

# print(df_housing.shape)
## modifiy some values in ocean_proximity  
df_housing['ocean_proximity'] = df_housing['ocean_proximity'].apply(lambda x : '1H OCEAN' if x == '<1H OCEAN' else x)

## Feature Extraction
df_housing['rooms_per_household'] = df_housing['total_rooms'] / df_housing['households']
df_housing['bedrooms_per_rooms'] = df_housing['total_bedrooms'] / df_housing['total_rooms']
df_housing['population_per_household'] = df_housing['population'] / df_housing['households']

## split the dataset features & target
## target --> median_house_value 
x = df_housing.drop(columns='median_house_value',axis=1)
y = df_housing['median_house_value']

## split data to train and test and shuffle it
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True,test_size=.15)

## show numerical and categorical columns
num_col = [col for col in x_train.columns if x_train[col].dtype in ['int32','int64','float32','float64']]
categ_col = [col for col in x_train.columns if x_train[col].dtype not in ['int32','int64','float32','float64']]

print('Numerical Columns : ',num_col)
print('Categorical Columns : ',categ_col)


num_pipeline = Pipeline(steps=[
                ('selector',DataFrameSelector(num_col)),
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ])
categ_pipeline = Pipeline(steps=[
                ('selector',DataFrameSelector(categ_col)),
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ohe',OneHotEncoder(sparse_output=False))
                ])
total_pipline = FeatureUnion(transformer_list=[
                                ('num',num_pipeline),
                                ('categ',categ_pipeline)
                                ])

x_train_final = total_pipline.fit_transform(x_train)

def preprocess_new(x_new):
    return total_pipline.transform(x_new)