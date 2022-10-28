import pandas as pd
import numpy as np
import pickle
import pyodbc 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
pd.options.mode.chained_assignment = None
import  mysql.connector
pd.options.display.float_format = '{:.5f}'.format

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from pandasql import sqldf
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
pd.options.mode.chained_assignment = None
import  mysql.connector
pd.options.display.float_format = '{:.5f}'.format


def read_data():
    df=pd.read_csv('PROVIDUS_CREDIT_TRAIN_TEST_DATA.csv',low_memory=False)
      
    return df


def pre_processing(df):
    df = df.apply(pd.to_numeric, errors  = 'ignore', downcast = 'float')
    df.FORCAST_DATE=pd.to_datetime(df.FORCAST_DATE)
    Day_name=df.FORCAST_DATE.dt.day_name()
    df['Day_name']=Day_name
    weekend=['Thursday', 'Friday', 'Monday', 'Wednesday', 'Tuesday']
    df=df[df.Day_name.isin(weekend)]
    df=df[['CREDIT_1_Day_Back','CREDIT_2_Day_Back', 'CREDIT_3_Day_Back',
       'CREDIT_4_Day_Back', 'CREDIT_5_Day_Back', 'CREDIT_6_Day_Back',
       'CREDIT_7_Day_Back', 'CREDIT_8_Day_Back', 'CREDIT_9_Day_Back',
       'CREDIT_10_Day_Back', 'CREDIT_11_Day_Back', 'CREDIT_12_Day_Back',
       'CREDIT_13_Day_Back', 'CREDIT_14_Day_Back', 'MAX_CREDIT_14_Day_Back',
       'Min_CREDIT_14_Day_Back', 'Sum_CREDIT_14_Day_Back',
       'Avg_CREDIT_14_Day_Back','NEXT_Number_1',
       'NEXT_Number_2', 'NEXT_Number_3', 'Previous_Number_1',
       'Previous_Number_2', 'Previous_Number_3','PREBUBHOL',
       'POSTBUBHOL','Target_Credit']]
    df=df[df.Target_Credit>=10000]
    return df


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, x_train,y_train,x_test, pred):
    r_square_train=pipe.score(x_train,y_train)
    r_square_test=pipe.score(x_test,y_test)
    mae=mean_absolute_error(y_test,prediction)
    mse=sqrt(mean_squared_error(y_test,prediction))
    return r_square_train, r_square_test, mae, mse



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    df=read_data()
    
    df=pre_processing(df)
        
        
    # Split the data into training and test sets. (0.75, 0.25) split.
    
    
    y=df.Target_Credit
    x=df.drop('Target_Credit',axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10)
    col=list(x.select_dtypes(include=['int64','float32']).columns)

    with mlflow.start_run():
        num_cols=Pipeline(steps=[('impute',SimpleImputer(strategy='constant',fill_value=-999)),
                             ('scaler',StandardScaler())])

        preprocess=ColumnTransformer(transformers=[('num_cols',num_cols,col)])
        pipe=Pipeline(steps=[('preprocess',preprocess),('Linear',LinearRegression())])

        pipe.fit(x_train,y_train)

        prediction=pipe.predict(x_test)

        (r_square_train, r_square_test, mae, mse) = eval_metrics(y_test,x_train,y_train,x_test,prediction)

        print('the training score is',r_square_train)
        print('the training score on test data',r_square_test)
        print('the mean absolute error is ',mae)
        print('the rmse absolute error is ',mse)
        
        #mlflow.log_param("alpha", alpha)
        mlflow.log_param("r_square_train", r_square_train)
        mlflow.log_metric("r_square_test", r_square_test)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(pipe, "model", registered_model_name="LinearRegressionModel")
        else:
            mlflow.sklearn.log_model(pipe, "model")