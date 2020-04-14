import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from env import get_db_url

def get_titanic_data():
    sql = """
    SELECT *
    FROM passengers
    """
    url = get_db_url('titanic_db')
    return pd.read_sql(sql, url)

def get_iris_data():
    sql = """
    SELECT * 
    FROM measurements
    JOIN species USING (species_id)
    """
    url = get_db_url('iris_db')
    return pd.read_sql(sql, url)

def split_my_data(df):
    '''
    Takes in df, train_pct and returns 2 items: train, test

    When using this function, in order to have usable datasets, be sure to call it thusly:
    train, test = split_my_data(df, train_pct)
    '''
    return train_test_split(df, train_size = 0.8, random_state = 294)

def ohe_encode_iris(df):
    encoded_values = sorted(list(df.species.unique()))
    le = preprocessing.LabelEncoder()
    iris_enc = le.fit_transform(df.species)
    iris_array = np.array(iris_enc).reshape(len(iris_enc), 1)
    ohe = preprocessing.OneHotEncoder(sparse=False, categories='auto')
    iris_ohe = ohe.fit_transform(iris_array)
    iris_enc = pd.DataFrame(data=iris_ohe, columns=encoded_values, index=df.index)
    iris = df.join(iris_enc)
    return iris

def prep_iris(df):
    """
    Takes in original iris df generated by acquire.get_iris_data() and returns
    a new df with transformations.
    Drops columns ['species_id', 'measurement_id'] since they aren't useful.
    Renames ['species_name'] to 'species' 
    Encodes species column as one_hot variables
    Splits the df into 80% train set, 20% test set
    """
    
    df = df.drop(columns=['species_id', 'measurement_id'])
    df = df.rename(columns={'species_name': 'species'})
    df = ohe_encode_iris(df)
    train, test = split_my_data(df)
    return train, test

def ohe_encode_titanic(df):
    encoded_values = sorted(list(df.embarked.unique()))
    le = preprocessing.LabelEncoder()
    titanic_enc = le.fit_transform(df.embarked)
    titanic_array = np.array(titanic_enc).reshape(len(titanic_enc), 1)
    ohe = preprocessing.OneHotEncoder(sparse=False, categories='auto')
    titanic_ohe = ohe.fit_transform(titanic_array)
    titanic_enc = pd.DataFrame(data=titanic_ohe, columns=encoded_values, index=df.index)
    df = df.join(titanic_enc)
    return df

def mm_scale_titanic(df):
    X = df[['age', 'fare']]
    scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(0,1)).fit(X)
    X = scaler.transform(X)
    X = pd.DataFrame(X)
    df['age_mm'] = X.iloc[:, :1]
    df['fare_mm'] = X.iloc[:, 1:2]
    return df

def prep_titanic(df):
    """
    Takes in original titanic df generated by acquire.get_titanic_data() and returns
    a new df with transformations.
    Fills null values in embark_town and embarked with the most frequent entry: Southampton.
    Encodes embarked column as one_hot variables.
    Scales age and fare columns with min-max scaler.
    Drops columns ['deck', 'embarked', 'age', 'fare'] since they are obsolete.
    Splits the df into 80% train set, 20% test set
    """
    df.embark_town = df.embark_town.fillna('Southampton')
    df.embarked = df.embarked.fillna('S')
    df = ohe_encode_titanic(df)
    df = mm_scale_titanic(df)
    df = df.drop(columns=['deck', 'embarked', 'age', 'fare'])
    train, test = split_my_data(df)
    return train, test