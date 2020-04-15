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

def split_my_data(df, train_size):
    '''
    Takes in df, train_pct and returns 2 items: train, test

    When using this function, in order to have usable datasets, be sure to call it thusly:
    train, test = split_my_data(df, train_pct)
    '''
    return train_test_split(df, train_size = train_size, random_state = 294)

def species_encode(train, test):
    le = preprocessing.LabelEncoder()
    train['species_enc'] = le.fit_transform(train.species)
    test['species_enc'] = le.fit_transform(test.species)    
    return train, test

def prep_iris(df):
    """
    Takes in original iris df generated by acquire.get_iris_data() and returns
    a new df with transformations.
    Splits the df into 80% train set, 20% test set
    Drops columns ['species_id', 'measurement_id'] since they aren't useful.
    Renames ['species_name'] to 'species' 
    Encodes species column as numerical values.
    """
    train, test = split_my_data(df, 0.7)
    train = train.drop(columns=['species_id', 'measurement_id'])
    test = test.drop(columns=['species_id', 'measurement_id'])
    train = train.rename(columns={'species_name': 'species'})
    test = test.rename(columns={'species_name': 'species'})
    train, test = species_encode(train, test)
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
    train, test = split_my_data(df, 0.7)
    train.embark_town = train.embark_town.fillna('Southampton')
    test.embark_town = test.embark_town.fillna('Southampton')
    train.embarked = train.embarked.fillna('S')
    test.embarked = test.embarked.fillna('S')
    train = ohe_encoder(train, 'embarked')
    test = ohe_encoder(test, 'embarked')
    train = mm_scale_titanic(train)
    test = mm_scale_titanic(test)
    train = train.drop(columns=['deck', 'embarked', 'age', 'fare'])
    test = test.drop(columns=['deck', 'embarked', 'age', 'fare'])
    return train, test

def ohe_encoder(df, col):
    encoded_values = sorted(list(df[col].unique()))
    le = preprocessing.LabelEncoder()
    enc = le.fit_transform(df[col])
    ohe_array = np.array(enc).reshape(len(enc), 1)
    ohe = preprocessing.OneHotEncoder(sparse=False, categories='auto')
    df_ohe = ohe.fit_transform(ohe_array)
    enc = pd.DataFrame(data=df_ohe, columns=encoded_values, index=df.index)
    df = df.join(enc)
    return df

def string_to_numerical_labeler(df, col):
    le = preprocessing.LabelEncoder()
    df[col] = le.fit_transform(df[col])
    return df, le

def inverse_labeler(df, col, le):
    df[col] = le.inverse_transform(df[col])
    return df