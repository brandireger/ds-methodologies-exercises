import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from env import get_db_url

def split_my_data(df, train_size):
    '''
    Takes in df, train_pct and returns 2 items: train, test

    When using this function, in order to have usable datasets, be sure to call it thusly:
    train, test = split_my_data(df, train_pct)
    '''
    return train_test_split(df, train_size = train_size, random_state = 294)

def get_titanic_data():
    sql = """
    SELECT *
    FROM passengers
    """
    url = get_db_url('titanic_db')
    return pd.read_sql(sql, url)

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

def sex_encode(df):
    le = preprocessing.LabelEncoder()
    df['sex_enc'] = le.fit_transform(df.sex)
    return df

def mm_scale_titanic(df):
    X = df[['age_imp', 'fare']]
    scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(0,1)).fit(X)
    X = scaler.transform(X)
    X = pd.DataFrame(X)
    df['age_mm'] = X.iloc[:, :1]
    df['fare_mm'] = X.iloc[:, 1:2]
    return df

def prep_titanic(df):
    """
    Takes in original titanic df generated by acquire.get_titanic_data() and returns a new df with transformations.
    Fills null values in embark_town and embarked with the most frequent entry: Southampton.
    Fills null values in age with the average of class
    One-hot encodes embarked column, label encodes sex column.
    Scales age and fare columns with min-max scaler.
    Drops column ['deck'] since they are mostly null.
    """
    df.embark_town = df.embark_town.fillna('Southampton')
    df.embarked = df.embarked.fillna('S')
    df['age_imp'] = (df["age"]
                     .fillna(df.groupby("pclass")["age"]
                             .transform("median")))
    df = ohe_encoder(df, 'embarked')
    df = sex_encode(df)
    df = mm_scale_titanic(df)
    df = df.drop(columns=['deck'])
    return df

def get_iris_data():
    sql = """
    SELECT * 
    FROM measurements
    JOIN species USING (species_id)
    """
    url = get_db_url('iris_db')
    return pd.read_sql(sql, url)

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




def string_to_numerical_labeler(df, col):
    le = preprocessing.LabelEncoder()
    df[col] = le.fit_transform(df[col])
    return df, le

def inverse_labeler(df, col, le):
    df[col] = le.inverse_transform(df[col])
    return df