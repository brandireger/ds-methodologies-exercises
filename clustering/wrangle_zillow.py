import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def missing_rows(df):
    new_df = pd.DataFrame(df.isna().sum())
    new_df = new_df.rename(columns={0 : 'num_rows_missing'})
    new_df['pct_rows_missing'] = (new_df.num_rows_missing / len(df)).round(4)
    return new_df

def missing_cols(df):
    new_df = pd.DataFrame(df.isnull().sum(axis=1).value_counts())
    new_df = new_df.reset_index()
    new_df = new_df.rename(columns={'index':'num_cols_missing', 0:'num_rows'})
    new_df['pct_cols_missing'] = (new_df.num_cols_missing / df.shape[1])*100
    return new_df

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df

def wrangle_zillow():
    df = pd.read_csv('zillow.csv', index_col='id')
    df = df.drop(columns='Unnamed: 0')
    # only properties that meet single use criteria
    single_use = [260, 261, 262, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)]
    # drop unnecessary columns
    df = remove_columns(df, ['architecturalstyletypeid',
                             'buildingclasstypeid',
                             'finishedsquarefeet13',
                             'finishedsquarefeet15',
                             'finishedsquarefeet50',
                             'finishedsquarefeet6',
                             'finishedfloor1squarefeet',
                             'pooltypeid10',
                             'pooltypeid2',
                             'pooltypeid7',
                             'fireplaceflag',
                             'airconditioningdesc',
                             'storydesc',
                             'heatingorsystemdesc',
                             'architecturalstyledesc',
                             'buildingclassdesc',
                             'typeconstructiondesc',
                             'yardbuildingsqft17',
                             'yardbuildingsqft26',
                             'calculatedbathnbr',
                             'fullbathcnt',
                             'threequarterbathnbr',
                             'typeconstructiontypeid',
                             'storytypeid',
                             'propertyzoningdesc', 
                             'calculatedfinishedsquarefeet', 
                             'regionidneighborhood',
                             'regionidcity',
                             'regionidzip',
                             'propertylandusetypeid',
                             'rawcensustractandblock',
                             'propertylandusedesc'
                            ])
    # fill nulls with 0 
    df.airconditioningtypeid.fillna(0, inplace=True)
    df.basementsqft.fillna(0, inplace=True)
    df.decktypeid.fillna(0, inplace=True)
    df.fireplacecnt.fillna(0, inplace=True)
    df.garagecarcnt.fillna(0, inplace=True)
    df.garagetotalsqft.fillna(0, inplace=True)
    df.hashottuborspa.fillna(0, inplace=True)
    df.lotsizesquarefeet.fillna(0, inplace=True)
    df.poolcnt.fillna(0, inplace=True)
    df.poolsizesum.fillna(0, inplace=True)
    df.taxdelinquencyflag.fillna(0, inplace=True)
    df.taxdelinquencyyear.fillna(0, inplace=True)
    # For heating type, None = 13
    df.heatingorsystemtypeid.fillna(13, inplace=True)
    # Fill nulls with most common value
    df.numberofstories.fillna(1, inplace=True)
    df.unitcnt.fillna(1, inplace=True)
    df.buildingqualitytypeid.fillna(6, inplace=True)
    df.yearbuilt.fillna(1955, inplace=True)
    # Drop rows with null values in certain columns
    df = df.dropna(subset=['structuretaxvaluedollarcnt',
                           'taxvaluedollarcnt',
                           'landtaxvaluedollarcnt',
                           'taxamount',
                           'censustractandblock',
                           'finishedsquarefeet12'])
    # set type for each column
    df = df.astype({'airconditioningtypeid': int,
                    'basementsqft': int,
                    'bedroomcnt': int,
                    'buildingqualitytypeid': int,
                    'decktypeid': int,
                    'finishedsquarefeet12': int,
                    'fips': int,
                    'fireplacecnt': int,
                    'garagecarcnt': int,
                    'garagetotalsqft': int,
                    'hashottuborspa': int,
                    'heatingorsystemtypeid': int,
                    'lotsizesquarefeet': int,
                    'poolcnt': int,
                    'poolsizesum': int,
                    'propertycountylandusecode': str,
                    'regionidcounty': int,
                    'roomcnt': int,
                    'unitcnt': int,
                    'yearbuilt': int,
                    'numberofstories': int,
                    'assessmentyear': int,
                    'taxdelinquencyflag': str,
                    'taxdelinquencyyear': int,
                    'censustractandblock': str})
    # Add column for counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange', 
                                   'Ventura'))
    return df
    
def split_my_data(df, train_pct):
    '''
    Takes in df, train_pct and returns 2 items:
    train, test

    When using this function, in order to have usable datasets, be sure to call it thusly:
    train, test = split_my_data(df, train_pct)
    '''
    return train_test_split(df, train_size = train_pct, random_state = 294)
