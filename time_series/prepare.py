import numpy as np
import pandas as pd
import os

from datetime import timedelta, datetime

import acquire

#-----------#
#  Helpers  #
#-----------#

def set_datetime_index(df, col):
    df[col] = pd.to_datetime(df[col])
    df = df.sort_values(col).set_index(col)
    return df

def make_year_col(df):
    df['year'] = df.index.year
    return df

def make_month_col(df):
    df['month'] = df.index.month
    return df

def make_day_col(df):
    df['day_of_week'] = df.index.day_name()
    return df

def make_sales_total_col(df):
    df['sales_total'] = df.sale_amount * df.item_price
    return df

def make_sales_diff_col(df):
    df['day_sales_diff'] = df.sale_amount.diff(1)
    return df

#--------------#
#  Sales Data  #
#--------------#

def prep_store_data():
    # check if csv exists, if not create raw df to be modified
    if os.path.exists('prepped_store_data.csv'):
        df = pd.read_csv('prepped_store_data.csv', index_col=0)
        return df
    else:
        df = get_store_data()
        # prep df per instructions: set sale_date as datetime column, make it the index, make new month col based on the sale_date index, make new day col based on sale_date index
        df = set_datetime_index(df, 'sale_date')    
        df = make_month_col(df)
        df = make_day_col(df)
        # create a new column derived from sale_amount (total items) and item_price
        df = make_sales_total_col(df)
        # create a new column that is the result of the current sales - the previous days sales
        df = make_sales_diff_col(df)
        # write modified df to new csv file
        df.to_csv('prepped_store_data.csv')
    return df

#---------------------#
#  German Power Data  #
#---------------------#

def prep_ops_data():
    # check if csv exists, if not create raw df to be modified
    if os.path.exists('prepped_ops_data.csv'):
        df = pd.read_csv('prepped_ops_data.csv', index_col=0)
        return df
    else:
        df = acquire.read_german()
        # prep df per instructions: set Date as datetime column, make it the index, make new month col based on the Date index, make new year col based on Date index
        df = set_datetime_index(df, 'Date')    
        df = make_month_col(df)
        df = make_year_col(df)
        # write modified df to new csv file
        df.to_csv('prepped_ops_data.csv')
    return df