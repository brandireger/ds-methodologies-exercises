import numpy as np
import pandas as pd
import os
import requests

def get_pieces(piece):
    if os.path.exists(piece + '.csv'):
        df = pd.read_csv(piece + '.csv', index_col=0)
    else:
        base_url = 'https://python.zach.lol'
        response = requests.get(base_url + '/api/v1/' + piece)
        data = response.json()
        df = pd.DataFrame(data['payload'][piece])

        url = data['payload']['next_page']

        while type(url) != type(None):
            response = requests.get(base_url + url)
            data = response.json()
            df = df.append(data['payload'][piece])
            url = data['payload']['next_page']
            
        df.to_csv(piece + '.csv', index=False)

    return df

def combine_sales(df1, df2, df3):
    if os.path.exists('merged_sales.csv'):
        honkin_sales = pd.read_csv('merged_sales.csv', index_col=0)
    else:
        honkin_sales = pd.merge(left=df1, right=df2, 
                                how='outer', left_on='item',
                                right_on='item_id', copy=False)
        honkin_sales = pd.merge(left=honkin_sales, right=df3,
                                how='outer', left_on='store',
                                right_on='store_id', copy=False)
        honkin_sales.drop(columns=['item', 'store'], inplace=True)
        honkin_sales.to_csv('merged_sales.csv')
    return honkin_sales

def get_store_data():
    items_df = get_pieces('items')
    stores_df = get_pieces('stores')
    sales_df = get_pieces('sales')
    df = combine_sales(sales_df, items_df, stores_df)
    return df

def get_German_csv():
    csv_url = 'https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv'
    response = requests.get(csv_url)
    content = response.content
    csv_file = open('German_power.csv', 'wb')
    csv_file.write(content)
    csv_file.close()
    return print('download successful')

def read_german():
    if os.path.exists('German_power.csv'):
        df = pd.read_csv('German_power.csv', index_col=0)
    else:
        get_German_csv()
        df = pd.read_csv('German_power.csv', index_col=0)
    return df