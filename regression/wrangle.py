import pandas as pd
from env import get_db_url

def get_data_from_sql():
    sql = """
    SELECT customer_id, monthly_charges, tenure, total_charges
    FROM customers
    WHERE contract_type_id = 3
    """
    url = get_db_url('telco_churn')
    tc_df = pd.read_sql(sql, url)
    return tc_df

def wrangle_telco():
    tc_df = get_data_from_sql()
    tc_df.total_charges = tc_df.total_charges.str.replace(' ', '0').astype(float)
    tc_df = tc_df.set_index('customer_id')
    return tc_df