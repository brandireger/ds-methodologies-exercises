import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_variable_pairs(df):
    sns.pairplot(df, kind="reg", plot_kws={'line_kws':{'color':'red'}, 'scatter_kws':{'alpha':0.5}})

def months_to_years(tenure_months, df):
    new_df = df.copy()
    new_df['tenure_years'] = (tenure_months / 12).round(decimals=0).astype(int)
    return new_df

def plot_categorical_and_continuous_vars(categorical_var, continuous_var, df):
    """
    Takes:
          string categorical_var, string continuous_var, df
    Returns:
          three plots of categorical var with continuous var
    """

    sns.catplot(x=categorical_var, y=continuous_var, data=df, kind = "boxen", palette='bright')
    plt.xlabel(categorical_var, fontsize=12)
    plt.ylabel(continuous_var, fontsize=12)
    plt.show()
    
    sns.catplot(x=categorical_var, y=continuous_var, data=df, kind="swarm", palette='bright')
    plt.xlabel(categorical_var, fontsize=12)
    plt.ylabel(continuous_var, fontsize=12)
    plt.show()
    
    sns.catplot(x=categorical_var, y=continuous_var, data=df, kind="bar", palette='bright')
    plt.xlabel(categorical_var, fontsize=12)
    plt.ylabel(continuous_var, fontsize=12)
    plt.show()
