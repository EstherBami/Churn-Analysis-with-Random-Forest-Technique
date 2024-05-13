import pandas as pd
import numpy as np
from data_ingestion import load_client_data, load_price_data 

def preprocess_data():
    """
    Loads, preprocesses, and combines client and price data
    Returns: The preprocessed DataFrame
    """
    # Load data
    df = load_client_data()
    price_df = load_price_data()

    # Convert all dates to datetime format
    date_cols = ["date_activ", "date_end", "date_modif_prod", "date_renewal"]
    df[date_cols] = df[date_cols].apply(pd.to_datetime, format='%Y-%m-%d')

    # FEATURE ENGINEERING
    # Create new features from the price_df to increase the predictive power of the features.
    # (1) Calculate the difference between off-peak prices in December and preceding January
    monthly_price_by_id = price_df.groupby(['id', 'price_date']).agg({
        'price_off_peak_var': 'mean', 
        'price_off_peak_fix': 'mean'
    }).reset_index()

    jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
    dec_prices = monthly_price_by_id.groupby('id').last().reset_index()

    diff = pd.merge(dec_prices.rename(columns={'price_off_peak_var': 'dec_1', 'price_off_peak_fix': 'dec_2'}), 
                    jan_prices.drop(columns='price_date'), 
                    on='id')

    diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
    diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']
    diff = diff[['id', 'offpeak_diff_dec_january_energy', 'offpeak_diff_dec_january_power']]

    df = pd.merge(df, diff, on='id')

    # Calculate the average price changes across individual periods
    mean_prices = price_df.groupby(['id']).agg({
        'price_off_peak_var': 'mean', 
        'price_peak_var': 'mean', 
        'price_mid_peak_var': 'mean',
        'price_off_peak_fix': 'mean',
        'price_peak_fix': 'mean',
        'price_mid_peak_fix': 'mean'    
    }).reset_index()

    mean_prices['off_peak_peak_var_mean_diff'] = mean_prices['price_off_peak_var'] - mean_prices['price_peak_var']
    mean_prices['peak_mid_peak_var_mean_diff'] = mean_prices['price_peak_var'] - mean_prices['price_mid_peak_var']
    mean_prices['off_peak_mid_peak_var_mean_diff'] = mean_prices['price_off_peak_var'] - mean_prices['price_mid_peak_var']
    mean_prices['off_peak_peak_fix_mean_diff'] = mean_prices['price_off_peak_fix'] - mean_prices['price_peak_fix']
    mean_prices['peak_mid_peak_fix_mean_diff'] = mean_prices['price_peak_fix'] - mean_prices['price_mid_peak_fix']
    mean_prices['off_peak_mid_peak_fix_mean_diff'] = mean_prices['price_off_peak_fix'] - mean_prices['price_mid_peak_fix']

    columns = [
        'id', 
        'off_peak_peak_var_mean_diff',
        'peak_mid_peak_var_mean_diff', 
        'off_peak_mid_peak_var_mean_diff',
        'off_peak_peak_fix_mean_diff', 
        'peak_mid_peak_fix_mean_diff', 
        'off_peak_mid_peak_fix_mean_diff'
    ]
    df = pd.merge(df, mean_prices[columns], on='id')

    # Calculate the maximum price changes across periods and months
    mean_prices_by_month = price_df.groupby(['id', 'price_date']).agg({
        'price_off_peak_var': 'mean', 
        'price_peak_var': 'mean', 
        'price_mid_peak_var': 'mean',
        'price_off_peak_fix': 'mean',
        'price_peak_fix': 'mean',
        'price_mid_peak_fix': 'mean'    
    }).reset_index()

    mean_prices_by_month['off_peak_peak_var_mean_diff'] = mean_prices_by_month['price_off_peak_var'] - mean_prices_by_month['price_peak_var']
    mean_prices_by_month['peak_mid_peak_var_mean_diff'] = mean_prices_by_month['price_peak_var'] - mean_prices_by_month['price_mid_peak_var']
    mean_prices_by_month['off_peak_mid_peak_var_mean_diff'] = mean_prices_by_month['price_off_peak_var'] - mean_prices_by_month['price_mid_peak_var']
    mean_prices_by_month['off_peak_peak_fix_mean_diff'] = mean_prices_by_month['price_off_peak_fix'] - mean_prices_by_month['price_peak_fix']
    mean_prices_by_month['peak_mid_peak_fix_mean_diff'] = mean_prices_by_month['price_peak_fix'] - mean_prices_by_month['price_mid_peak_fix']
    mean_prices_by_month['off_peak_mid_peak_fix_mean_diff'] = mean_prices_by_month['price_off_peak_fix'] - mean_prices_by_month['price_mid_peak_fix']

    max_diff_across_periods_months = mean_prices_by_month.groupby(['id']).agg({
        'off_peak_peak_var_mean_diff': 'max',
        'peak_mid_peak_var_mean_diff': 'max',
        'off_peak_mid_peak_var_mean_diff': 'max',
        'off_peak_peak_fix_mean_diff': 'max',
        'peak_mid_peak_fix_mean_diff': 'max',
        'off_peak_mid_peak_fix_mean_diff': 'max'
    }).reset_index().rename(
        columns={
            'off_peak_peak_var_mean_diff': 'off_peak_peak_var_max_monthly_diff',
            'peak_mid_peak_var_mean_diff': 'peak_mid_peak_var_max_monthly_diff',
            'off_peak_mid_peak_var_mean_diff': 'off_peak_mid_peak_var_max_monthly_diff',
            'off_peak_peak_fix_mean_diff': 'off_peak_peak_fix_max_monthly_diff',
            'peak_mid_peak_fix_mean_diff': 'peak_mid_peak_fix_max_monthly_diff',
            'off_peak_mid_peak_fix_mean_diff': 'off_peak_mid_peak_fix_max_monthly_diff'
        }
    )

    columns = [
        'id',
        'off_peak_peak_var_max_monthly_diff',
        'peak_mid_peak_var_max_monthly_diff',
        'off_peak_mid_peak_var_max_monthly_diff',
        'off_peak_peak_fix_max_monthly_diff',
        'peak_mid_peak_fix_max_monthly_diff',
        'off_peak_mid_peak_fix_max_monthly_diff'
    ]

    df = pd.merge(df, max_diff_across_periods_months[columns], on='id')

    # Calculate tenure: How long a company has been a client
    df['tenure'] = ((df['date_end'] - df['date_activ']).dt.days / 365).astype(int)
    tenure_churn = df.groupby(['tenure']).agg({'churn': 'mean'}).sort_values(by='churn', ascending=False)
    return df

if __name__ == "__main__":
    df = preprocess_data()
    print(df.head())
