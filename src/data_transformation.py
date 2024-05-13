import pandas as pd
import numpy as np
from datetime import datetime
from data_preprocessing import preprocess_data

def transform_data(df):
    """
    Applies transformations to the preprocessed DataFrame
    Args:
        df: The preprocessed DataFrame from data_preprocessing.py
    Returns:
        The transformed DataFrame
    """
    # We'll begin by copying the dataframe to avoid modifying the original data
    df = df.copy()

    # Define function to convert timedelta to months
    def convert_months(reference_date, df, column):
        time_delta = reference_date - df[column]
        months = (time_delta.dt.days / 30).astype(int)
        #months = (time_delta / np.timedelta64(1, 'M')).astype(int)
        return months

    # Create reference date
    reference_date = datetime(2016, 1, 1)

    # Create columns
    df['months_activ'] = convert_months(reference_date, df, 'date_activ')
    df['months_to_end'] = -convert_months(reference_date, df, 'date_end')
    df['months_modif_prod'] = convert_months(reference_date, df, 'date_modif_prod')
    df['months_renewal'] = convert_months(reference_date, df, 'date_renewal')

    # Drop datetime columns
    remove = [
        'date_activ',
        'date_end',
        'date_modif_prod',
        'date_renewal'
    ]
    df = df.drop(columns=remove)

    # Replace 't' and 'f' with 1 and 0 in 'has_gas' column
    df['has_gas'] = df['has_gas'].replace(['t', 'f'], [1, 0])

    # Convert 'channel_sales' and 'origin_up' into categorical type
    df['channel_sales'] = df['channel_sales'].astype('category')
    df['origin_up'] = df['origin_up'].astype('category')

    # One-hot encoding for 'channel_sales' and 'origin_up' columns
    df = pd.get_dummies(df, columns=['channel_sales'], prefix='channel')
    df = df.drop(columns=['channel_sddiedcslfslkckwlfkdpoeeailfpeds', 'channel_epumfxlbckeskwekxbiuasklxalciiuu'])
    df = pd.get_dummies(df, columns=['origin_up'], prefix='origin_up')
    df = df.drop(columns=['origin_up_MISSING', 'origin_up_usapbepcfoloekilkwsdiboslwaxobdp', 'origin_up_ewxeelcelemmiwuafmddpobolfuxioce'])

    # Apply log10 transformation to skewed numerical columns
    skewed = [
        'cons_12m', 
        'cons_gas_12m', 
        'cons_last_month',
        'forecast_cons_12m', 
        'forecast_cons_year', 
        'forecast_discount_energy',
        'forecast_meter_rent_12m', 
        'forecast_price_energy_off_peak',
        'forecast_price_energy_peak', 
        'forecast_price_pow_off_peak'
    ]
    df[skewed] = np.log10(df[skewed] + 1)

    return df

if __name__ == "__main__":
    df = preprocess_data()
    transformed_df = transform_data(df)
    print(transformed_df.head())
