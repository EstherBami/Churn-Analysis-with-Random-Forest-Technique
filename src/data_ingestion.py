import pandas as pd

# Define function to read csv files
def load_client_data():
    """ Reads client and price data from CSV files
    Returns: The loaded client dataframe
    """
    df = pd.read_csv('data\client_data.csv')
    return df

def load_price_data():
    price_df = pd.read_csv('data\price_data.csv')
    return price_df

# Load the client and price datasets
df = load_client_data()
price_df = load_price_data()

# Explore data
print(df.head())
print(price_df.head())
