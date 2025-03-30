import datetime
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from psx import stocks, tickers
    
    # Get all available tickers
    print("Fetching available tickers...")
    all_tickers = tickers()
    print(f"Total number of companies: {len(all_tickers)}")

    # Get data for a specific company (e.g., HBL - Habib Bank Limited)
    print("\nFetching data for HBL...")
    data = stocks("HBL", start=datetime.date(2023, 1, 1), end=datetime.date.today())

    # Display the first few rows of the data
    print("\nFirst few rows of HBL stock data:")
    print(data.head())
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("\nThis might be because:")
    print("1. The PSX website might be temporarily unavailable")
    print("2. The package might need updates to work with the current PSX website")
    print("3. There might be network connectivity issues") 