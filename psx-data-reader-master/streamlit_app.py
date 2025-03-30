import streamlit as st
import datetime
import sys
import os
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from psx import stocks, tickers
    
    # Streamlit app title
    st.title("PSX Data Reader")
    st.markdown("## Pakistan Stock Exchange Data Viewer")
    
    # Get all available tickers
    st.write("Fetching available tickers...")
    all_tickers = tickers()
    st.write(f"Total number of companies: {len(all_tickers)}")

    # Get data for a specific company (e.g., HBL - Habib Bank Limited)
    st.write("\nFetching data for HBL...")
    data = stocks("HBL", start=datetime.date(2023, 1, 1), end=datetime.date.today())

    # Display the first few rows of the data
    st.write("\nFirst few rows of HBL stock data:")
    st.dataframe(data.head())

    # Plotting the data
    st.line_chart(data[['Open', 'High', 'Low', 'Close']])

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("\nThis might be because:")
    st.write("1. The PSX website might be temporarily unavailable")
    st.write("2. The package might need updates to work with the current PSX website")
    st.write("3. There might be network connectivity issues") 