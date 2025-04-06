import streamlit as st
import pandas as pd
import plotly.express as px
import re
import numpy as np
import base64
import io
from financial_analysis import (
    load_data, 
    analyze_data, 
    generate_response,
    get_yfinance_data
)
from advanced_visualizations import (
    create_financial_ratio_chart,
    create_performance_comparison,
    create_forecast_chart,
    create_financial_wordcloud,
    create_interactive_chart,
    create_revenue_chart,
    create_net_income_chart,
    create_growth_chart
)

# Set page configuration
st.set_page_config(
    page_title="Financial Analysis Chatbot",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom icon
def add_logo():
    st.markdown(
        """
        <style>
        .logo-img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 80px;
            height: 80px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with open("assets/financial_icon.svg", "r") as f:
        svg = f.read()
        b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        html = f'<img src="data:image/svg+xml;base64,{b64}" class="logo-img">'
        st.sidebar.markdown(html, unsafe_allow_html=True)

try:
    add_logo()
except:
    pass  # Continue if logo file not found

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Upload", "Company Analysis", "Chatbot", "Financial Visualization", "Real-time Data"])

# Home page
if page == "Home":
    st.title("Financial Analysis Chatbot")
    st.subheader("An interactive tool for financial data analysis")
    
    st.markdown("""
    Welcome to the Financial Analysis Chatbot! This tool helps you analyze financial data
    of companies through natural language queries and interactive visualizations.
    
    ### Features:
    - Upload your own financial data CSV file
    - Analyze company financial metrics
    - Ask questions about financial performance
    - Visualize financial trends and comparisons
    - Get real-time financial data from Yahoo Finance
    
    ### How to use:
    1. Start by uploading your financial data in the **Data Upload** tab
    2. Explore individual company metrics in the **Company Analysis** tab
    3. Ask questions about the financial data in the **Chatbot** tab
    4. Create custom visualizations in the **Financial Visualization** tab
    5. Get the latest financial data in the **Real-time Data** tab
    
    ### Sample queries you can ask:
    - What is Apple's revenue?
    - How has Microsoft's net income changed?
    - Show me Tesla's financial performance
    - Compare Microsoft, Apple, and Tesla
    - What's the forecast for Google?
    """)
    
    # Display a sample visualization if data is loaded
    if st.session_state.df is not None:
        st.subheader("Sample Visualization")
        fig = create_revenue_chart(st.session_state.df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload data in the 'Data Upload' tab to enable visualizations.")

# Data Upload page
elif page == "Data Upload":
    st.title("Data Upload")
    st.write("Upload your financial data CSV file or use the default sample data")
    
    upload_option = st.radio(
        "Choose data source:",
        ["Upload CSV file", "Use sample data"]
    )
    
    if upload_option == "Upload CSV file":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file, thousands=',')
                
                # Verify that the CSV has the required columns
                required_columns = [
                    'Company', 'Fiscal Year', 'Total Revenue (in millions)', 
                    'Net Income (in millions)', 'Total Assets (in millions)',
                    'Total Liabilities (in millions)', 'Cash Flow from Operating Activities (in millions)'
                ]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"The uploaded CSV is missing the following required columns: {', '.join(missing_columns)}")
                    st.info("Please ensure your CSV has the following columns: " + ", ".join(required_columns))
                else:
                    # Store the dataframe and analyze it
                    st.session_state.df = df
                    st.session_state.analysis_data = analyze_data(df)
                    
                    st.success("Data uploaded and analyzed successfully!")
                    
                    # Show a preview of the data
                    st.subheader("Data Preview")
                    st.dataframe(df.head())
                    
                    # Show some basic statistics
                    st.subheader("Data Summary")
                    st.write(f"Number of companies: {df['Company'].nunique()}")
                    st.write(f"Years covered: {df['Fiscal Year'].min()} to {df['Fiscal Year'].max()}")
                    st.write(f"Total number of records: {len(df)}")
                    
            except Exception as e:
                st.error(f"Error processing the uploaded file: {str(e)}")
                st.info("Please ensure your CSV file is properly formatted.")
        
    else:  # Use sample data
        if st.button("Load Sample Data"):
            # Load sample data
            df = load_data()
            
            # Store the dataframe and analyze it
            st.session_state.df = df
            st.session_state.analysis_data = analyze_data(df)
            
            st.success("Sample data loaded and analyzed successfully!")
            
            # Show a preview of the data
            st.subheader("Data Preview")
            st.dataframe(df.head())

# Company Analysis page
elif page == "Company Analysis":
    st.title("Company Analysis")
    
    if st.session_state.df is None:
        st.warning("No data available. Please upload data in the 'Data Upload' tab.")
    else:
        # Get list of companies
        companies = st.session_state.df['Company'].unique()
        
        # Company selector
        selected_company = st.selectbox("Select a company to analyze:", companies)
        
        # Filter data for selected company
        company_data = st.session_state.df[st.session_state.df['Company'] == selected_company]
        
        # Key financial metrics
        st.subheader("Key Financial Metrics")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        # Get the latest year data
        latest_year = company_data['Fiscal Year'].max()
        latest_data = company_data[company_data['Fiscal Year'] == latest_year].iloc[0]
        
        # Display metrics in columns
        with col1:
            st.metric(
                label="Revenue (millions)",
                value=f"${latest_data['Total Revenue (in millions)']:,.0f}",
                delta=f"{latest_data.get('Revenue Growth (%)', 0):.1f}%" 
                if 'Revenue Growth (%)' in latest_data and not pd.isna(latest_data['Revenue Growth (%)']) else None
            )
            
            st.metric(
                label="Total Assets (millions)",
                value=f"${latest_data['Total Assets (in millions)']:,.0f}",
                delta=f"{latest_data.get('Asset Growth (%)', 0):.1f}%" 
                if 'Asset Growth (%)' in latest_data and not pd.isna(latest_data['Asset Growth (%)']) else None
            )
        
        with col2:
            st.metric(
                label="Net Income (millions)",
                value=f"${latest_data['Net Income (in millions)']:,.0f}",
                delta=f"{latest_data.get('Net Income Growth (%)', 0):.1f}%" 
                if 'Net Income Growth (%)' in latest_data and not pd.isna(latest_data['Net Income Growth (%)']) else None
            )
            
            st.metric(
                label="Total Liabilities (millions)",
                value=f"${latest_data['Total Liabilities (in millions)']:,.0f}",
                delta=f"{latest_data.get('Liability Growth (%)', 0):.1f}%" 
                if 'Liability Growth (%)' in latest_data and not pd.isna(latest_data['Liability Growth (%)']) else None
            )
        
        with col3:
            st.metric(
                label="Cash Flow (millions)",
                value=f"${latest_data['Cash Flow from Operating Activities (in millions)']:,.0f}",
                delta=f"{latest_data.get('Cash Flow Growth (%)', 0):.1f}%" 
                if 'Cash Flow Growth (%)' in latest_data and not pd.isna(latest_data['Cash Flow Growth (%)']) else None
            )
            
            # ROA (Return on Assets)
            roa = (latest_data['Net Income (in millions)'] / latest_data['Total Assets (in millions)']) * 100
            st.metric(
                label="Return on Assets (%)",
                value=f"{roa:.1f}%"
            )
        
        # Revenue and Net Income Trends
        st.subheader("Revenue and Net Income Trends")
        
        # Plot the trends
        fig = create_interactive_chart(st.session_state.df, selected_company)
        st.plotly_chart(fig, use_container_width=True)
        
        # Financial Ratios
        st.subheader("Financial Ratios")
        
        # Create financial ratio chart
        fig = create_financial_ratio_chart(st.session_state.df, selected_company)
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth Analysis
        st.subheader("Growth Analysis")
        
        fig = create_growth_chart(st.session_state.df, selected_company)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data to display growth analysis.")
        
        # Forecast
        st.subheader("Simple Forecast (Linear Projection)")
        
        # Create forecast chart
        fig = create_forecast_chart(st.session_state.df, selected_company, 'Total Revenue (in millions)')
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Note: This is a simple linear projection based on historical data and should not be considered a comprehensive forecast.")
        else:
            st.info("Insufficient data to create a forecast.")

# Chatbot page
elif page == "Chatbot":
    st.title("Financial Analysis Chatbot")
    
    if st.session_state.df is None or st.session_state.analysis_data is None:
        st.warning("No data available. Please upload data in the 'Data Upload' tab.")
    else:
        # Get list of companies
        companies = st.session_state.df['Company'].unique()
        
        # Company selector
        selected_company = st.selectbox("Select a company to query:", companies)
        
        # Chat interface
        st.subheader(f"Ask questions about {selected_company}")
        
        # Suggested queries
        with st.expander("Suggested Queries"):
            st.write("""
            Here are some example queries you can ask:
            - What is the revenue of {company}?
            - How has {company}'s net income changed?
            - Tell me about {company}'s assets and liabilities
            - What is {company}'s cash flow?
            - How fast is {company} growing?
            - Give me an overview of {company}'s performance
            - Compare {company} with other companies
            - What is the trend in {company}'s revenue?
            - What's the forecast for {company}?
            """.format(company=selected_company))
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Bot:** {message['content']}")
        
        # Query input
        user_query = st.text_input("Your question:", key="user_query")
        
        if st.button("Ask") and user_query:
            # Add user query to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Generate response
            response = generate_response(user_query, st.session_state.analysis_data, selected_company)
            
            # Add bot response to chat history
            st.session_state.chat_history.append({"role": "bot", "content": response})
            
            # Rerun to display updated chat history
            st.rerun()

# Financial Visualization page
elif page == "Financial Visualization":
    st.title("Financial Visualization")
    
    if st.session_state.df is None:
        st.warning("No data available. Please upload data in the 'Data Upload' tab.")
    else:
        # Get list of companies
        companies = st.session_state.df['Company'].unique().tolist()
        
        # Select visualization type
        viz_type = st.selectbox(
            "Select visualization type:",
            ["Revenue Over Time", "Net Income Over Time", "Company Comparison", "Financial Ratios", "Growth Analysis", "Forecast"]
        )
        
        if viz_type in ["Revenue Over Time", "Net Income Over Time"]:
            # Multi-select for companies
            selected_companies = st.multiselect(
                "Select companies:",
                companies,
                default=companies[:min(3, len(companies))]
            )
            
            if selected_companies:
                if viz_type == "Revenue Over Time":
                    fig = create_revenue_chart(st.session_state.df, selected_companies)
                else:  # Net Income Over Time
                    fig = create_net_income_chart(st.session_state.df, selected_companies)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least one company to visualize.")
        
        elif viz_type == "Company Comparison":
            # Metric selection
            metric = st.selectbox(
                "Select metric to compare:",
                [
                    "Total Revenue (in millions)",
                    "Net Income (in millions)",
                    "Total Assets (in millions)",
                    "Total Liabilities (in millions)",
                    "Cash Flow from Operating Activities (in millions)",
                    "ROA (%)",
                    "Profit Margin (%)",
                    "Debt-to-Asset Ratio"
                ]
            )
            
            # Multi-select for companies
            selected_companies = st.multiselect(
                "Select companies to compare:",
                companies,
                default=companies[:min(3, len(companies))]
            )
            
            if len(selected_companies) >= 2:
                fig = create_performance_comparison(st.session_state.df, selected_companies, metric)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least two companies to compare.")
        
        elif viz_type == "Financial Ratios":
            # Company selector
            selected_company = st.selectbox("Select a company:", companies)
            
            fig = create_financial_ratio_chart(st.session_state.df, selected_company)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Growth Analysis":
            # Company selector
            selected_company = st.selectbox("Select a company:", companies)
            
            fig = create_growth_chart(st.session_state.df, selected_company)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data to display growth analysis.")
        
        elif viz_type == "Forecast":
            # Company selector
            selected_company = st.selectbox("Select a company:", companies)
            
            # Metric selection
            metric = st.selectbox(
                "Select metric to forecast:",
                [
                    "Total Revenue (in millions)",
                    "Net Income (in millions)",
                    "Cash Flow from Operating Activities (in millions)"
                ]
            )
            
            # Forecast periods
            periods = st.slider("Forecast periods (years):", 1, 5, 2)
            
            fig = create_forecast_chart(st.session_state.df, selected_company, metric, periods)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Note: This is a simple linear projection based on historical data and should not be considered a comprehensive forecast.")
            else:
                st.info("Insufficient data to create a forecast.")

# Real-time Data page
elif page == "Real-time Data":
    st.title("Real-time Financial Data")
    st.write("Get the latest financial data from Yahoo Finance")
    
    # Input for tickers
    ticker_input = st.text_input(
        "Enter stock tickers (comma-separated):",
        "MSFT,AAPL,TSLA,GOOGL,AMZN"
    )
    
    # Parse tickers
    tickers = [ticker.strip() for ticker in ticker_input.split(",") if ticker.strip()]
    
    if st.button("Fetch Data"):
        if tickers:
            with st.spinner("Fetching financial data from Yahoo Finance..."):
                # Get yfinance data
                yf_data = get_yfinance_data(tickers)
                
                if yf_data is not None and not yf_data.empty:
                    st.success(f"Successfully retrieved data for {yf_data['Company'].nunique()} companies.")
                    
                    # Store the dataframe and analyze it
                    st.session_state.df = yf_data
                    st.session_state.analysis_data = analyze_data(yf_data)
                    
                    # Show a preview of the data
                    st.subheader("Data Preview")
                    st.dataframe(yf_data)
                    
                    # Show a visualization
                    st.subheader("Revenue Comparison")
                    fig = create_revenue_chart(yf_data)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not retrieve financial data. Please check the ticker symbols or try again later.")
                    st.info("If the data retrieval fails, you can still use the uploaded data from the 'Data Upload' tab.")
        else:
            st.warning("Please enter at least one ticker symbol.")

# Add footer
st.markdown("---")
st.markdown("Financial Analysis Chatbot | Built with Streamlit")
