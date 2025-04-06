import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from financial_analysis import load_data, analyze_data, generate_response, generate_visualization
from nlp_processor import analyze_query, extract_financial_terms
from advanced_visualizations import (
    create_financial_ratio_chart,
    create_performance_comparison,
    create_forecast_chart,
    create_financial_wordcloud,
    create_interactive_chart
)
import base64
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Financial Analysis Chatbot",
    page_icon="💹",
    layout="wide"
)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'company_comparison' not in st.session_state:
    st.session_state.company_comparison = []

# App title and description
st.title("Financial Analysis Chatbot")

# Custom CSS for improved UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 0.75rem;
    }
    .chat-message.user {
        background-color: #F0F2F6;
    }
    .chat-message.bot {
        background-color: #E3F2FD;
    }
    .chat-message .avatar {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }
    .chat-message .user-avatar {
        background-color: #6C63FF;
        color: white;
    }
    .chat-message .bot-avatar {
        background-color: #FF6584;
        color: white;
    }
    .chat-message .content {
        flex: 1;
    }
    .metric-card {
        border: 1px solid #E0E0E0;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E3F2FD;
    }
</style>
""", unsafe_allow_html=True)

# Description with more details
st.markdown("""
This enhanced financial analysis chatbot provides intelligent insights about company financial performance.
""")

# Main layout with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📊 Data Explorer", "📈 Advanced Analytics", "📤 Data Upload", "📋 Summary"])

# Data loading - do this outside the tabs to avoid reloading
sidebar = st.sidebar
sidebar.title("Settings")
uploaded_file = sidebar.file_uploader("Upload financial data (CSV)", type="csv")

# Data loading
if uploaded_file is not None:
    # Load data from uploaded file
    df = pd.read_csv(uploaded_file, thousands=',')
    sidebar.success("Data uploaded successfully!")
else:
    # Load the provided sample data
    df = load_data()
    sidebar.info("Using sample financial data for Microsoft, Tesla, and Apple.")

# Process the data
analysis_data = analyze_data(df)

# Company selection
companies = df['Company'].unique()
selected_company = sidebar.selectbox("Select a company to analyze:", companies)

# Multi-select for company comparison
compare_companies = sidebar.multiselect(
    "Select companies to compare:", 
    companies,
    default=[selected_company]
)
st.session_state.company_comparison = compare_companies

# Filter data for the selected company
company_data = df[df['Company'] == selected_company]

# Fiscal year range for filtering
years = sorted(df['Fiscal Year'].unique())
year_range = sidebar.slider(
    "Select Year Range:",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=(int(min(years)), int(max(years)))
)

# Filter data based on the year range
filtered_df = df[(df['Fiscal Year'] >= year_range[0]) & (df['Fiscal Year'] <= year_range[1])]
filtered_company_data = company_data[(company_data['Fiscal Year'] >= year_range[0]) & (company_data['Fiscal Year'] <= year_range[1])]

# TAB 1: CHAT INTERFACE
with tab1:
    st.header("Financial Analysis Chat")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div class="avatar user-avatar">👤</div>
                <div class="content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot">
                <div class="avatar bot-avatar">🤖</div>
                <div class="content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input with autocomplete suggestions
    st.subheader("Ask me about financial performance")
    
    # Generate query suggestions based on the selected company
    query_suggestions = [
        f"What is the total revenue of {selected_company}?",
        f"How has the net income of {selected_company} changed over the years?",
        f"What are the assets and liabilities of {selected_company}?",
        f"How has the cash flow of {selected_company} changed?",
        f"Show me the revenue growth of {selected_company}",
        f"Compare {selected_company} with other companies",
        f"What is the forecast for {selected_company}?",
        f"Show me financial ratios for {selected_company}",
        f"What is the profit margin of {selected_company}?",
        f"How efficient is {selected_company} at using its assets?"
    ]
    
    # Select from suggestions or create custom query
    selected_query = st.selectbox(
        "Select a question or type your own:", 
        [""] + query_suggestions,
        key="query_select"
    )
    
    # Text input with auto-complete
    user_query = st.text_input(
        "Or type your question here:", 
        value=selected_query,
        key="query_input"
    )
    
    # Process query when submitted
    if st.button("Ask Question", key="ask_button"):
        if user_query:
            # Add user query to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Generate response
            response = generate_response(user_query, analysis_data, selected_company)
            
            # Add response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Generate visualization based on the query
            fig = generate_visualization(user_query, filtered_company_data, selected_company)
            
            # Display the latest response and visualization
            st.markdown(f"""
            <div class="chat-message bot">
                <div class="avatar bot-avatar">🤖</div>
                <div class="content">{response}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Clear the input field after submission
            st.rerun()
    
    # Option to clear chat history
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

# TAB 2: DATA EXPLORER
with tab2:
    st.header("Financial Data Explorer")
    
    # Display basic financial information for the selected company
    st.subheader(f"{selected_company} - Financial Overview")
    
    latest_year = company_data['Fiscal Year'].max()
    latest_data = company_data[company_data['Fiscal Year'] == latest_year].iloc[0]
    
    # Handle NaN values for growth percentages
    revenue_growth = latest_data.get('Revenue Growth (%)', 0)
    revenue_growth = 0 if pd.isna(revenue_growth) else revenue_growth
    
    net_income_growth = latest_data.get('Net Income Growth (%)', 0)
    net_income_growth = 0 if pd.isna(net_income_growth) else net_income_growth
    
    cash_flow_growth = latest_data.get('Cash Flow Growth (%)', 0)
    cash_flow_growth = 0 if pd.isna(cash_flow_growth) else cash_flow_growth
    
    profit_margin = latest_data.get('Profit Margin (%)', 0)
    profit_margin = 0 if pd.isna(profit_margin) else profit_margin
    
    roa = latest_data.get('ROA (%)', 0)
    roa = 0 if pd.isna(roa) else roa
    
    # Improved metrics with more context
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Revenue", f"${latest_data['Total Revenue (in millions)']:,}M", delta=f"{revenue_growth:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Net Income", f"${latest_data['Net Income (in millions)']:,}M", delta=f"{net_income_growth:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Cash Flow", f"${latest_data['Cash Flow from Operating Activities (in millions)']:,}M", delta=f"{cash_flow_growth:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional financial metrics
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Profit Margin", f"{profit_margin:.1f}%", delta=None)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Return on Assets", f"{roa:.1f}%", delta=None)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col6:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        debt_ratio = latest_data['Total Liabilities (in millions)'] / latest_data['Total Assets (in millions)']
        st.metric("Debt-to-Asset Ratio", f"{debt_ratio:.2f}", delta=None)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Data table with enhanced formatting
    st.subheader("Detailed Financial Data")
    
    # Options for data display
    display_options = st.radio(
        "Choose display type:",
        options=["Table View", "Interactive Chart"],
        horizontal=True
    )
    
    if display_options == "Table View":
        st.dataframe(
            filtered_company_data.sort_values("Fiscal Year", ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        # Interactive chart options
        chart_type = st.selectbox(
            "Select chart type:",
            ["line", "bar", "area", "scatter"],
            key="chart_type"
        )
        
        # Metric selection for the chart
        available_metrics = [col for col in filtered_company_data.columns if '(in millions)' in col or '(%)' in col]
        selected_metrics = st.multiselect(
            "Select metrics to display:",
            available_metrics,
            default=["Total Revenue (in millions)", "Net Income (in millions)"],
            key="selected_metrics"
        )
        
        if selected_metrics:
            # Create and display the interactive chart
            custom_chart = create_interactive_chart(filtered_company_data, chart_type, selected_metrics)
            st.plotly_chart(custom_chart, use_container_width=True)
        else:
            st.warning("Please select at least one metric to display.")

# TAB 3: ADVANCED ANALYTICS
with tab3:
    st.header("Advanced Financial Analytics")
    
    # Subtabs for different analysis types
    analysis_type = st.radio(
        "Choose analysis type:",
        ["Financial Ratios", "Comparative Analysis", "Growth Analysis", "Forecasting", "Word Cloud"],
        horizontal=True
    )
    
    if analysis_type == "Financial Ratios":
        st.subheader(f"Financial Ratios Analysis - {selected_company}")
        
        # Ratio chart
        ratio_chart = create_financial_ratio_chart(filtered_company_data, selected_company)
        st.plotly_chart(ratio_chart, use_container_width=True)
        
        # Explanation of ratios
        with st.expander("Understanding Financial Ratios"):
            st.markdown("""
            **Debt-to-Asset Ratio**: Measures how much of a company's assets are financed by debt. 
            A higher ratio indicates higher leverage and potentially higher risk.
            
            **Return on Assets (ROA)**: Shows how efficiently a company is using its assets to generate profits. 
            A higher ROA indicates better asset utilization.
            
            **Profit Margin**: Indicates how much of each dollar of revenue is kept as profit. 
            A higher profit margin suggests better cost control and pricing power.
            """)
    
    elif analysis_type == "Comparative Analysis":
        st.subheader("Company Comparison")
        
        if len(st.session_state.company_comparison) < 2:
            st.warning("Please select at least two companies for comparison in the sidebar.")
        else:
            # Metric for comparison
            comparison_metric = st.selectbox(
                "Select metric for comparison:",
                ["Revenue", "Net Income", "Assets", "Liabilities", "Cash Flow"],
                key="comparison_metric"
            )
            
            # Mapping to actual column names
            metric_mapping = {
                "Revenue": "revenue",
                "Net Income": "net_income",
                "Assets": "assets",
                "Liabilities": "liabilities",
                "Cash Flow": "cash_flow"
            }
            
            # Create comparison chart
            comparison_chart = create_performance_comparison(
                filtered_df, 
                st.session_state.company_comparison,
                metric_mapping[comparison_metric]
            )
            
            st.plotly_chart(comparison_chart, use_container_width=True)
            
            # Add comparison table
            st.subheader("Comparative Summary (Latest Year)")
            
            # Get the latest year data for each company
            latest_data_per_company = []
            
            for company in st.session_state.company_comparison:
                company_df = filtered_df[filtered_df['Company'] == company]
                if not company_df.empty:
                    max_year = company_df['Fiscal Year'].max()
                    latest_row = company_df[company_df['Fiscal Year'] == max_year].iloc[0]
                    latest_data_per_company.append({
                        'Company': company,
                        'Year': max_year,
                        'Revenue': f"${latest_row['Total Revenue (in millions)']:,}M",
                        'Net Income': f"${latest_row['Net Income (in millions)']:,}M",
                        'Profit Margin': f"{latest_row['Profit Margin (%)']:.1f}%",
                        'ROA': f"{latest_row['ROA (%)']:.1f}%",
                        'Debt Ratio': f"{latest_row['Debt-to-Asset Ratio']:.2f}"
                    })
            
            if latest_data_per_company:
                comparison_df = pd.DataFrame(latest_data_per_company)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    elif analysis_type == "Growth Analysis":
        st.subheader(f"Growth Analysis - {selected_company}")
        
        # Growth metrics selection
        growth_metrics = [col for col in filtered_company_data.columns if 'Growth' in col]
        
        selected_growth_metrics = st.multiselect(
            "Select growth metrics to display:",
            growth_metrics,
            default=growth_metrics[:3] if len(growth_metrics) >= 3 else growth_metrics,
            key="selected_growth_metrics"
        )
        
        if selected_growth_metrics:
            # Create growth chart
            fig = px.line(
                filtered_company_data, 
                x="Fiscal Year", 
                y=selected_growth_metrics,
                title=f"{selected_company} - Growth Metrics Over Time",
                markers=True,
                labels={"value": "Growth (%)", "variable": "Metric"}
            )
            
            fig.update_layout(
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one growth metric to display.")
    
    elif analysis_type == "Forecasting":
        st.subheader(f"Financial Forecasting - {selected_company}")
        
        # Number of periods to forecast
        forecast_periods = st.slider(
            "Number of years to forecast:",
            min_value=1,
            max_value=5,
            value=2
        )
        
        # Create forecast chart
        forecast_chart = create_forecast_chart(filtered_company_data, selected_company, forecast_periods)
        
        if forecast_chart:
            st.plotly_chart(forecast_chart, use_container_width=True)
            
            st.warning("""
            **Note:** This forecast uses simple linear regression based on historical data.
            It should not be used for investment decisions. Real financial forecasting requires
            more sophisticated models and consideration of many external factors.
            """)
        else:
            st.error("Unable to generate forecast. Insufficient historical data.")
    
    elif analysis_type == "Word Cloud":
        st.subheader(f"Financial Term Cloud - {selected_company}")
        
        # Generate word cloud from financial data
        try:
            wordcloud_img = create_financial_wordcloud(filtered_company_data, selected_company)
            
            if wordcloud_img:
                st.image(f"data:image/png;base64,{wordcloud_img}", use_column_width=True)
            else:
                st.info("Unable to generate word cloud. Displaying financial summary instead.")
                
                # Show financial summary as an alternative
                if selected_company in analysis_data:
                    company_info = analysis_data[selected_company]
                    st.write(f"""
                    ### {selected_company} Financial Summary:
                    - Revenue: ${company_info['total_revenue']:,} million
                    - Net Income: ${company_info['net_income']:,} million
                    - Total Assets: ${company_info['total_assets']:,} million
                    - Total Liabilities: ${company_info['total_liabilities']:,} million
                    - Cash Flow: ${company_info['cash_flow']:,} million
                    """)
        except Exception as e:
            st.error(f"Word cloud generation failed. Displaying financial summary instead.")
            # Show financial summary as an alternative
            if selected_company in analysis_data:
                company_info = analysis_data[selected_company]
                st.write(f"""
                ### {selected_company} Financial Summary:
                - Revenue: ${company_info['total_revenue']:,} million
                - Net Income: ${company_info['net_income']:,} million
                - Total Assets: ${company_info['total_assets']:,} million
                - Total Liabilities: ${company_info['total_liabilities']:,} million
                - Cash Flow: ${company_info['cash_flow']:,} million
                """)

# TAB 4: DATA UPLOAD TAB
with tab4:
    st.header("Upload Financial Data")
    
    st.markdown("""
    ### Upload Additional Company Financial Data
    
    Upload a CSV file with financial data for additional companies. The file should have the following columns:
    - Company: Name of the company
    - Fiscal Year: Year of the financial data
    - Total Revenue (in millions): Revenue in millions of dollars
    - Net Income (in millions): Net income in millions of dollars
    - Total Assets (in millions): Assets in millions of dollars
    - Total Liabilities (in millions): Liabilities in millions of dollars
    - Cash Flow from Operating Activities (in millions): Cash flow in millions of dollars
    """)
    
    st.info("""
    **Testing the Upload Feature**
    
    For testing purposes, a sample file with financial data for Amazon, Google, and Netflix is available.
    You can download the sample file from [this link](./sample_financial_data.csv) and then upload it to see how 
    the application processes and integrates new company data.
    """, icon="ℹ️")
    
    # Add a download button for the sample file
    with open('sample_financial_data.csv', 'r') as f:
        csv_data = f.read()
    
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sample_financial_data.csv">Download Sample Financial Data</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read uploaded data
            uploaded_data = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = [
                'Company', 'Fiscal Year', 'Total Revenue (in millions)', 
                'Net Income (in millions)', 'Total Assets (in millions)', 
                'Total Liabilities (in millions)', 'Cash Flow from Operating Activities (in millions)'
            ]
            
            missing_columns = [col for col in required_columns if col not in uploaded_data.columns]
            
            if missing_columns:
                st.error(f"Error: The uploaded file is missing the following required columns: {', '.join(missing_columns)}")
            else:
                # Check if there are any companies that are not in the original dataset
                new_companies = list(set(uploaded_data['Company']) - set(df['Company']))
                existing_companies = list(set(uploaded_data['Company']).intersection(set(df['Company'])))
                
                # Create a copy of the global dataframe to modify
                temp_df = df.copy()
                
                # Add new companies to the dataset (for this session only)
                if new_companies:
                    # Filter just the new companies' data
                    new_company_data = uploaded_data[uploaded_data['Company'].isin(new_companies)]
                    
                    # Merge with current dataframe
                    temp_df = pd.concat([temp_df, new_company_data], ignore_index=True)
                    
                    # Update the filtered_df which is used throughout the app
                    globals()['filtered_df'] = temp_df
                    
                    # Update the list of available companies
                    all_companies = sorted(temp_df['Company'].unique())
                    
                    # Automatically select the newly added companies for comparison
                    for company in new_companies:
                        if company not in st.session_state.company_comparison:
                            st.session_state.company_comparison.append(company)
                    
                    st.success(f"Successfully added data for new companies: {', '.join(new_companies)}")
                
                # Update existing company data if present
                if existing_companies:
                    # For each existing company, check if there are new fiscal years to add
                    for company in existing_companies:
                        existing_data = temp_df[temp_df['Company'] == company]
                        uploaded_company_data = uploaded_data[uploaded_data['Company'] == company]
                        
                        # Get years that are in the uploaded data but not in existing data
                        existing_years = set(existing_data['Fiscal Year'])
                        uploaded_years = set(uploaded_company_data['Fiscal Year'])
                        new_years = uploaded_years - existing_years
                        
                        if new_years:
                            # Add records for new years
                            new_year_data = uploaded_company_data[uploaded_company_data['Fiscal Year'].isin(new_years)]
                            temp_df = pd.concat([temp_df, new_year_data], ignore_index=True)
                            
                            # Update the filtered_df
                            globals()['filtered_df'] = temp_df
                            
                            st.info(f"Added new fiscal years {', '.join(map(str, new_years))} for {company}")
                        
                        # Check for updates to existing years
                        common_years = uploaded_years.intersection(existing_years)
                        if common_years:
                            st.warning(f"The uploaded file contains data for existing years {', '.join(map(str, common_years))} for {company}. Existing data was not overwritten.")
                
                # If no new companies or years were added
                if not new_companies and not any(uploaded_years - existing_years for company in existing_companies 
                                              for existing_years, uploaded_years in [(set(temp_df[temp_df['Company'] == company]['Fiscal Year']), 
                                                                                  set(uploaded_data[uploaded_data['Company'] == company]['Fiscal Year']))]
                                              if company in existing_companies):
                    st.info("The uploaded data already exists in the dataset. No new data was added.")
                
                # Display preview of the updated dataset
                st.subheader("Preview of Updated Dataset")
                st.dataframe(temp_df.sort_values(["Company", "Fiscal Year"], ascending=[True, False]).head(10), use_container_width=True)
                
                # Offer to recalculate metrics with the new data
                if st.button("Recalculate Financial Metrics"):
                    # Update the global dataframe to include the new data
                    globals()['df'] = temp_df
                    
                    # Recalculate all metrics
                    # Add growth percentages
                    for company in df['Company'].unique():
                        company_data = df[df['Company'] == company].sort_values('Fiscal Year')
                        
                        if len(company_data) > 1:
                            # Calculate growth rates for each metric
                            for metric in ['Total Revenue (in millions)', 'Net Income (in millions)', 'Cash Flow from Operating Activities (in millions)']:
                                col_name = f"{metric.split('(')[0].strip()} Growth (%)"
                                df.loc[df['Company'] == company, col_name] = df[df['Company'] == company][metric].pct_change() * 100
                            
                            # Calculate financial ratios
                            df.loc[df['Company'] == company, 'Profit Margin (%)'] = (df[df['Company'] == company]['Net Income (in millions)'] / df[df['Company'] == company]['Total Revenue (in millions)']) * 100
                            df.loc[df['Company'] == company, 'ROA (%)'] = (df[df['Company'] == company]['Net Income (in millions)'] / df[df['Company'] == company]['Total Assets (in millions)']) * 100
                            df.loc[df['Company'] == company, 'Debt-to-Asset Ratio'] = df[df['Company'] == company]['Total Liabilities (in millions)'] / df[df['Company'] == company]['Total Assets (in millions)']
                    
                    # Update the filtered dataframe with the recalculated metrics
                    globals()['filtered_df'] = df
                    
                    # Rerun the app to reflect the changes
                    st.rerun()
        
        except Exception as e:
            st.error(f"Error processing the uploaded file: {str(e)}")

# TAB 5: SUMMARY TAB
with tab5:
    st.header("Financial Summary Report")
    
    # Generate a comprehensive financial summary
    if selected_company in analysis_data:
        company_info = analysis_data[selected_company]
        
        st.subheader(f"Executive Summary for {selected_company}")
        
        # Financial highlights
        st.markdown(f"""
        ### Financial Highlights for Fiscal Year {company_info['latest_year']}
        
        **Revenue Performance:**  
        Total Revenue: ${company_info['total_revenue']:,} million
        {f"Year-over-Year Growth: {company_info['revenue_pct']:.1f}%" if 'revenue_pct' in company_info else ""}
        
        **Profitability:**  
        Net Income: ${company_info['net_income']:,} million
        {f"Year-over-Year Growth: {company_info['net_income_pct']:.1f}%" if 'net_income_pct' in company_info else ""}
        Profit Margin: {company_info['profit_margin']:.1f}%
        
        **Financial Position:**  
        Total Assets: ${company_info['total_assets']:,} million
        Total Liabilities: ${company_info['total_liabilities']:,} million
        Debt-to-Asset Ratio: {company_info['debt_ratio']:.2f}
        
        **Cash Flow:**  
        Operating Cash Flow: ${company_info['cash_flow']:,} million
        {f"Year-over-Year Growth: {company_info['cash_flow_pct']:.1f}%" if 'cash_flow_pct' in company_info else ""}
        
        **Efficiency:**  
        Return on Assets: {company_info['roa']:.1f}%
        """)
        
        # Key trends
        if 'yearly_data' in company_info:
            yearly_records = company_info['yearly_data']
            if len(yearly_records) > 1:
                # Sort by year
                yearly_records = sorted(yearly_records, key=lambda x: x['Fiscal Year'])
                
                # Create a simple trend visualization
                years = [record['Fiscal Year'] for record in yearly_records]
                revenues = [record['Total Revenue (in millions)'] for record in yearly_records]
                net_incomes = [record['Net Income (in millions)'] for record in yearly_records]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=revenues,
                    mode='lines+markers',
                    name='Revenue',
                    line=dict(color='#1E88E5', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=net_incomes,
                    mode='lines+markers',
                    name='Net Income',
                    line=dict(color='#43A047', width=3)
                ))
                
                fig.update_layout(
                    title=f"{selected_company} - Key Financial Trends",
                    xaxis_title="Fiscal Year",
                    yaxis_title="Amount (in millions $)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.subheader("Export Options")
    
    export_format = st.radio(
        "Select export format:",
        ["CSV", "Excel", "JSON"],
        horizontal=True
    )
    
    if st.button("Export Data"):
        company_filtered_data = filtered_df[filtered_df['Company'] == selected_company]
        
        if export_format == "CSV":
            csv_data = company_filtered_data.to_csv(index=False)
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{selected_company}_financial_data.csv">Download CSV file</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif export_format == "Excel":
            # For Excel, we'd normally create a file, but here we'll just provide info
            st.info("Excel export would be available in a production environment.")
            
        elif export_format == "JSON":
            json_data = company_filtered_data.to_json(orient="records")
            b64 = base64.b64encode(json_data.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="{selected_company}_financial_data.json">Download JSON file</a>'
            st.markdown(href, unsafe_allow_html=True)

# Footer with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.sidebar.markdown("---")
st.sidebar.markdown(f"Financial Analysis Chatbot<br>Last updated: {timestamp}", unsafe_allow_html=True)
