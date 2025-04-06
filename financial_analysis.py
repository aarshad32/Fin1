import pandas as pd
import plotly.express as px
import re
import numpy as np

# Try to import yfinance safely
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception as e:
    print(f"Warning: yfinance could not be imported: {str(e)}")
    YFINANCE_AVAILABLE = False

# Import custom modules
from nlp_processor import analyze_query
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

def load_data(uploaded_file=None):
    """Load financial data from an uploaded file or use default data"""
    try:
        if uploaded_file is not None:
            # Try to load the file from the uploaded file
            df = pd.read_csv(uploaded_file, thousands=',')
        else:
            # Try to load from a local file
            try:
                # Check multiple possible locations for the file
                possible_paths = [
                    './attached_assets/Financial_data.csv',
                    './Financial_data.csv',
                    './sample_financial_data.csv'
                ]
                
                for path in possible_paths:
                    try:
                        df = pd.read_csv(path, thousands=',')
                        print(f"Successfully loaded data from {path}")
                        break
                    except:
                        continue
                else:  # This runs if the loop completes without breaking
                    raise FileNotFoundError("Could not find financial data file in any expected location")
            except Exception as e:
                print(f"Error loading financial data: {str(e)}")
                # Fall back to sample data
                raise  # Re-raise to trigger the sample data creation
    except Exception as e:
        print(f"Using sample data due to: {str(e)}")
        # If file not found or error, create a DataFrame with the provided sample data
        data = {
            'Company': ['Microsoft', 'Microsoft', 'Microsoft', 'Tesla', 'Tesla', 'Tesla', 'Apple', 'Apple', 'Apple'],
            'Fiscal Year': [2024, 2023, 2022, 2024, 2023, 2022, 2024, 2023, 2022],
            'Total Revenue (in millions)': [245122, 211915, 198270, 97690, 96773, 81462, 391035, 383285, 394328],
            'Net Income (in millions)': [88136, 72361, 72738, 7153, 14974, 12587, 93736, 96995, 99803],
            'Total Assets (in millions)': [512163, 411976, 364840, 122070, 106618, 82338, 364980, 352583, 352755],
            'Total Liabilities (in millions)': [243686, 205753, 198298, 48390, 43009, 36440, 308030, 290437, 302083],
            'Cash Flow from Operating Activities (in millions)': [118548, 87582, 89035, 14923, 13256, 14724, 29943, 30737, 35929]
        }
        df = pd.DataFrame(data)
        print("Created sample financial data")
    
    return df

def get_yfinance_data(tickers, period="2y"):
    """Get financial data from yfinance API"""
    # Check if yfinance is available
    if not YFINANCE_AVAILABLE:
        print("yfinance module is not available, using sample data instead")
        # Return None to trigger using sample data
        return None
        
    try:
        # Dictionary to store financial data
        financial_data = {
            'Company': [],
            'Fiscal Year': [],
            'Total Revenue (in millions)': [],
            'Net Income (in millions)': [],
            'Total Assets (in millions)': [],
            'Total Liabilities (in millions)': [],
            'Cash Flow from Operating Activities (in millions)': []
        }
        
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            
            # Safely get financials
            try:
                income_stmt = stock.income_stmt
                balance_sheet = stock.balance_sheet
                cash_flow = stock.cashflow
            except Exception as e:
                print(f"Error getting financial data for {ticker}: {str(e)}")
                continue
            
            if income_stmt is None or income_stmt.empty:
                continue
                
            # Process each year in the data
            for year in income_stmt.columns:
                try:
                    fiscal_year = year.year
                    
                    # Total Revenue (may be listed under different names)
                    revenue = None
                    try:
                        revenue = income_stmt.loc.get('Total Revenue', {}).get(year, None)
                        if revenue is None:
                            revenue = income_stmt.loc.get('Revenue', {}).get(year, None)
                    except:
                        pass
                        
                    # Net Income
                    net_income = None
                    try:
                        net_income = income_stmt.loc.get('Net Income', {}).get(year, None)
                    except:
                        pass
                    
                    # Skip if essential data is missing
                    if revenue is None or net_income is None:
                        continue
                        
                    # Total Assets and Liabilities (if available)
                    total_assets = None
                    total_liabilities = None
                    try:
                        if balance_sheet is not None:
                            total_assets = balance_sheet.loc.get('Total Assets', {}).get(year, None)
                            total_liabilities = balance_sheet.loc.get('Total Liabilities', {}).get(year, None)
                    except:
                        pass
                    
                    # Cash Flow from Operations (if available)
                    operating_cash_flow = None
                    try:
                        if cash_flow is not None:
                            operating_cash_flow = cash_flow.loc.get('Operating Cash Flow', {}).get(year, None)
                    except:
                        pass
                    
                    # Append data to lists
                    financial_data['Company'].append(ticker)
                    financial_data['Fiscal Year'].append(fiscal_year)
                    financial_data['Total Revenue (in millions)'].append(float(revenue) / 1e6 if revenue is not None else None)
                    financial_data['Net Income (in millions)'].append(float(net_income) / 1e6 if net_income is not None else None)
                    financial_data['Total Assets (in millions)'].append(float(total_assets) / 1e6 if total_assets is not None else None)
                    financial_data['Total Liabilities (in millions)'].append(float(total_liabilities) / 1e6 if total_liabilities is not None else None)
                    financial_data['Cash Flow from Operating Activities (in millions)'].append(float(operating_cash_flow) / 1e6 if operating_cash_flow is not None else None)
                except Exception as e:
                    print(f"Error processing year data for {ticker}: {str(e)}")
                    continue
                
        # Create DataFrame
        yf_df = pd.DataFrame(financial_data)
        
        # Drop rows with missing revenue or income
        yf_df = yf_df.dropna(subset=['Total Revenue (in millions)', 'Net Income (in millions)'])
        
        if not yf_df.empty:
            return yf_df
        else:
            print("No valid financial data retrieved from yfinance")
            return None
            
    except Exception as e:
        print(f"Error in yfinance data retrieval: {str(e)}")
        return None

def analyze_data(df):
    """Process and analyze the financial data"""
    # Calculate growth rates
    df['Revenue Growth (%)'] = df.groupby(['Company'])['Total Revenue (in millions)'].pct_change() * 100
    df['Net Income Growth (%)'] = df.groupby(['Company'])['Net Income (in millions)'].pct_change() * 100
    df['Asset Growth (%)'] = df.groupby(['Company'])['Total Assets (in millions)'].pct_change() * 100
    df['Liability Growth (%)'] = df.groupby(['Company'])['Total Liabilities (in millions)'].pct_change() * 100
    df['Cash Flow Growth (%)'] = df.groupby(['Company'])['Cash Flow from Operating Activities (in millions)'].pct_change() * 100
    
    # Calculate additional financial metrics
    # ROA - Return on Assets
    df['ROA (%)'] = (df['Net Income (in millions)'] / df['Total Assets (in millions)']) * 100
    
    # Profit Margin
    df['Profit Margin (%)'] = (df['Net Income (in millions)'] / df['Total Revenue (in millions)']) * 100
    
    # Debt-to-Asset Ratio
    df['Debt-to-Asset Ratio'] = df['Total Liabilities (in millions)'] / df['Total Assets (in millions)']
    
    # Group by Company and calculate summary statistics
    analysis_data = {}
    
    for company in df['Company'].unique():
        company_data = df[df['Company'] == company]
        
        # Latest year data
        latest_year = company_data['Fiscal Year'].max()
        latest_data = company_data[company_data['Fiscal Year'] == latest_year].iloc[0]
        
        # Previous year data
        previous_year = latest_year - 1
        previous_data = company_data[company_data['Fiscal Year'] == previous_year]
        
        if not previous_data.empty:
            previous_data = previous_data.iloc[0]
            
            # Calculate changes
            revenue_change = latest_data['Total Revenue (in millions)'] - previous_data['Total Revenue (in millions)']
            revenue_pct = (revenue_change / previous_data['Total Revenue (in millions)']) * 100
            
            net_income_change = latest_data['Net Income (in millions)'] - previous_data['Net Income (in millions)']
            net_income_pct = (net_income_change / previous_data['Net Income (in millions)']) * 100
            
            cash_flow_change = latest_data['Cash Flow from Operating Activities (in millions)'] - previous_data['Cash Flow from Operating Activities (in millions)']
            cash_flow_pct = (cash_flow_change / previous_data['Cash Flow from Operating Activities (in millions)']) * 100
            
            roa_change = latest_data['ROA (%)'] - previous_data['ROA (%)']
            profit_margin_change = latest_data['Profit Margin (%)'] - previous_data['Profit Margin (%)']
            debt_ratio_change = latest_data['Debt-to-Asset Ratio'] - previous_data['Debt-to-Asset Ratio']
        
            analysis_data[company] = {
                'latest_year': latest_year,
                'total_revenue': latest_data['Total Revenue (in millions)'],
                'net_income': latest_data['Net Income (in millions)'],
                'total_assets': latest_data['Total Assets (in millions)'],
                'total_liabilities': latest_data['Total Liabilities (in millions)'],
                'cash_flow': latest_data['Cash Flow from Operating Activities (in millions)'],
                'revenue_change': revenue_change,
                'revenue_pct': revenue_pct,
                'net_income_change': net_income_change,
                'net_income_pct': net_income_pct,
                'cash_flow_change': cash_flow_change,
                'cash_flow_pct': cash_flow_pct,
                'roa': latest_data['ROA (%)'],
                'roa_change': roa_change,
                'profit_margin': latest_data['Profit Margin (%)'],
                'profit_margin_change': profit_margin_change,
                'debt_ratio': latest_data['Debt-to-Asset Ratio'],
                'debt_ratio_change': debt_ratio_change,
                'yearly_data': company_data.to_dict('records')
            }
        else:
            # If no previous year data, just include the latest year's data
            analysis_data[company] = {
                'latest_year': latest_year,
                'total_revenue': latest_data['Total Revenue (in millions)'],
                'net_income': latest_data['Net Income (in millions)'],
                'total_assets': latest_data['Total Assets (in millions)'],
                'total_liabilities': latest_data['Total Liabilities (in millions)'],
                'cash_flow': latest_data['Cash Flow from Operating Activities (in millions)'],
                'roa': latest_data['ROA (%)'],
                'profit_margin': latest_data['Profit Margin (%)'],
                'debt_ratio': latest_data['Debt-to-Asset Ratio'],
                'yearly_data': company_data.to_dict('records')
            }
    
    return analysis_data

def generate_response(query, analysis_data, company):
    """Generate a response to a financial query using NLP processing"""
    # Use the NLP processor to analyze the query
    query_analysis = analyze_query(query, company)
    
    # Check if company exists in analysis data
    if company not in analysis_data:
        return f"Sorry, I don't have information about {company}."
    
    company_data = analysis_data[company]
    
    # Handle different query types based on the NLP analysis
    query_type = query_analysis.get('query_type')
    is_comparison = query_analysis.get('is_comparison', False)
    is_trend = query_analysis.get('is_trend', False)
    is_forecast = query_analysis.get('is_forecast', False)
    
    # Revenue queries
    if query_type == 'revenue_query':
        if is_trend:
            if 'revenue_pct' in company_data:
                growth_text = "grew" if company_data['revenue_pct'] > 0 else "declined"
                return (f"{company}'s revenue {growth_text} by {abs(company_data['revenue_pct']):.1f}% "
                        f"from {company_data['latest_year']-1} to {company_data['latest_year']}, "
                        f"changing from ${company_data['total_revenue'] - company_data['revenue_change']:,.0f} million "
                        f"to ${company_data['total_revenue']:,.0f} million.")
            else:
                return f"{company}'s revenue for {company_data['latest_year']} was ${company_data['total_revenue']:,.0f} million."
        else:
            return (f"{company}'s total revenue for {company_data['latest_year']} was "
                    f"${company_data['total_revenue']:,.0f} million.")
    
    # Net income queries
    elif query_type == 'net_income_query':
        if is_trend:
            if 'net_income_change' in company_data:
                change_direction = "increased" if company_data['net_income_change'] > 0 else "decreased"
                return (f"{company}'s net income {change_direction} from "
                        f"${company_data['net_income'] - company_data['net_income_change']:,.0f} million to "
                        f"${company_data['net_income']:,.0f} million (a change of {company_data['net_income_pct']:.1f}%) "
                        f"between {company_data['latest_year']-1} and {company_data['latest_year']}.")
            else:
                return f"{company}'s net income for {company_data['latest_year']} was ${company_data['net_income']:,.0f} million."
        else:
            return f"{company}'s net income for {company_data['latest_year']} was ${company_data['net_income']:,.0f} million."
    
    # Assets and liabilities queries
    elif query_type == 'assets_liabilities_query':
        return (f"{company}'s total assets for {company_data['latest_year']} were "
                f"${company_data['total_assets']:,.0f} million, and total liabilities were "
                f"${company_data['total_liabilities']:,.0f} million. The debt-to-asset ratio is "
                f"{company_data['debt_ratio']:.2f}.")
    
    # Cash flow queries
    elif query_type == 'cash_flow_query':
        if is_trend:
            if 'cash_flow_change' in company_data:
                change_direction = "increased" if company_data['cash_flow_change'] > 0 else "decreased"
                return (f"{company}'s cash flow from operating activities {change_direction} from "
                        f"${company_data['cash_flow'] - company_data['cash_flow_change']:,.0f} million to "
                        f"${company_data['cash_flow']:,.0f} million (a change of {company_data['cash_flow_pct']:.1f}%) "
                        f"between {company_data['latest_year']-1} and {company_data['latest_year']}.")
            else:
                return (f"{company}'s cash flow from operating activities for {company_data['latest_year']} "
                        f"was ${company_data['cash_flow']:,.0f} million.")
        else:
            return (f"{company}'s cash flow from operating activities for {company_data['latest_year']} "
                    f"was ${company_data['cash_flow']:,.0f} million.")
    
    # Growth queries
    elif query_type == 'growth_query':
        response = f"{company}'s growth metrics for {company_data['latest_year']}:\n"
        
        if 'revenue_pct' in company_data:
            response += f"• Revenue Growth: {company_data['revenue_pct']:.1f}%\n"
        
        if 'net_income_pct' in company_data:
            response += f"• Net Income Growth: {company_data['net_income_pct']:.1f}%\n"
        
        if 'cash_flow_pct' in company_data:
            response += f"• Cash Flow Growth: {company_data['cash_flow_pct']:.1f}%\n"
            
        return response
    
    # Performance overview queries
    elif query_type == 'performance_query':
        response = f"{company}'s financial performance in {company_data['latest_year']}:\n"
        response += f"• Revenue: ${company_data['total_revenue']:,.0f} million\n"
        response += f"• Net Income: ${company_data['net_income']:,.0f} million\n"
        response += f"• Total Assets: ${company_data['total_assets']:,.0f} million\n"
        response += f"• Total Liabilities: ${company_data['total_liabilities']:,.0f} million\n"
        response += f"• Cash Flow from Operations: ${company_data['cash_flow']:,.0f} million\n"
        
        # Add financial ratios
        response += f"• Return on Assets: {company_data['roa']:.1f}%\n"
        response += f"• Profit Margin: {company_data['profit_margin']:.1f}%\n"
        response += f"• Debt-to-Asset Ratio: {company_data['debt_ratio']:.2f}\n"
        
        return response
    
    # Handle forecast queries with a note about the simple model
    elif is_forecast:
        response = f"Based on {company}'s historical data, a simple linear projection suggests:\n"
        
        # Only provide forecast if we have enough data
        if 'revenue_change' in company_data:
            # Project revenue
            current_revenue = company_data['total_revenue']
            revenue_change = company_data['revenue_change']
            projected_revenue = current_revenue + revenue_change
            
            response += f"• Projected Revenue for {company_data['latest_year']+1}: ${projected_revenue:,.0f} million\n"
            
            # Project net income
            if 'net_income_change' in company_data:
                current_net_income = company_data['net_income']
                net_income_change = company_data['net_income_change']
                projected_net_income = current_net_income + net_income_change
                
                response += f"• Projected Net Income for {company_data['latest_year']+1}: ${projected_net_income:,.0f} million\n"
            
            # Add a note about the simplicity of the model
            response += "\nNote: This is a simple linear projection based on year-over-year change and should not be considered a comprehensive forecast."
        else:
            response += "Insufficient historical data to generate a meaningful forecast."
        
        return response
    
    # Comparison queries
    elif is_comparison:
        # This requires data about all companies to compare
        response = f"Comparing {company} with other companies in the dataset:\n"
        
        # Get all companies
        all_companies = list(analysis_data.keys())
        
        if len(all_companies) <= 1:
            return f"No other companies available to compare with {company}."
        
        # Revenue comparison
        response += f"\nRevenue (latest year, in millions):\n"
        for comp in all_companies:
            comp_data = analysis_data[comp]
            response += f"• {comp}: ${comp_data['total_revenue']:,.0f}\n"
        
        # Net income comparison
        response += f"\nNet Income (latest year, in millions):\n"
        for comp in all_companies:
            comp_data = analysis_data[comp]
            response += f"• {comp}: ${comp_data['net_income']:,.0f}\n"
        
        # Profit margin comparison
        response += f"\nProfit Margin (latest year):\n"
        for comp in all_companies:
            comp_data = analysis_data[comp]
            response += f"• {comp}: {comp_data['profit_margin']:.1f}%\n"
        
        return response
    
    # If query type not recognized or is None
    return f"I don't have enough information to answer that query about {company}. Try asking about revenue, net income, assets, liabilities, cash flow, growth, or overall performance."
