import pandas as pd
import plotly.express as px
import re
import numpy as np

# Import new NLP and visualization modules
from nlp_processor import analyze_query
from advanced_visualizations import (
    create_financial_ratio_chart,
    create_performance_comparison,
    create_forecast_chart,
    create_financial_wordcloud,
    create_interactive_chart
)

def load_data():
    """Load the sample financial data"""
    try:
        # Try to load the file from the attached_assets folder
        df = pd.read_csv('./attached_assets/Financial_data.csv', thousands=',')
    except FileNotFoundError:
        # If file not found, create a DataFrame with the provided sample data
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
    
    return df

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
                        f"changing from ${company_data['total_revenue'] - company_data['revenue_change']:,} million "
                        f"to ${company_data['total_revenue']:,} million.")
            else:
                return f"{company}'s revenue for {company_data['latest_year']} was ${company_data['total_revenue']:,} million."
        else:
            return (f"{company}'s total revenue for {company_data['latest_year']} was "
                    f"${company_data['total_revenue']:,} million.")
    
    # Net income queries
    elif query_type == 'net_income_query':
        if is_trend:
            if 'net_income_change' in company_data:
                change_direction = "increased" if company_data['net_income_change'] > 0 else "decreased"
                return (f"{company}'s net income {change_direction} from "
                        f"${company_data['net_income'] - company_data['net_income_change']:,} million to "
                        f"${company_data['net_income']:,} million (a change of {company_data['net_income_pct']:.1f}%) "
                        f"between {company_data['latest_year']-1} and {company_data['latest_year']}.")
            else:
                return f"{company}'s net income for {company_data['latest_year']} was ${company_data['net_income']:,} million."
        else:
            return f"{company}'s net income for {company_data['latest_year']} was ${company_data['net_income']:,} million."
    
    # Assets and liabilities queries
    elif query_type == 'assets_liabilities_query':
        return (f"{company}'s total assets for {company_data['latest_year']} were "
                f"${company_data['total_assets']:,} million, and total liabilities were "
                f"${company_data['total_liabilities']:,} million. The debt-to-asset ratio is "
                f"{company_data['debt_ratio']:.2f}.")
    
    # Cash flow queries
    elif query_type == 'cash_flow_query':
        if is_trend:
            if 'cash_flow_change' in company_data:
                change_direction = "increased" if company_data['cash_flow_change'] > 0 else "decreased"
                return (f"{company}'s cash flow from operating activities {change_direction} from "
                        f"${company_data['cash_flow'] - company_data['cash_flow_change']:,} million to "
                        f"${company_data['cash_flow']:,} million (a change of {company_data['cash_flow_pct']:.1f}%) "
                        f"between {company_data['latest_year']-1} and {company_data['latest_year']}.")
            else:
                return (f"{company}'s cash flow from operating activities for {company_data['latest_year']} "
                        f"was ${company_data['cash_flow']:,} million.")
        else:
            return (f"{company}'s cash flow from operating activities for {company_data['latest_year']} "
                    f"was ${company_data['cash_flow']:,} million.")
    
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
        response += f"• Revenue: ${company_data['total_revenue']:,} million\n"
        response += f"• Net Income: ${company_data['net_income']:,} million\n"
        response += f"• Total Assets: ${company_data['total_assets']:,} million\n"
        response += f"• Total Liabilities: ${company_data['total_liabilities']:,} million\n"
        response += f"• Cash Flow from Operations: ${company_data['cash_flow']:,} million\n"
        
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
                current_income = company_data['net_income']
                income_change = company_data['net_income_change']
                projected_income = current_income + income_change
                
                response += f"• Projected Net Income for {company_data['latest_year']+1}: ${projected_income:,.0f} million\n"
        
        response += "\nNote: This is a simple linear projection based on limited historical data and should not be used for investment decisions."
        return response
    
    # If no specific query type matched, use fallback for general questions
    else:
        return (f"{company}'s financial performance in {company_data['latest_year']}:\n"
                f"• Revenue: ${company_data['total_revenue']:,} million\n"
                f"• Net Income: ${company_data['net_income']:,} million\n"
                f"• Total Assets: ${company_data['total_assets']:,} million\n"
                f"• Total Liabilities: ${company_data['total_liabilities']:,} million\n"
                f"• Cash Flow from Operations: ${company_data['cash_flow']:,} million")

def generate_visualization(query, company_data, company):
    """Generate an appropriate visualization based on the query using NLP analysis"""
    # Use the NLP processor to analyze the query
    query_analysis = analyze_query(query, company)
    
    # Extract query information
    query_type = query_analysis.get('query_type')
    is_comparison = query_analysis.get('is_comparison', False)
    is_trend = query_analysis.get('is_trend', False)
    is_forecast = query_analysis.get('is_forecast', False)
    metrics = query_analysis.get('metrics', ['all'])
    
    # Map metrics to column names
    metric_map = {
        'revenue': 'Total Revenue (in millions)',
        'net_income': 'Net Income (in millions)',
        'assets': 'Total Assets (in millions)',
        'liabilities': 'Total Liabilities (in millions)',
        'cash_flow': 'Cash Flow from Operating Activities (in millions)',
        'all': None  # Special case handled separately
    }
    
    # Revenue visualization
    if query_type == 'revenue_query':
        if 'growth' in query.lower():
            title = f"{company} - Revenue Growth Over Time"
            fig = px.line(company_data, x="Fiscal Year", 
                          y="Revenue Growth (%)", 
                          title=title,
                          markers=True,
                          labels={"Revenue Growth (%)": "Growth (%)"})
            return fig
        else:
            title = f"{company} - Revenue"
            fig = px.bar(company_data, x="Fiscal Year", 
                         y="Total Revenue (in millions)",
                         title=title,
                         labels={"Total Revenue (in millions)": "Revenue (in millions $)"})
            return fig
    
    # Net income visualization
    elif query_type == 'net_income_query':
        if 'growth' in query.lower():
            title = f"{company} - Net Income Growth Over Time"
            fig = px.line(company_data, x="Fiscal Year", 
                          y="Net Income Growth (%)", 
                          title=title,
                          markers=True,
                          labels={"Net Income Growth (%)": "Growth (%)"})
            return fig
        else:
            title = f"{company} - Net Income"
            fig = px.bar(company_data, x="Fiscal Year", 
                         y="Net Income (in millions)",
                         title=title,
                         labels={"Net Income (in millions)": "Net Income (in millions $)"})
            return fig
    
    # Assets and liabilities visualization
    elif query_type == 'assets_liabilities_query':
        title = f"{company} - Assets and Liabilities"
        fig = px.bar(company_data, x="Fiscal Year", 
                     y=["Total Assets (in millions)", "Total Liabilities (in millions)"],
                     title=title,
                     barmode="group",
                     labels={"value": "Amount (in millions $)", "variable": "Metric"})
        return fig
    
    # Cash flow visualization
    elif query_type == 'cash_flow_query':
        if 'growth' in query.lower():
            title = f"{company} - Cash Flow Growth Over Time"
            fig = px.line(company_data, x="Fiscal Year", 
                          y="Cash Flow Growth (%)", 
                          title=title,
                          markers=True,
                          labels={"Cash Flow Growth (%)": "Growth (%)"})
            return fig
        else:
            title = f"{company} - Cash Flow from Operating Activities"
            fig = px.line(company_data, x="Fiscal Year", 
                          y="Cash Flow from Operating Activities (in millions)",
                          title=title,
                          markers=True,
                          labels={"Cash Flow from Operating Activities (in millions)": "Cash Flow (in millions $)"})
            return fig
    
    # Growth visualization - comprehensive growth chart
    elif query_type == 'growth_query':
        title = f"{company} - Financial Growth Metrics"
        growth_metrics = [col for col in company_data.columns if 'Growth' in col]
        fig = px.line(company_data, x="Fiscal Year", 
                      y=growth_metrics,
                      title=title,
                      markers=True,
                      labels={"value": "Growth (%)", "variable": "Metric"})
        return fig
    
    # Performance overview - use financial ratio chart
    elif query_type == 'performance_query':
        return create_financial_ratio_chart(company_data, company)
    
    # Comparison chart (if available)
    elif is_comparison:
        # This would need access to all data to compare companies
        # For now, we'll just display the current company with a note
        title = f"{company} - Financial Overview (Comparison not available)"
        fig = px.line(company_data, x="Fiscal Year", 
                      y=["Total Revenue (in millions)", "Net Income (in millions)", 
                         "Cash Flow from Operating Activities (in millions)"],
                      title=title,
                      markers=True,
                      labels={"value": "Amount (in millions $)", "variable": "Metric"})
        return fig
    
    # Forecast visualization
    elif is_forecast:
        return create_forecast_chart(company_data, company)
    
    # If specific metrics were identified
    elif metrics and metrics != ['all']:
        columns = [metric_map.get(metric) for metric in metrics if metric in metric_map and metric_map.get(metric)]
        if columns:
            title = f"{company} - Selected Financial Metrics"
            fig = create_interactive_chart(company_data, 'line', columns)
            return fig
    
    # Default visualization - financial overview
    title = f"{company} - Financial Overview"
    fig = px.line(company_data, x="Fiscal Year", 
                  y=["Total Revenue (in millions)", "Net Income (in millions)", 
                     "Cash Flow from Operating Activities (in millions)"],
                  title=title,
                  markers=True,
                  labels={"value": "Amount (in millions $)", "variable": "Metric"})
    return fig
