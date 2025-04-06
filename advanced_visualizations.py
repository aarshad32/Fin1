import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def create_revenue_chart(df, companies=None):
    """Create a line chart for revenue over time"""
    if companies:
        df = df[df['Company'].isin(companies)]
    
    fig = px.line(
        df, 
        x='Fiscal Year', 
        y='Total Revenue (in millions)', 
        color='Company',
        markers=True,
        title='Revenue Over Time',
        labels={'Total Revenue (in millions)': 'Revenue (in millions)', 'Fiscal Year': 'Year'}
    )
    fig.update_layout(
        xaxis_title='Fiscal Year',
        yaxis_title='Revenue (in millions)',
        legend_title='Company',
        hovermode='x unified'
    )
    return fig

def create_net_income_chart(df, companies=None):
    """Create a line chart for net income over time"""
    if companies:
        df = df[df['Company'].isin(companies)]
    
    fig = px.line(
        df, 
        x='Fiscal Year', 
        y='Net Income (in millions)', 
        color='Company',
        markers=True,
        title='Net Income Over Time',
        labels={'Net Income (in millions)': 'Net Income (in millions)', 'Fiscal Year': 'Year'}
    )
    fig.update_layout(
        xaxis_title='Fiscal Year',
        yaxis_title='Net Income (in millions)',
        legend_title='Company',
        hovermode='x unified'
    )
    return fig

def create_financial_ratio_chart(df, company):
    """Create financial ratios visualization"""
    company_data = df[df['Company'] == company].sort_values('Fiscal Year')
    
    # Calculate ratios if not already present
    if 'ROA (%)' not in company_data.columns:
        company_data['ROA (%)'] = (company_data['Net Income (in millions)'] / company_data['Total Assets (in millions)']) * 100
    
    if 'Profit Margin (%)' not in company_data.columns:
        company_data['Profit Margin (%)'] = (company_data['Net Income (in millions)'] / company_data['Total Revenue (in millions)']) * 100
    
    if 'Debt-to-Asset Ratio' not in company_data.columns:
        company_data['Debt-to-Asset Ratio'] = company_data['Total Liabilities (in millions)'] / company_data['Total Assets (in millions)']
    
    # Create subplots with 3 rows
    fig = make_subplots(rows=3, cols=1, 
                       subplot_titles=("Return on Assets (%)", "Profit Margin (%)", "Debt-to-Asset Ratio"),
                       vertical_spacing=0.1)
    
    # Add traces for each ratio
    fig.add_trace(
        go.Scatter(x=company_data['Fiscal Year'], y=company_data['ROA (%)'], mode='lines+markers', name='ROA (%)'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=company_data['Fiscal Year'], y=company_data['Profit Margin (%)'], mode='lines+markers', name='Profit Margin (%)'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=company_data['Fiscal Year'], y=company_data['Debt-to-Asset Ratio'], mode='lines+markers', name='Debt-to-Asset Ratio'),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        title_text=f"Financial Ratios for {company}",
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_performance_comparison(df, companies, metric='Total Revenue (in millions)'):
    """Create a performance comparison chart"""
    if not companies or len(companies) < 2:
        return None
    
    # Filter dataframe for selected companies
    filtered_df = df[df['Company'].isin(companies)].sort_values(['Company', 'Fiscal Year'])
    
    # Create the comparison chart
    fig = px.bar(
        filtered_df,
        x='Fiscal Year',
        y=metric,
        color='Company',
        barmode='group',
        title=f'Comparison of {metric} Among Companies',
        labels={metric: metric.split('(')[0].strip()}
    )
    
    fig.update_layout(
        xaxis_title='Fiscal Year',
        yaxis_title=metric.split('(')[0].strip(),
        legend_title='Company',
        hovermode='x unified'
    )
    
    return fig

def create_forecast_chart(df, company, metric='Total Revenue (in millions)', periods=2):
    """Create a forecast chart using simple linear projection"""
    company_data = df[df['Company'] == company].sort_values('Fiscal Year')
    
    if len(company_data) < 2:
        return None
    
    # Extract years and values
    years = company_data['Fiscal Year'].values
    values = company_data[metric].values
    
    # Calculate linear regression (simple forecast)
    x = np.array(range(len(years)))
    slope, intercept = np.polyfit(x, values, 1)
    
    # Generate forecast years and values
    last_year = years[-1]
    forecast_years = list(years) + [last_year + i + 1 for i in range(periods)]
    forecast_x = np.array(range(len(years) + periods))
    forecast_values = slope * forecast_x + intercept
    
    # Create dataframe for visualization
    forecast_df = pd.DataFrame({
        'Year': forecast_years,
        'Value': forecast_values,
        'Type': ['Historical'] * len(years) + ['Forecast'] * periods
    })
    
    # Plot the forecast
    fig = px.line(
        forecast_df, 
        x='Year', 
        y='Value', 
        color='Type',
        title=f'Forecast for {company} - {metric}',
        markers=True,
        line_dash='Type'
    )
    
    # Add the actual historical values
    fig.add_trace(
        go.Scatter(
            x=years, 
            y=values, 
            mode='markers',
            marker=dict(size=10),
            name='Actual'
        )
    )
    
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title=metric,
        hovermode='x unified'
    )
    
    return fig

def create_financial_wordcloud(text):
    """Create a wordcloud of financial terms"""
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=100).generate(text)
    
    # Create a matplotlib figure
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # Convert matplotlib figure to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

def create_interactive_chart(df, company, metrics=None):
    """Create a multi-metric interactive chart"""
    company_data = df[df['Company'] == company].sort_values('Fiscal Year')
    
    if metrics is None:
        metrics = [
            'Total Revenue (in millions)',
            'Net Income (in millions)',
            'Cash Flow from Operating Activities (in millions)'
        ]
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    for i, metric in enumerate(metrics):
        if i == 0:
            # First metric on primary y-axis
            fig.add_trace(
                go.Scatter(
                    x=company_data['Fiscal Year'],
                    y=company_data[metric],
                    name=metric.split('(')[0].strip(),
                    mode='lines+markers'
                ),
                secondary_y=False
            )
        else:
            # Additional metrics on secondary y-axis
            fig.add_trace(
                go.Scatter(
                    x=company_data['Fiscal Year'],
                    y=company_data[metric],
                    name=metric.split('(')[0].strip(),
                    mode='lines+markers'
                ),
                secondary_y=True
            )
    
    # Set titles
    fig.update_layout(
        title_text=f"Financial Performance Metrics for {company}",
        hovermode='x unified'
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text="Fiscal Year")
    
    # Set y-axes titles
    fig.update_yaxes(title_text=metrics[0].split('(')[0].strip(), secondary_y=False)
    if len(metrics) > 1:
        fig.update_yaxes(title_text=metrics[1].split('(')[0].strip(), secondary_y=True)
    
    return fig

def create_growth_chart(df, company):
    """Create a chart showing growth rates"""
    company_data = df[df['Company'] == company].sort_values('Fiscal Year')
    
    # Calculate growth rates if they don't exist
    if 'Revenue Growth (%)' not in company_data.columns:
        company_data['Revenue Growth (%)'] = company_data['Total Revenue (in millions)'].pct_change() * 100
    
    if 'Net Income Growth (%)' not in company_data.columns:
        company_data['Net Income Growth (%)'] = company_data['Net Income (in millions)'].pct_change() * 100
    
    # Drop the first row which will have NaN for growth rates
    company_data = company_data.dropna(subset=['Revenue Growth (%)'])
    
    if len(company_data) == 0:
        return None
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=company_data['Fiscal Year'],
            y=company_data['Revenue Growth (%)'],
            name='Revenue Growth (%)',
            marker_color='blue'
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=company_data['Fiscal Year'],
            y=company_data['Net Income Growth (%)'],
            name='Net Income Growth (%)',
            marker_color='green'
        )
    )
    
    # Set layout
    fig.update_layout(
        title=f'Growth Rates for {company}',
        xaxis_title='Fiscal Year',
        yaxis_title='Growth Rate (%)',
        barmode='group',
        hovermode='x unified'
    )
    
    return fig
