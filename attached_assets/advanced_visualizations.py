import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
import numpy as np

def create_financial_ratio_chart(company_data, company):
    """Create financial ratio charts for the selected company"""
    # Calculate common financial ratios
    company_data = company_data.copy()
    
    # Current ratio (if we had current assets and current liabilities)
    # For now we'll use total assets/liabilities
    company_data['Debt-to-Asset Ratio'] = company_data['Total Liabilities (in millions)'] / company_data['Total Assets (in millions)']
    
    # Return on Assets (ROA)
    company_data['Return on Assets'] = company_data['Net Income (in millions)'] / company_data['Total Assets (in millions)']
    
    # Profit Margin
    company_data['Profit Margin'] = company_data['Net Income (in millions)'] / company_data['Total Revenue (in millions)']
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=company_data['Fiscal Year'], y=company_data['Debt-to-Asset Ratio'],
                 mode='lines+markers', name='Debt-to-Asset Ratio'),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=company_data['Fiscal Year'], y=company_data['Return on Assets'],
                 mode='lines+markers', name='Return on Assets (ROA)'),
        secondary_y=True,
    )
    
    fig.add_trace(
        go.Scatter(x=company_data['Fiscal Year'], y=company_data['Profit Margin'],
                 mode='lines+markers', name='Profit Margin'),
        secondary_y=True,
    )
    
    # Add figure title and axis labels
    fig.update_layout(
        title_text=f"{company} - Financial Ratios Analysis",
        hovermode="x unified"
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Debt-to-Asset Ratio", secondary_y=False)
    fig.update_yaxes(title_text="ROA & Profit Margin", secondary_y=True)
    
    return fig

def create_performance_comparison(all_data, companies, metric):
    """Create a comparison chart for multiple companies based on a specific metric"""
    # Identify the metric and title
    metric_map = {
        'revenue': 'Total Revenue (in millions)',
        'net_income': 'Net Income (in millions)',
        'assets': 'Total Assets (in millions)',
        'liabilities': 'Total Liabilities (in millions)',
        'cash_flow': 'Cash Flow from Operating Activities (in millions)'
    }
    
    if metric in metric_map:
        metric_column = metric_map[metric]
    else:
        metric_column = 'Total Revenue (in millions)'  # Default
    
    # Filter data for selected companies
    filtered_data = all_data[all_data['Company'].isin(companies)]
    
    # Create bar chart
    fig = px.bar(
        filtered_data, 
        x='Fiscal Year', 
        y=metric_column,
        color='Company', 
        barmode='group',
        title=f"Company Comparison - {metric_column}",
        labels={"value": f"{metric_column}", "variable": "Company"}
    )
    
    # Add data labels
    fig.update_traces(texttemplate='%{y:.1f}B', textposition='outside')
    
    # Improve interactivity
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
    
    return fig

def create_forecast_chart(company_data, company, periods=2):
    """Create a simple forecast for future periods based on historical trend"""
    # Convert to DataFrame if it's not already
    if not isinstance(company_data, pd.DataFrame):
        return None
    
    # Get the years
    years = company_data['Fiscal Year'].values
    
    # Ensure at least 2 data points for forecasting
    if len(years) < 2:
        return None
    
    # Create a DataFrame with the forecast
    forecast_df = company_data.copy()
    
    # Add forecast periods
    last_year = forecast_df['Fiscal Year'].max()
    future_years = list(range(last_year + 1, last_year + periods + 1))
    
    # Metrics to forecast
    metrics = [
        'Total Revenue (in millions)', 
        'Net Income (in millions)', 
        'Total Assets (in millions)',
        'Total Liabilities (in millions)',
        'Cash Flow from Operating Activities (in millions)'
    ]
    
    # Create new rows for future years
    for year in future_years:
        new_row = {'Fiscal Year': year, 'Company': company}
        forecast_df = pd.concat([forecast_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Sort by year
    forecast_df = forecast_df.sort_values('Fiscal Year')
    
    # For each metric, calculate a simple linear forecast
    for metric in metrics:
        # Get actual values
        y = company_data[metric].values
        x = np.arange(len(y))
        
        # Skip if there are NaNs
        if np.isnan(y).any():
            continue
            
        # Fit linear regression
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        # Predict for all years (including future)
        all_years = forecast_df['Fiscal Year'].unique()
        x_pred = np.arange(len(all_years))
        
        # Store predictions
        predictions = p(x_pred)
        
        # Update the forecast DataFrame
        for i, year in enumerate(all_years):
            mask = forecast_df['Fiscal Year'] == year
            
            # Check if it's a forecast year
            if year > last_year:
                forecast_df.loc[mask, metric] = predictions[i]
    
    # Create traces for actual and forecast data
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Split data into actual and forecast
    actual_data = forecast_df[forecast_df['Fiscal Year'] <= last_year]
    forecast_data = forecast_df[forecast_df['Fiscal Year'] > last_year]
    
    # Add traces for each metric
    colors = px.colors.qualitative.Plotly
    
    for i, metric in enumerate(metrics):
        color = colors[i % len(colors)]
        
        # Actual data
        fig.add_trace(
            go.Scatter(
                x=actual_data['Fiscal Year'], 
                y=actual_data[metric],
                mode='lines+markers', 
                name=f"{metric} (Actual)",
                line=dict(color=color)
            )
        )
        
        # Forecast data (if it has the metric)
        if metric in forecast_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast_data['Fiscal Year'], 
                    y=forecast_data[metric],
                    mode='lines+markers', 
                    name=f"{metric} (Forecast)",
                    line=dict(color=color, dash='dash')
                )
            )
    
    # Update layout
    fig.update_layout(
        title_text=f"{company} - Financial Metrics Forecast",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_financial_wordcloud(financial_data, company):
    """Create a word cloud from financial data"""
    # Extract relevant financial data
    if financial_data is None or not isinstance(financial_data, pd.DataFrame):
        return None
    
    company_data = financial_data[financial_data['Company'] == company]
    
    if company_data.empty:
        return None
    
    # Create a text summary of the financial data
    latest_year_data = company_data.sort_values('Fiscal Year', ascending=False).iloc[0]
    
    text_data = f"""
    {company} Financial Performance
    Revenue {latest_year_data.get('Total Revenue (in millions)', 0)} million
    Net Income {latest_year_data.get('Net Income (in millions)', 0)} million
    Assets {latest_year_data.get('Total Assets (in millions)', 0)} million
    Liabilities {latest_year_data.get('Total Liabilities (in millions)', 0)} million
    Cash Flow {latest_year_data.get('Cash Flow from Operating Activities (in millions)', 0)} million
    """
    
    # Add growth metrics
    growth_metrics = [col for col in latest_year_data.index if 'Growth' in col]
    for metric in growth_metrics:
        value = latest_year_data.get(metric, 0)
        if not pd.isna(value):
            text_data += f"{metric} {value:.1f}% "
    
    # Generate the word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text_data)
    
    # Convert to image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    # Save to a BytesIO object
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    
    # Get the base64 string
    img_str = base64.b64encode(img_bytes.read()).decode()
    
    # Close the figure to free memory
    plt.close()
    
    return img_str

def create_interactive_chart(company_data, chart_type, metrics=None):
    """Create custom interactive charts based on user preference"""
    if metrics is None:
        metrics = ['Total Revenue (in millions)', 'Net Income (in millions)']
    
    company = company_data['Company'].iloc[0] if not company_data.empty else "Company"
    
    if chart_type == 'line':
        fig = px.line(
            company_data, 
            x='Fiscal Year', 
            y=metrics,
            title=f"{company} - Financial Metrics Over Time",
            markers=True,
            labels={"value": "Amount (in millions $)", "variable": "Metric"}
        )
        
    elif chart_type == 'bar':
        fig = px.bar(
            company_data, 
            x='Fiscal Year', 
            y=metrics,
            title=f"{company} - Financial Metrics Comparison",
            barmode='group',
            labels={"value": "Amount (in millions $)", "variable": "Metric"}
        )
        
    elif chart_type == 'area':
        fig = px.area(
            company_data, 
            x='Fiscal Year', 
            y=metrics,
            title=f"{company} - Financial Metrics Trends",
            labels={"value": "Amount (in millions $)", "variable": "Metric"}
        )
        
    elif chart_type == 'scatter':
        # Only works with 2 metrics
        if len(metrics) >= 2:
            fig = px.scatter(
                company_data, 
                x=metrics[0], 
                y=metrics[1],
                size='Fiscal Year',
                color='Fiscal Year',
                title=f"{company} - {metrics[0]} vs {metrics[1]} Relationship",
                labels={metrics[0]: f"{metrics[0]} (in millions $)", metrics[1]: f"{metrics[1]} (in millions $)"}
            )
        else:
            # Fallback to line chart
            fig = px.line(
                company_data, 
                x='Fiscal Year', 
                y=metrics,
                title=f"{company} - Financial Metrics Over Time",
                markers=True,
                labels={"value": "Amount (in millions $)", "variable": "Metric"}
            )
    else:
        # Default to line chart
        fig = px.line(
            company_data, 
            x='Fiscal Year', 
            y=metrics,
            title=f"{company} - Financial Metrics Over Time",
            markers=True,
            labels={"value": "Amount (in millions $)", "variable": "Metric"}
        )
    
    # Add hover data
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>" +
                      "%{y} million $<br>"
    )
    
    # Improve layout
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
    
    return fig