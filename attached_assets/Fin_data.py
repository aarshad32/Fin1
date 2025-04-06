import pandas as pd
df = pd.read_csv('Financial_data.csv', thousands=',')

df.head()

df['Revenue Growth (%)'] = df.groupby(['Company'])['Total Revenue (in millions)'].pct_change() * 100
df['Net Income Growth (%)'] = df.groupby(['Company'])['Net Income (in millions)'].pct_change() * 100

# Group by Company and Year and calculate summary statistics
grouped = df.groupby(['Company', 'Fiscal Year']).agg({
    'Total Revenue (in millions)': ['sum', 'mean', 'max', 'min'],
    'Net Income (in millions)': ['sum', 'mean', 'max', 'min'],
    'Total Assets (in millions)': ['sum', 'mean', 'max', 'min'],
    'Total Liabilities (in millions)': ['sum', 'mean', 'max', 'min'],
    'Cash Flow from Operating Activities (in millions)': ['sum', 'mean', 'max', 'min']
}).reset_index()

# Calculate year-over-year growth for each financial metric
df['Revenue Growth (%)'] = df.groupby('Company')['Total Revenue (in millions)'].pct_change() * 100
df['Net Income Growth (%)'] = df.groupby('Company')['Net Income (in millions)'].pct_change() * 100
df['Asset Growth (%)'] = df.groupby('Company')['Total Assets (in millions)'].pct_change() * 100
df['Liability Growth (%)'] = df.groupby('Company')['Total Liabilities (in millions)'].pct_change() * 100
df['Cash Flow Growth (%)'] = df.groupby('Company')['Cash Flow from Operating Activities (in millions)'].pct_change() * 100

# Descriptive statistics of the DataFrame
df.describe()

import matplotlib.pyplot as plt

# Plot revenue growth over the years for each company
plt.figure(figsize=(10,6))
for company in df['Company'].unique():
    company_data = df[df['Company'] == company]
    plt.plot(company_data['Fiscal Year'], company_data['Revenue Growth (%)'], label=company)

plt.title('Revenue Growth (%) Over Years')
plt.xlabel('Fiscal Year')
plt.ylabel('Revenue Growth (%)')
plt.legend()
plt.show()

"""Summary of Findings:
Microsoft:
Total Revenue increased significantly from 2022 to 2023, reflecting a strong year-over-year performance.

Net Income saw a steady growth trend in 2024, with a substantial 15% increase from 2023.

Total Assets and Liabilities both showed an upward trend, but liabilities have grown at a faster rate, indicating potential risks in leveraging.

Cash Flow from Operating Activities grew by 25%, reflecting a strong operational performance.

Tesla:
Tesla’s Revenue Growth was relatively consistent, with small fluctuations, yet still increasing year-over-year.

Net Income Growth was strong in 2023 but dropped significantly in 2024. Further investigation into costs and margins may be necessary.

Total Assets and Total Liabilities grew, but liabilities remain at a manageable level.
"""