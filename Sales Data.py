import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


import pymysql
from IPython.display import display



 #Database connection details
db_config = {
    "host": "localhost",  # Change to your database host
    "user": "root",       # Change to your database username
    "password": "Sql@2025",  # Change to your database password
    "database": "sales"  # Change to your database name
}

#SQL Queries
queries = {
    "question1: During the transactions that occurred in 2021, in which month did the total transaction value (after_discount) reach its highest? Use is_valid = 1 to filter transaction data. Source table : order_detail" :
        """SELECT
            DATE_FORMAT(order_date, '%m') AS Month_ID,
            DATE_FORMAT(order_date, '%M') AS Month,
            DATE_FORMAT(order_date, '%Y') AS Year,
            SUM(after_discount) AS total_transaction
        FROM order_detail
        WHERE YEAR(order_date) = 2021 
          AND is_valid = 1
        GROUP BY Month_ID, Month, Year
        ORDER BY total_transaction DESC;
    """,
    "question2 During transactions in the year 2022, which category generated the highest transaction value? Use is_valid = 1 to filter transaction data. Source table : order_detail, sku_detail" :
    """
        SELECT
            DATE_FORMAT(ordet.order_date, '%Y') AS year,
            skudet.category,
            SUM(ordet.after_discount) AS total_transaction
        FROM
            order_detail AS ordet
            JOIN sku_detail AS skudet ON ordet.sku_id = skudet.id
        WHERE
            ordet.is_valid = 1 
            AND DATE_FORMAT(ordet.order_date, '%Y') = '2022'
        GROUP BY
            DATE_FORMAT(ordet.order_date, '%Y'),
            skudet.category
        ORDER BY
            total_transaction DESC;"""}

# Connect to the database
try:
    connection = pymysql.connect(**db_config)
    print("Database connection successful!")

    # Execute each query and display results
    for question, query in queries.items():
        print(f"\n--- {question} ---")
        df = pd.read_sql(query, connection)
        display(df)  # Display the DataFrame in the notebook

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if connection:
        connection.close()
        print("Database connection closed.")

import os
# Database connection details
db_config = {
    "host": "localhost",  # Change to your database host
    "user": "root",  # Change to your database username
    "password": "Sql@2025",  # Change to your database password
    "database": "sales"  # Change to your database name
}

# Folder path to save CSV files
output_folder = r"C:\Users\selva\Downloads\Google Looker project\dataset"  # Replace with your folder path
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

# List of tables to export
tables = ["order_detail", "sku_detail", "payment_detail", "customer_detail"]  # Add your table names here

try:
    # Connect to the database
    connection = pymysql.connect(**db_config)
    print("Database connection successful!")

    for table in tables:
        print(f"Exporting table: {table}")

        # SQL query to fetch all data from the table
        query = f"SELECT * FROM {table};"

        # Read table data into a DataFrame
        df = pd.read_sql(query, connection)

        # Save the DataFrame to a CSV file
        output_file = os.path.join(output_folder, f"{table}.csv")
        df.to_csv(output_file, index=False)

        print(f"Table {table} exported successfully to {output_file}.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if connection:
        connection.close()
        print("Database connection closed.")
df_od = pd.read_csv(os.path.join(output_folder, "order_detail.csv"))
df_sd = pd.read_csv(os.path.join(output_folder, "sku_detail.csv"))
df_pd = pd.read_csv(os.path.join(output_folder, "payment_detail.csv"))
df_cd = pd.read_csv(os.path.join(output_folder, "customer_detail.csv"))

# Merge the dataframes
# Before merge we need to rename column in order to avoid duplicate column value
# So the key column will be excluded when joined
df_sd.rename(columns={'id':'sku_id'}, inplace=True)
df_cd.rename(columns={'id':'customer_id'}, inplace=True)
df_pd.rename(columns={'id':'payment_id'}, inplace=True)

# Merge the dataframes using left join
df = pd.DataFrame(df_od\
                  # Merge order_detail with sku_detail on column sku_id
                  .merge(df_sd, how='left', on='sku_id')\
                  # Merge the result to customer_detail on column customer_id
                  .merge(df_cd, how='left', on='customer_id')\
                  # Merge the result to payment_detail on column payment_id
                  .merge(df_pd, how='left', on='payment_id')
                  )
# check the dataframe information
df.info()
print(df_pd.columns)
print(df_od.columns)
df_pd['payment_id'] = df_pd['payment_id'].astype(str)  # Convert to string
df_od['id'] = df_od['id'].astype(str)                  # Convert to string

# Merge order_detail with payment_detail without renaming columns
df_sample = pd.merge(df_pd, df_od, how='left', left_on='payment_id', right_on='id')

# Display info about the resulting DataFrame
df_sample.info()

# change columns to datetime format using for loop
# use pandas.to_datetime to convert
for x in ['order_date', 'registered_date']:
  df[x] = pd.to_datetime(df[x])

df.info()

#Question 1

"""" Sales Prediction for the Next Quarter Using Historical Data Scenario: 
The Sales Team wants to predict the total sales for the next quarter (Q2 2023) based on historical sales data.
Requirements:..1. Use the sales data from Q1 2022 to Q4 2022 to build a model that predicts total sales for Q2 2023.
2. Evaluate the model’s accuracy using a relevant error metric (e.g., MAE, RMSE). 
3. Provide predictions for the upcoming quarter and identify any trends. 
Key Features to Use: • order\_date • qty\_ordered • 
Machine learning model libraries (e.g., Scikit-learn, Statsmodels)"""


df['order_date'] = pd.to_datetime(df['order_date'])

# Extract time features
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
df['quarter'] = df['order_date'].dt.to_period('Q')
# Filter only 2022 sales
df = df[df['year'] == 2022]

# Aggregate total sales per quarter
quarterly_sales = df.groupby('quarter')['qty_ordered'].sum().reset_index()
print(quarterly_sales)
print(f"Quantity order by quarter:"+str(df['qty_ordered'].sum()))

# Prepare numeric features
X = np.arange(len(quarterly_sales)).reshape(-1, 1)  # e.g., 0 for Q1, 1 for Q2, etc.
y = quarterly_sales['qty_ordered'].values


model = LinearRegression()
model.fit(X, y)

# Predict for next quarter (index = 4)
from math import sqrt
next_quarter_index = len(X)  # = 4 → corresponds to Q1 2023
next_quarter_pred = model.predict([[next_quarter_index + 1]])  # = 5 → Q2 2023
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
rmse = sqrt(mean_squared_error(y, y_pred))

print(f'MAE: {mae}, RMSE: {rmse}')

print(f"Predicted Sales for Q2 2023: {next_quarter_pred[0]:.2f}")



plt.figure(figsize=(10, 5))

# Plot actual and predicted sales
plt.plot(quarterly_sales.index, y, label='Actual Sales', marker='o')
plt.plot(quarterly_sales.index, y_pred, label='Predicted Sales', linestyle='--', marker='o')
quarters = ['Q1 2022', 'Q2 2022', 'Q3 2022', 'Q4 2022']
plt.xticks(ticks=range(len(quarters)), labels=quarters)



# Add axis labels and title
plt.xlabel("Quarter of 2022")
plt.ylabel("Sales Quantity")
plt.title("Quarterly Sales Trend")
plt.legend()
plt.grid(True)

# Annotate actual values
for i, value in enumerate(y):
    plt.text(quarterly_sales.index[i], value + 5, f'{value:.0f}', ha='center', fontsize=9, color='blue')

# Annotate predicted values
for i, value in enumerate(y_pred):
    plt.text(quarterly_sales.index[i], value - 10, f'{value:.0f}', ha='center', fontsize=9, color='green')

plt.tight_layout()
plt.show()

# Question number 2
"""Sales Performance Based on Payment Method Scenario: The Finance Team wants to understand how each payment method performs in terms of revenue, 
quantity sold, and net profit. Requirements: • Create a table that lists each payment method (payment_method) along with:
o Total sales (SUM(before_discount)) o Total quantity sold (SUM(qty_ordered))
o Total net profit (SUM(after_discount - cogs)) • 
Add filters to view this data by month or quarter. Key Features to Use: • payment_method.• before_discount • qty_ordered • cogs"""

df['order_date'] = pd.to_datetime(df['order_date'])

# Extract month and quarter
df['month'] = df['order_date'].dt.to_period('M')
df['quarter'] = df['order_date'].dt.to_period('Q')
df['net_profit'] = df['after_discount'] - df['cogs']
summary = df.groupby('payment_method').agg(
    total_sales=pd.NamedAgg(column='before_discount', aggfunc='sum'),
    total_quantity=pd.NamedAgg(column='qty_ordered', aggfunc='sum'),
    total_net_profit=pd.NamedAgg(column='net_profit', aggfunc='sum')
).reset_index()
summary_by_month = df.groupby(['payment_method', 'month']).agg(
    total_sales=('before_discount', 'sum'),
    total_quantity=('qty_ordered', 'sum'),
    total_net_profit=('net_profit', 'sum')
).reset_index()
print(summary_by_month)


import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Melt the summary DataFrame for grouped bar chart
summary_melted = summary_by_month.melt(
    id_vars=['payment_method','month'],
    value_vars=['total_sales', 'total_net_profit','total_quantity'],
    var_name='Metric',
    value_name='Value'
)

plt.figure(figsize=(10, 5))
sns.barplot(x='payment_method', y='Value', hue='Metric', data=summary_melted,errorbar=None)
plt.yscale('log')  # Or annotate with text
plt.title('Sales Performance by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.legend(title='Metric')

# Apply formatter to avoid scientific notation
formatter = FuncFormatter(lambda x, _: f'{int(x):,}')
plt.gca().yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.show()


# Specify the file path and name
file_path = r"C:\Users\selva\Downloads\Google Looker project\dataset2\finaldataset.csv" # Update with your desired folder path

os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Save to CSV
df.to_csv(file_path, index=False)

print(f"File successfully saved to {file_path}")








