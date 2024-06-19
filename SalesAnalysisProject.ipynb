#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import os 
import matplotlib.pyplot as plt
import calendar
from itertools import combinations
from collections import Counter


# # Reading the files

# In[3]:


#TO SEE THE AVAILABLE FILES IN THE DIRECTORY OR FOLDER
files= [file for file in os.listdir('C:/Projects Completed/Sales Project/Dataset')]

for file in files:
    print(file)


# In[4]:


#TO READ ALL AVAILABLE FILES IN THE DIRECTORY OR FOLDER AND COMBINE THEM IN TO A SIGNLE FILE
files= [file for file in os.listdir('C:/Projects Completed/Sales Project/Dataset')]

all_months_data = pd.DataFrame()

for file in files:
    data= pd.read_csv(r"C:/Projects Completed/Sales Project/Dataset/"+file)
    all_months_data = pd.concat([all_months_data,data])
    
# all_months_data.head()

all_months_data.to_csv("C:/Projects Completed/Sales Project/all_months_data.csv",index=False)


# In[5]:


all_months_data.info()


# In[6]:


#TO READ THE CONSOLIDATED DATA FILE
df=pd.read_csv(r"C:/Projects Completed/Sales Project/all_months_data.csv")
df.head()


# # Data cleaning

# In[7]:


df.shape


# In[8]:


#TO CHECK INFO ABOUT THE DF
df.info()


# In[9]:


# TO CHECK THE UNIQUE VALUES IN PRODUCT COLUMN
df['Product'].unique()


# In[10]:


#TO CHECK DUPLICATE 
df.duplicated().sum()


# In[11]:


#TO REMOVE ALL DUPLICATES 
df=df.drop_duplicates()
df.head()


# In[12]:


# TO CHECK THE NULL VALUES IN DF
df.isna().sum()


# In[13]:


# TO SEE THE ROW WITH NULL VALUES
df[df.isnull().any(axis=1)]


# In[14]:


# REMOVE THE ROWS WITH NULL VALUES
df.dropna(inplace=True)


# In[15]:


# SPLITTING ADDRESS IN TO 3 SEPARATE COLUMNS
df[["Street_address","City","ZIP"]]=df["Purchase Address"].str.split(',',3,expand=True)
df.head()


# In[16]:


# SPLITTING CITY IN TO 2 SEPARATE COLUMNS
df[["State","ZIP"]]=df["ZIP"].str.split(expand=True)

df.head()


# In[17]:


# DROPPING "PURCHASE ADDRESS" COLUMN
df= df.drop(columns="Purchase Address")
df.head()


# In[100]:


# TO CHECK THE UNIQUE VALUES IN CITY COLUMN
df['City'].unique()


# In[101]:


# TO CHECK THE UNIQUE VALUES IN State COLUMN
df['State'].unique()


# In[18]:


# FILTER ROWS WITH NULL VALUE
rows_with_null = df[df.isnull().any(axis=1)]
print(rows_with_null)


# In[19]:


#TO REMOVE THE ROWS WITH NULL
df = df.dropna()


# In[20]:


# Convert 'Quantity Ordered' to integer
df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
df['Quantity Ordered']= df['Quantity Ordered'].astype('int32')

# Convert 'Price Each' to float (assuming it represents a price)
df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce')

# Convert 'Order Date' to datetime, dropping the time part
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# Print the data types of the DataFrame after conversion
print(df.dtypes)


# # Task 1 : Adding Month & Year Column

# In[21]:


# Extract month value into a new column 'Month' & 'Year'
df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year
df.head()


# In[22]:


#Creating a MonthYear Column
df['MonthYear'] = df['Order Date'].dt.to_period('M') 
df.head()


# In[23]:


#Creating a Hour Column
df['Hour'] = df['Order Date'].dt.hour 
df.head()


# # Task 2 : Calculating  & adding Sales column

# In[24]:


# Calculate sales value into a new column 'Sales'
df['Sales']=df['Quantity Ordered']*df['Price Each']
df.head()


# # PROBLEM & SOLUTIONS

# # Problem 1: What was the best month for sales in 2019 ? How much was earned that month ?

# In[25]:


Monthly_sales=df[df['Year']==2019] # we want to know sales of 2019
Monthly_sales= Monthly_sales.groupby('Month')['Sales'].sum().reset_index()
Monthly_sales


# In[26]:


months_num = range(1,13)
month_names= [calendar.month_name[m] for m in months_num]
print(month_names)


# In[27]:


#Plotting the column bar chart
plt.figure(figsize=(12, 6))  # Adjust dimensions as necessary

plt.bar(month_names,Monthly_sales['Sales'],color='teal')

plt.xlabel("Months",fontsize=12)
plt.ylabel("Sales in USD",fontsize=12)
plt.title(" Sales by Months",fontsize=14)

# Adjust y-axis tick formatting to display full numbers
plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

plt.xticks(rotation="vertical",size=10)  # Rotate x-axis labels for better visibility

# Display plot
plt.tight_layout()  # Ensure all elements fit nicely in the figure
plt.show()


# # Solution
# 
# The best month for sales is December with $4,608,295.70.
# 
# Overall Trend: 
# 
# The sales data shows fluctuations throughout the year, with some months significantly higher or lower than others.
# 
# Monthly Performance:
# 
# January starts at $1,812,742.87 and increases gradually until March.
# 
# There is a noticeable peak in October ($3,734,777.86 )  and December ( $4,608,295.70), indicating potential seasonal spikes, possibly due to holidays or special promotions.
# 
# Sales dip in the middle months (June, July, August, September) compared to the peak months.

# # Problem 2: Which city had the highest sales? Which state had the highest sales?

# In[115]:


city_sales= df.groupby('City')['Sales'].sum().reset_index()
city_sales['Sales'] = city_sales['Sales'].round(0)

city_sales


# In[57]:


#Plotting the column bar chart for citywise sales
plt.figure(figsize=(12, 6))  # Adjust dimensions as necessary

plt.bar(city_sales['City'],city_sales['Sales'],color="teal")

plt.xlabel("City",fontsize=12)
plt.ylabel("Sales in USD",fontsize=12)
plt.title("Citywise Sales",fontsize=14)

plt.xticks(rotation=45)

# Display plot
plt.tight_layout()  # Ensure all elements fit nicely in the figure
plt.show()


# In[30]:


# Finding Statewsie sales 
state_sales= df.groupby('State')['Sales'].sum().reset_index()
state_sales['Sales'] = city_sales['Sales'].round(0)

state_sales


# In[59]:


#Plotting the column bar chart for statewise sales
plt.figure(figsize=(12, 6))  # Adjust dimensions as necessary

plt.bar(state_sales['State'],state_sales['Sales'],color="teal")

plt.xlabel("States",fontsize=12)
plt.ylabel("Sales in USD",fontsize=12)
plt.title("Statewise Sales",fontsize=14)

# Display plot
plt.tight_layout()  # Ensure all elements fit nicely in the figure
plt.show()


# # Solution
# 
# Citywise sales :
# 
# San Francisco leads with the highest sales ($8,254,744.00), followed by Los Angeles ($5,448,304.00) and New York City ($4,661,867.00).
# 
# Austin has the lowest sales ($1,818,044.00) among the cities listed.
# 
# Statewise Sales:
# 
# Washington (WA) leads with the highest sales ($8,254,744.00), followed by Oregon (OR) ($4,661,867.00) and New York (NY) ($5,448,304.00).
# 
# Texas (TX) has the lowest sales ($2,319,332.00) among the states listed.
# 

# # Problem 3: What time should we display advertisements to maximize likelihood of customer buying product?

# In[32]:


#Number of orders each hour
hour_counts=df.groupby('Hour').size().reset_index(name='Orders')
print(hour_counts)


# In[75]:


# Line chart for hourly purchases count
hours = df['Hour'].value_counts().sort_index().index
counts = df['Hour'].value_counts().sort_index()

plt.plot(hours, counts,color="teal")
plt.xlabel('Hour')
plt.ylabel('Number of orders')
plt.title('Orders every hour')
plt.xticks(hours)
plt.grid(True)
plt.tight_layout()
plt.show()


# # Solution:
# 
# The peak hours for orders are generally from 10 AM to 8 PM.
# 
# Specifically, the hours with consistently high order numbers include 10 AM, 11 AM, 12 PM, 1 PM, 6 PM, 7 PM, and 8 PM.
# 
# The optimal times to display advertisements to maximize the likelihood of customer purchases are during these peak hours: 10 AM - 12 PM, 12 PM - 1 PM, and 6 PM - 8 PM.

# # Problem 4: What products are most often sold together?

# In[34]:


#To check rows with duplicate order id
orderids=df[df['Order ID'].duplicated(keep=False)]
orderids.head(10)


# In[35]:


#Creating a group column where product for each ordered is summarized
orderids['Grouped']=orderids.groupby('Order ID')['Product'].transform(lambda x : ','.join(x))
orderids.head()

#droppig duplicates to get products for each order id
groups=orderids[['Order ID','Grouped']].drop_duplicates()
groups.head()


# In[50]:


# Finding number of orders for each group
count= Counter()

for row in groups['Grouped']:
    row_list=row.split(',')
    count.update(Counter(combinations(row_list,2)))
    
# Convert the Counter object to a list of tuples
count_list = [(pair, count) for pair, count in count.items()]

# Create a dataframe from the list of tuples
group_count = pd.DataFrame(count_list, columns=['Pair', 'Count'])

# Sort the dataframe by 'Count' column in descending order
group_count = group_count.sort_values(by='Count', ascending=False)

# Reset index for better display (optional)
group_count = group_count.reset_index(drop=True)


# Print or display the dataframe
display(group_count)


# In[82]:


# Viewing Top 10 Groups
# Display only the top 10 groups with counts
top_10_groups = group_count.head(10)

# Print or display the top 10 groups
display(top_10_groups)


# In[84]:


#Plotting the column bar chart for Top 10 groups
top_10_groups = top_10_groups.sort_values(by='Count', ascending=True)
plt.figure(figsize=(12, 6))  # Adjust dimensions as necessary

# Extracting x-axis labels (converting tuples to strings)
y_labels = [', '.join(pair) for pair in top_10_groups['Pair']]

plt.barh(y_labels, top_10_groups['Count'],color="teal")

plt.ylabel("Product Grouped",fontsize=12)
plt.xlabel("Number of orders",fontsize=12)
plt.title("Top 10 groups",fontsize=14)

# Display plot
plt.tight_layout()  # Ensure all elements fit nicely in the figure
plt.grid(axis='x')
plt.show()


# # Solution
# 
# 1. iPhone and Lightning Charging Cable: Sold together 1002 times.
# 2. Google Phone and USB-C Charging Cable: Sold together 985 times.
# 3. iPhone and Wired Headphones: Sold together 447 times.
# 4. Google Phone and Wired Headphones: Sold together 413 times.
# 5. Vareebadd Phone and USB-C Charging Cable: Sold together 361 times.

# # Problem 5: What product sold the most ?

# In[116]:


# Knowing highest sold products

# Group by product and sum the quantity ordered
products= df.groupby('Product')
quantity= products.sum()['Quantity Ordered'].sort_values(ascending=False).reset_index()

print(quantity)


# In[96]:


# Plotting bar chart for most sold products

plt.figure(figsize=(10,6))

# Sort the DataFrame by 'Quantity Ordered' in descending order
quantity = quantity.sort_values(by='Quantity Ordered', ascending=True)

plt.barh(quantity['Product'],quantity['Quantity Ordered'],color="teal")
plt.xlabel("Quantity Ordered")
plt.ylabel("Products")
plt.title("Most sold Products")
plt.grid(axis='x')
plt.show()


# # Solution
# 
# Best-Selling Product: From the provided data, the product with the highest quantity ordered is AAA Batteries (4-pack), totaling 30,986 units sold.
# 
# Top Selling Products:
# 
# Based on quantity ordered, the top-selling products are:
# AAA Batteries (4-pack): 30,986 units
# AA Batteries (4-pack): 27,615 units
# USB-C Charging Cable: 23,931 units
# Lightning Charging Cable: 23,169 units
# Wired Headphones: 20,524 units

# # Problme 6 : Which cities have sold the most products?

# In[103]:


#Heatmap showing quantity ordered for City Vs Products

# Aggregate counts of each (City, Product) combination
heatmap_data = df.groupby(['City', 'Product']).size().reset_index(name='Count')

# Pivot the dataframe to have City as rows, Product as columns, and Count as values
heatmap_pivot = heatmap_data.pivot(index='City', columns='Product', values='Count')

# Plotting the heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_pivot, annot=True, cmap='YlGnBu', fmt='g',annot_kws={"fontsize": 6}) 
plt.title('Heatmap of quantity for City vs Product')
plt.xlabel('Product')
plt.ylabel('City')
plt.show()


# # Solution: 
# From the quantity heatmap:
# 
# Los Angeles appears to have high sales quantities across various products, especially for high-value items like "Macbook Pro Laptop," "Lightning Charging Cable," and "iPhone."
# 
# San Francisco also shows significant quantities, particularly for "Macbook Pro Laptop," "AAA Batteries (4-pack)," "Lightning Charging Cable," and "iPhone."
# 
# New York City has high sales in categories such as "Macbook Pro Laptop," "Lightning Charging Cable," and "iPhone."

# # Problem 7 : Which cities have sold the highest amount of each product?

# In[111]:


#Heatmap showing Sales for City Vs Products
# Sum of sales of each (City, Product) combination
heatmap_data1 = df.groupby(['City', 'Product'])['Sales'].sum().reset_index()

# Pivot the dataframe to have City as rows, Product as columns, and Count as values
heatmap_pivot1 = heatmap_data1.pivot(index='City', columns='Product', values='Sales')

# Convert sales to millions for annotation
heatmap_pivot1= (heatmap_pivot1 / 1e6).round(2)  # Divide by 1 million to convert to million

# Plotting the heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_pivot1, annot=True, cmap='YlGnBu', fmt='g',annot_kws={"fontsize": 8})  
plt.title('Heatmap of Sales for City vs Product')
plt.xlabel('Product')
plt.ylabel('City')
plt.show()


# # Solution: 
# 
# For each product, the city with the darkest color in the heatmap indicates the highest quantity sold.
# 
# 20in Monitor: San Francisco (993 units)
# 
# 27in 4K Gaming Monitor: Los Angeles (1003 units)
# 
# 27in FHD Monitor: San Francisco (1454 units)
# 
# 34in Ultrawide Monitor: San Francisco (1799 units)
# 
# AA Batteries (4-pack): San Francisco (4897 units)
# 
# AAA Batteries (4-pack): San Francisco (4928 units)
# 
# Apple Airpods Headphones: New York City (2768 units)
# 
# Bose SoundSport Headphones: New York City (2775 units)
# 
# Flatscreen TV: Los Angeles (3305 units)
# 
# Google Phone: Los Angeles (2447 units)
# 
# LG Dryer: San Francisco (3282 units)
# 
# LG Washing Machine: San Francisco (3184 units)
# 
# Lightning Charging Cable: San Francisco (5157 units)
# 
# Macbook Pro Laptop: San Francisco (1133 units)
# 
# ThinkPad Laptop: San Francisco (5357 units)
# 
# USB-C Charging Cable: New York City (2708 units)
# 
# Vareebadd Phone: Los Angeles (3448 units)
# 
# Wired Headphones: New York City (881 units)
# 
# iPhone: San Francisco (1659 units)
