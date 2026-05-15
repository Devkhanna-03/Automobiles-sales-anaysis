import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("automobile_data.csv")


print("Dataset shape:", df.shape)
print("\nDataset Preview:\n", df.head())   
print("\nStatistical Summary:\n", df.describe())
print("\nNull Value Count:\n", df.isnull().sum())



# Remove the duplicate rows
df = df.drop_duplicates()

# Fill missing values with median
df = df.fillna(df.median(numeric_only=True))

# Convert year column to integer
df['Year'] = df['Year'].astype(int)

print("\nDataset After Cleaning:\n", df.head())


year_sales = df.groupby('Year')['Sales'].sum()

plt.figure()
year_sales.plot()
plt.title("Year-wise Automobile Sales Trend")
plt.xlabel("Year")
plt.ylabel("Sales (Units)")
plt.show()



state_sales = df.groupby('State')['Sales'].sum().sort_values(ascending=False)

plt.figure()
state_sales.plot(kind='bar')
plt.title("State-wise Vehicle Demand")
plt.xlabel("State")
plt.ylabel("Sales (Units)")
plt.show()



segment_share = df.groupby('Vehicle_Type')['Sales'].sum()

plt.figure()
plt.pie(segment_share, labels=segment_share.index, autopct='%1.1f%%')
plt.title("Vehicle Segment Market Share")
plt.show()



top_models = df.groupby('Model')['Sales'].sum().sort_values(ascending=False).head(10)

print("\nTop Selling Car Models:\n", top_models)



print("\n---- Conclusion Summary ----")

print("Highest Automobile Sales Year:", year_sales.idxmax(), "with", year_sales.max(), "units.")
print("Top Performing State:", state_sales.idxmax(), "with", state_sales.max(), "units.")
print("Most Demanded Segment:", segment_share.idxmax())
