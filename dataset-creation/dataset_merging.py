import pandas as pd
import os

# List of CSV file names
csv_files = [
    'aita-datafiles\\2018\Reddit_AITA_2018_Raw.csv',
    'aita-datafiles\\2019\Reddit_AITA_2019_Raw.csv',
    'aita-datafiles\\2020\Reddit_AITA_2020_Raw.csv',
    'aita-datafiles\\2021\Reddit_AITA_2021_Raw.csv',
    'aita-datafiles\\2022\Reddit_AITA_2022_Raw.csv'
]

# Initialize an empty list to store the DataFrames
dataframes = []

# Loop through each CSV file
for file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    dataframes.append(df)

# Concatenate all DataFrames vertically
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('aita-datafiles\Reddit_AITA_2018_to_2022.csv', index=False)