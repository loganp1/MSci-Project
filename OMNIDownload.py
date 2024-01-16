import urllib3
import csv
import re
import pandas as pd

# Create an instance of the urllib3 PoolManager
http = urllib3.PoolManager()

# Specify the URLs and local file paths for the two files
url1 = input("Paste data file URL from OMNIWeb: ")
fname1 = input("Enter desired filename: ")
fname1+=".txt"

url2 = input("Paste format file URL from OMNIWeb: ")  # Replace with the URL for the second file
fname2 = 'format.txt'  # Replace with the desired local file path for the second file
"\n"
# Helper function to download and save data from a URL to a file
def download_and_save(url, local_file_path):
    try:
        # Send an HTTP GET request to the URL
        response = http.request('GET', url)

        # Check if the request was successful (status code 200)
        if response.status == 200:
            # Open the local file in binary write mode and save the downloaded data
            with open(local_file_path, 'wb') as local_file:
                local_file.write(response.data)
            print(f"Data downloaded and saved to '{local_file_path}'")
        else:
            print(f"Failed to download data. Status code: {response.status}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Download and save the first file
download_and_save(url1, fname1)

# Download and save the second file
download_and_save(url2, fname2)





# Define a function to parse data from a file and save it to a CSV file
def parse_and_save_data(input_file, output_file):
    data = []

    with open(input_file, "r") as file:
        for line in file:
            columns = re.split(r'\s+', line.strip())
            data.append(columns)

    # Save the data to a CSV file
    with open(output_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)


# Define the output CSV file paths based on the input file names
csvname = fname1.replace(".txt", ".csv")


# Parse and save data from the first file
parse_and_save_data(fname1,csvname)

# Parse and save data from the second file


print(f"Data from {fname1} saved as {csvname}")



# Create a list to store the values from the second column
second_column_values = []

# Open the text file for reading
def extract_second_column_values(file_path):
    # Create a list to store the values from the second column
    second_column_values = []

    # Open the text file for reading
    with open(file_path, "r") as file:
        for line in file:
            # Split the line into words based on whitespace
            words = line.split()

            # Check if the line contains at least two words (to avoid errors)
            if len(words) >= 2:
                # Extract the second column value as a combination of words (excluding the last word)
                second_column = ' '.join(words[1:len(words)-1])
                second_column_values.append(second_column)

    return second_column_values[2:]

colTitles=extract_second_column_values("format.txt")
df = pd.read_csv(csvname, header=None)
df.columns=colTitles
from datetime import datetime, timedelta
def UnixConv(df, seconds=True, minutes=True,DayVar="Day",YearVar="Year"):
    if seconds==False:
        column_name = 'Seconds'
        df.insert(4, column_name, 0)
    if minutes==False:
        column_name = 'Minute'
        df.insert(3, column_name, 0)
    def convert_to_unix_timestamp(row):
        year = int(row[YearVar].item())
        day = int(row[DayVar].item())
        hour = int(row['Hour'].item())
        minute = int(row['Minute'].item())
        second = int(row['Seconds'].item())
        date = datetime(year, 1, 1) + timedelta(days=day - 1, hours=hour, minutes=minute, seconds=second)
        unix_timestamp = int((date - datetime(1970, 1, 1)).total_seconds())
        return unix_timestamp

    df['Time'] = df.apply(convert_to_unix_timestamp, axis=1)
    df.drop(columns=[YearVar, DayVar, 'Hour', 'Minute', 'Seconds'], inplace=True)
    df = df[['Time'] + [col for col in df.columns if col != 'Time']]
    return df
secs=input("Seconds present?, Y/N: ")
if secs=="N":
    secs=False
else:
    secs=True
    
minutes=input("Minutes present?, Y/N: ")
if minutes=="N":
    minutes=False
else:
    minutes=True

df=UnixConv(df,secs, minutes)
#df=UnixConv(df,secs)
df.to_csv(csvname, index=False)

print("CSV save complete")







