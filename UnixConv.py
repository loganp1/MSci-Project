import pandas as pd
from datetime import datetime, timedelta

def convert_time_columns_to_unix_seconds(df, seconds=True,DayVar="Day",YearVar="Year"):
    if seconds==False:
        column_name = 'Seconds'
        df.insert(5, column_name, 0)
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

fname="DSCOVRBData.csv"
df = pd.read_csv("DSCOVRBData.csv")
df2=pd.read_csv("ARTEMISBData.csv")



# Convert time columns to Unix seconds
df = convert_time_columns_to_unix_seconds(df,seconds=False)
df.to_csv(fname,index=0)





print(df)
