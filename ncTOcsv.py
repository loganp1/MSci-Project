import netCDF4 as nc
import csv
import matplotlib.pyplot as plt
import pandas as pd

def nc_to_csv(nc_filename, csv_filename):
    try:
        # Open the NetCDF file
        dataset = nc.Dataset(nc_filename)

        # Get the names of all variables and convert to a list
        variable_names = list(dataset.variables.keys())

        # Create a CSV file for writing
        with open(csv_filename, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Write variable names as the header row in the CSV file
            writer.writerow(variable_names)

            # Get the number of records (assuming all variables have the same length)
            num_records = len(dataset.variables[variable_names[0]])

            # Iterate through the records and write data to the CSV file
            for i in range(num_records):
                row_data = [dataset.variables[var_name][i] for var_name in variable_names]
                writer.writerow(row_data)

        print(f"Data from '{nc_filename}' successfully saved to '{csv_filename}'.")

    except Exception as e:
        print(f"Error: {str(e)}")

    finally:
        # Close the NetCDF dataset
        if dataset:
            dataset.close()

"""

# Example usage:
nc_file = 'dataInit.nc'  # Replace with the path to your NetCDF file
csv_file = 'output3.csv'    # Replace with the desired output CSV file name
nc_to_csv(nc_file, csv_file)
"""
ncname=input("Enter name of .nc file: ")
csvname=input("Enter desired csv file name: ")
nc_to_csv(ncname, csvname)
