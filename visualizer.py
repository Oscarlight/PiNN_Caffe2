import numpy as np

def write_csv_file (file_name, data, description):
    '''' Write data to .csv file'''
    np.savetxt(file_name, data, delimiter=',', newline='\n', header=description)

