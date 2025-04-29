import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

f_id = open('25k_run.gpx')

# calculate average pace
latitude_vector = []
longitude_vector = []
time_vector = []

# loop 

for line in f_id:
    if len(line) > 3:
        c_space = line.replace(" ", "")
        c_newline = c_space.replace("\n", "")
        print(c_space)
        print(c_newline)
        if "<trkseg" in c_newline:
            print(c_newline)
            row = c_newline.split(' ')
            # print(row)
            """
            lat = float(row[1])
            lon = float(row[3])
            latitude_vector.append(lat)
            longitude_vector.append(lon)
            """