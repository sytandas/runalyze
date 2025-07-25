"""
This is simple analysis of two tcx file. 
"""
import math, sys
from datetime import datetime
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: python3 ana.py <file1.tcx> <file2.tcx>")
    sys.exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]
tree_1 = ET.parse(file1).getroot()
tree_2 = ET.parse(file2).getroot()


# TCX files use a default namespace
ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

def file_extract(x):
    latitude_vector = []
    longitude_vector = []
    distance_vector = []
    heart_rate_vector = []
    pace_vector = []
    time_vector = []
    cadence_vector = []

    prev_time = None
    prev_distance = None
    activity_date = None

    for tp in x.findall('.//tcx:Trackpoint', ns):
        # Time
        time_el = tp.find('tcx:Time', ns)
        if time_el is not None:
            current_time = datetime.fromisoformat(time_el.text.replace('Z', '+00:00'))
            time_vector.append(current_time)
            if len(time_vector) == 1:
                activity_date = current_time.date()
        else:
            continue  # skip this point if no time

        # Position
        lat_el = tp.find('tcx:Position/tcx:LatitudeDegrees', ns)
        lon_el = tp.find('tcx:Position/tcx:LongitudeDegrees', ns)
        lat = float(lat_el.text) if lat_el is not None else None
        lon = float(lon_el.text) if lon_el is not None else None
        latitude_vector.append(lat)
        longitude_vector.append(lon)

        # Distance
        dist_el = tp.find('tcx:DistanceMeters', ns)
        dist = float(dist_el.text) if dist_el is not None else None
        distance_vector.append(dist)

        # HR
        hr_el = tp.find('tcx:HeartRateBpm/tcx:Value', ns)
        hr = int(hr_el.text) if hr_el is not None else None
        heart_rate_vector.append(hr)

        # Cadence
        cad_el = tp.find('tcx:Cadence', ns)
        cadence = int(cad_el.text) if cad_el is not None else None
        cadence_vector.append(cadence)

        # Pace
        if prev_time is not None and prev_distance is not None and dist is not None:
            delta_t = (current_time - prev_time).total_seconds() / 60.0
            delta_d = (dist - prev_distance) / 1000.0
            pace = delta_t / delta_d if delta_d > 0 else None
            pace_vector.append(pace)
        else:
            pace_vector.append(None)

        prev_time = current_time
        prev_distance = dist

    # Summary
    total_distance_km = (distance_vector[-1] if distance_vector[-1] is not None else 0) / 1000 
    hr_values = [hr for hr in heart_rate_vector if hr is not None]
    avg_hr = np.mean(hr_values) if hr_values else None
    max_hr = np.max(hr_values) if hr_values else None

    cad_values = [c for c in cadence_vector if c is not None]
    avg_cad = np.mean(cad_values) * 2 if cad_values else None

    pace_values = [p for p in pace_vector if p is not None and p < 20]
    avg_pace = np.mean(pace_values) if pace_values else None
    pace_min = int(avg_pace) if avg_pace else 0
    pace_sec = int((avg_pace - pace_min) * 60) if avg_pace else 0

    total_time_min = (
        (time_vector[-1] - time_vector[0]).total_seconds() / 60.0 if time_vector else 0
    )

    print(f"Date of the activity: {activity_date}")
    print(f"Total distance: {total_distance_km:.2f} km")
    print(f"Total time: {total_time_min:.1f} min")
    print(f"Average pace: {pace_min}:{pace_sec:02d} min/km" if avg_pace else "Average pace: N/A")
    print(f"Average heart rate: {avg_hr:.0f} bpm" if avg_hr else "Heart rate: N/A")
    print(f"Max hr: {max_hr:.0f} bpm" if max_hr else "Max hr: N/A")
    print(f"Average cadence: {avg_cad:.0f} spm" if avg_cad else "Cadence: N/A")

    return {
        'time': time_vector,
        'distance': distance_vector,
        'heart_rate': heart_rate_vector,
        'pace': pace_vector,
        'cadence': cadence_vector
    }

data_1 = file_extract(tree_1)
print("")
data_2= file_extract(tree_2)

# Implementation of dynamic time warping (DTW) to compare two runs changing fintess measures.  
def dtw(s1_og, s2_og):
    s1 = np.array([x for x in s1_og if x is not None])
    s2 = np.array([x for x in s2_og if x is not None])

    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0
    path = {}

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s1[i-1] - s2[j-1])
            choices = [
                (dtw_matrix[i-1, j], (i-1, j)),         # insertion
                (dtw_matrix[i, j-1], (i, j-1)),         # deletion
                (dtw_matrix[i-1, j-1], (i-1, j-1))      # match
            ]
            min_cost, prev = min(choices, key=lambda x: x[0])
            dtw_matrix[i, j] = cost + min_cost
            path[(i, j)] = prev

    # reconstruction of path
    alignment_path = []
    i, j = m, n
    while(i, j) in path:
        alignment_path.append((i-1, j-1)) # adjust for 0-based indexing
        i, j = path[(i, j)]
    alignment_path.reverse()

    return dtw_matrix[n, m], alignment_path, s1, s2

# visualization 
def plot_dtw_alignment(s1_og, s2_og, label='Metric'):
    distance, path, s1, s2 = dtw(s1_og, s2_og)
    plt.figure(figsize=(10, 6))
    plt.plot(s1, label=f'{label} - Run 1', color="red")
    plt.plot(s2, label=f'{label} - Run 2', color="green")

    for i, j in path[::max(1, len(path) // 100)]:
        plt.plot([i, j], [s1[i], s2[j],], color='grey', linewidth=0.5, alpha=0.5)

    plt.title(f'DTW alignment on {label} \n DTW distance: {distance: .2f}')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(label)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# plot for pace, hr, cadence etc  
plot_dtw_alignment(data_1['cadence'], data_2['cadence'], label="cadence")
plot_dtw_alignment(data_1['heart_rate'], data_2['heart_rate'], label="heart_rate")   
plot_dtw_alignment(data_1['pace'], data_2['pace'], label="pace")

# TODO: Analyzing the ploting how the fitness improved e.g. low hr at same pace, high cadence ~ efficiency.
# TODO: Visualize trends and optionally apply DTW for route-based comparison or finding other algorithm to do. 
# TODO: One line spell from a model