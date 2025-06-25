"""
Parsed files and plot fitness improvement.
TODO: 1 line comment about two runs.
"""
import os
from datetime import datetime
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TCX namespace
ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

# Direcotry of file extraction function
def file_extract(root):
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

    for tp in root.findall('.//tcx:Trackpoint', ns):
        time_el = tp.find('tcx:Time', ns)
        if time_el is not None:
            current_time = datetime.fromisoformat(time_el.text.replace('Z', '+00:00'))
            time_vector.append(current_time)
            if len(time_vector) == 1:
                activity_date = current_time.date()
        else:
            continue

        lat_el = tp.find('tcx:Position/tcx:LatitudeDegrees', ns)
        lon_el = tp.find('tcx:Position/tcx:LongitudeDegrees', ns)
        lat = float(lat_el.text) if lat_el is not None else None
        lon = float(lon_el.text) if lon_el is not None else None
        latitude_vector.append(lat)
        longitude_vector.append(lon)

        dist_el = tp.find('tcx:DistanceMeters', ns)
        dist = float(dist_el.text) if dist_el is not None else None
        distance_vector.append(dist)

        hr_el = tp.find('tcx:HeartRateBpm/tcx:Value', ns)
        hr = int(hr_el.text) if hr_el is not None else None
        heart_rate_vector.append(hr)

        cad_el = tp.find('tcx:Cadence', ns)
        cadence = int(cad_el.text) if cad_el is not None else None
        cadence_vector.append(cadence)

        if prev_time and prev_distance and dist is not None:
            delta_t = (current_time - prev_time).total_seconds() / 60.0
            delta_d = (dist - prev_distance) / 1000.0
            pace = delta_t / delta_d if delta_d > 0 else None
            pace_vector.append(pace)
        else:
            pace_vector.append(None)

        prev_time = current_time
        prev_distance = dist

    if not time_vector or not distance_vector:
        return None

    total_distance_km = (distance_vector[-1] if distance_vector[-1] else 0) / 1000
    total_time_min = (time_vector[-1] - time_vector[0]).total_seconds() / 60 if time_vector else 0

    hr_values = [hr for hr in heart_rate_vector if hr is not None]
    avg_hr = np.mean(hr_values) if hr_values else None
    max_hr = np.max(hr_values) if hr_values else None

    cad_values = [c for c in cadence_vector if c is not None]
    avg_cad = np.mean(cad_values) * 2 if cad_values else None

    pace_values = [p for p in pace_vector if p is not None and p < 20]
    avg_pace = np.mean(pace_values) if pace_values else None

    return {
        'activity_date': activity_date,
        'total_distance_km': total_distance_km,
        'total_time_min': total_time_min,
        'avg_pace_min_per_km': avg_pace,
        'avg_hr': avg_hr,
        'max_hr': max_hr,
        'avg_cadence': avg_cad,
        'time_series': {
            'time': time_vector,
            'distance': distance_vector,
            'heart_rate': heart_rate_vector,
            'pace': pace_vector,
            'cadence': cadence_vector
        }
    }

# Loding tcx directory
folder_path = 'coros/' 
tcx_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tcx')]

workouts = []
for f in sorted(tcx_files):  # sort for time consistency
    try:
        root = ET.parse(f).getroot()
        result = file_extract(root)
        if result:
            print(f"Parsed: {f}")
            workouts.append(result)
        else:
            print(f"Skipped: {f}")
    except Exception as e:
        print(f"Error in {f}: {e}")

# Build DataFrame
summary_df = pd.DataFrame([{
    'date': w['activity_date'],
    'distance_km': w['total_distance_km'],
    'duration_min': w['total_time_min'],
    'avg_pace': w['avg_pace_min_per_km'],
    'avg_hr': w['avg_hr'],
    'max_hr': w['max_hr'],
    'avg_cadence': w['avg_cadence']
} for w in workouts if w and w['avg_pace_min_per_km'] and w['avg_hr'] and w['avg_cadence']])

print("Summary DF shape:", summary_df.shape)
print(summary_df)

# Normalize metrics
summary_df['norm_pace'] = (summary_df['avg_pace'].max() - summary_df['avg_pace']) / (summary_df['avg_pace'].max() - summary_df['avg_pace'].min())
summary_df['norm_hr'] = (summary_df['avg_hr'].max() - summary_df['avg_hr']) / (summary_df['avg_hr'].max() - summary_df['avg_hr'].min())
summary_df['norm_cadence'] = (summary_df['avg_cadence'] - summary_df['avg_cadence'].min()) / (summary_df['avg_cadence'].max() - summary_df['avg_cadence'].min())

# Composite fitness score adjust weight as needed
summary_df['fitness_score'] = summary_df[['norm_pace', 'norm_hr', 'norm_cadence']].mean(axis=1)


# Improvement in HR efficiency
summary_df['pace_per_hr'] = summary_df['avg_pace'] / summary_df['avg_hr']
x = summary_df['pace_per_hr_diff'] = summary_df['pace_per_hr'].diff()
print(x)


# Or use correlation to see how HR relates to pace
y = summary_df[['avg_pace', 'avg_hr']].corr()
print(y)

# Cli output of improvement:: 
print("\nSession-wise HR Efficiency Trend:")
for i in range(1, len(summary_df)):
    date = summary_df['date'].iloc[i]
    diff = summary_df['pace_per_hr_diff'].iloc[i]
    status = (
        "↑ Improved" if diff < -0.001 else 
        "↓ Declined" if diff > 0.001 else 
        "→ Stable"
    )
    print(f"{date}: HR efficiency change {diff:.4f} => {status}")

# Generate one liner AI-style insight from improvement in pace/hr efficiency
latest_improvement = summary_df['pace_per_hr_diff'].iloc[-1]

if latest_improvement < -0.001:
    insight = "Heart rate efficiency improved—you're running faster per unit effort. Great progress!"
elif latest_improvement > 0.001:
    insight = "Heart rate efficiency slightly declined—consider recovery or checking fatigue."
else:
    insight = "Heart rate efficiency stable—consistency is key, keep it up!"

print("AI Insight:", insight)

# Plot fitness trend 
plt.figure(figsize=(10, 5))
plt.plot(summary_df['date'], summary_df['fitness_score'], marker='o')
plt.title("Fitness Improvement Over Time")
plt.xlabel("Date")
plt.ylabel("Fitness Score")
plt.grid(True)
plt.tight_layout()
#plt.show()