import os
import xml.etree.ElementTree as ET
from datetime import datetime
import csv
import numpy as np

# Namespace
ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

# Input/output
input_dir = 'coros'  # directory with TCX files
output_file = 'training_summary.csv'

# Header for the summary file
header = ['filename', 'date', 'sport_type', 'total_distance_km', 'duration_min',
          'avg_pace_min_per_km', 'avg_heart_rate_bpm', 'avg_cadence_spm']

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.tcx'):
            continue

        filepath = os.path.join(input_dir, filename)
        try:
            tree = ET.parse(filepath)
        except Exception as e:
            print(f"Failed to parse {filename}: {e}")
            continue

        root = tree.getroot()

        # Extract sport type from the first <Activity>
        activity_el = root.find('.//tcx:Activity', ns)
        sport_type = activity_el.attrib.get('Sport') if activity_el is not None else "Unknown"

        time_vector = []
        distance_vector = []
        heart_rate_vector = []
        cadence_vector = []
        pace_vector = []

        prev_time = None
        prev_distance = None

        for tp in root.findall('.//tcx:Trackpoint', ns):
            time_el = tp.find('tcx:Time', ns)
            if time_el is None:
                continue
            current_time = datetime.fromisoformat(time_el.text.replace('Z', '+05:30'))
            time_vector.append(current_time)

            dist_el = tp.find('tcx:DistanceMeters', ns)
            dist = float(dist_el.text) if dist_el is not None else None
            distance_vector.append(dist)

            hr_el = tp.find('.//tcx:HeartRateBpm/tcx:Value', ns)
            hr = int(hr_el.text) if hr_el is not None else None
            heart_rate_vector.append(hr)

            cad_el = tp.find('tcx:Cadence', ns)
            cad = int(cad_el.text)*2 if cad_el is not None else None
            cadence_vector.append(cad)

            # Skip if time or distance did not increase - new correction *********************
            if prev_time is not None and current_time <= prev_time:
                continue
            # end **************************
            if prev_time is not None and prev_distance is not None and dist is not None:
                delta_t = (current_time - prev_time).total_seconds() / 60.0
                delta_d = (dist - prev_distance) / 1000.0
                if delta_d > 0 and delta_t > 0:
                    pace = delta_t / delta_d
                    if 2 < pace < 20:
                        pace_vector.append(pace)

            prev_time = current_time
            # new correction *********************
            if prev_distance is not None and dist is not None and dist <= prev_distance:
                continue
            # end ***********************
            prev_distance = dist

        if not time_vector or not distance_vector:
            continue

        # sanity check if works - start ******************************
        delta_t = (current_time - prev_time).total_seconds()
        # delta_d = (dist - prev_distance)
        if dist is not None and prev_distance is not None:
            delta_d = dist - prev_distance
        else:
            delta_d = None  # or some default, like 0, depending on your use case


        # Skip large gaps or zero movement
        if delta_t <= 0 or delta_d <= 0 or delta_t > 120 or delta_d > 500:
            pace_vector.append(None)
        else:
            pace = (delta_t / 60.0) / (delta_d / 1000.0)
            pace_vector.append(pace)
        # sanity check if works - end *************************
        
        total_distance_km = (distance_vector[-1] if distance_vector[-1] else 0) / 1000
        total_time_min = (time_vector[-1] - time_vector[0]).total_seconds() / 60.0
        # Cleaned lists (excluding None)
        clean_hr = [h for h in heart_rate_vector if h is not None]
        clean_cad = [c for c in cadence_vector if c is not None]
        clean_pace = [p for p in pace_vector if p is not None and p < 20]
        # Safe averaging
        avg_hr = np.mean(clean_hr) if clean_hr else None
        avg_cad = np.mean(clean_cad) if clean_cad else None
        # avg_pace = np.mean(pace_vector) if pace_vector else None
        # new - start ******************************
        cleaned_pace_vector = [p for p in pace_vector if p is not None]
        avg_pace = np.mean(cleaned_pace_vector) if cleaned_pace_vector else None
        # new - end ******************************
        activity_date = time_vector[0].date() 

        writer.writerow([
            filename,
            activity_date.isoformat(),
            sport_type,
            f"{total_distance_km:.2f}",
            f"{total_time_min:.1f}",
            f"{avg_pace:.2f}" if avg_pace is not None else "",
            f"{avg_hr:.0f}" if avg_hr else "",
            f"{avg_cad:.0f}" if avg_cad else ""
        ])

print('Done')