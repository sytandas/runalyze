import math
from datetime import datetime

f_id = open('test_data/468659183674228748.gpx')

latitude_vector = []
longitude_vector = []
time_vector = []
ele_vector=[]
date_vector=[]

for line in f_id:
    if len(line) > 3:
        c_space = line.replace(" ","")
        c_newline = c_space.replace("\n","")
        if "<trkpt" in c_newline:
            row = c_newline.split('"')
            lat = float(row[1])
            lon = float(row[3])
            latitude_vector.append(lat)
            longitude_vector.append(lon)
        
        if "<time>" in c_newline:
            time_str = c_newline.strip().split("<time>")[1].split("</time>")[0]
            timestamp = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
            time_vector.append(timestamp)
            date_vector = timestamp.date().isoformat()
        

# haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371 # earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2 
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1- a))
    return R * c

# distance and time 
total_distance = 0.0 #km 
total_time_minutes = (time_vector[-1] - time_vector[0]).total_seconds() / 60.0

for i in range(1, len(latitude_vector)):
    total_distance += haversine(
        latitude_vector[i-1], longitude_vector[i-1],
        latitude_vector[i], longitude_vector[i]
    )

# pace 
if total_distance > 0:
    pace_min_per_km = total_time_minutes / total_distance
    pace_min = int(pace_min_per_km)
    pace_sec = int((pace_min_per_km - pace_min) * 60)
    speed_kmh = total_distance / (total_time_minutes / 60.0)

    print(f"Date of activity: {date_vector}")
    print(f"Total distance: {total_distance:.2f} km")
    print(f"Total time: {total_time_minutes:.1f} minutes")
    print(f"Pace: {pace_min}:{pace_sec:02d} min/km")
    print(f"Speed: {speed_kmh:.2f} km/h")

