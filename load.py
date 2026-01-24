"""
Central Idea: 
Load balance using HR each week.
Primarily Idea came from Banister Impulse-Response Model.
Training Load -> Fitness - Fatigue = Performance
How close become the breakthrough line graphically. 
"""

import math, sys
from datetime import datetime
import xml.etree.ElementTree as ET
from statistics import mean, variance
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: python3 ana.py <file1.tcx> <file2.tcx>")
    sys.exit(1)

tree_1 = ET.parse(sys.argv[1]).getroot()
tree_2 = ET.parse(sys.argv[2]).getroot()


# add ActivityExtension namespace
ns = {
    'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
    'ae':  'http://www.garmin.com/xmlschemas/ActivityExtension/v2'
}
