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

# i/o
input_dir = 'coros'  # directory with TCX files
input_file = 'training_summary.csv'

# add ActivityExtension namespace
ns = {
    'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
    'ae':  'http://www.garmin.com/xmlschemas/ActivityExtension/v2'
}

# sort out weekly load first (mon-sun) each week from data availability
# now the fun part is i will set the approach or coaching method to gauge the load  **
# from parse file -> summery it should easy #TODO