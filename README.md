# 🏃 Run Analyzer
A Python tool to analyze running data (heart rate, pace, cadence) from `.tcx` files — useful for performance tracking and coaching feedback.
---
## 📂 Overview

- Parses `.tcx` files from GPS devices (e.g. Garmin, COROS).
- Compares two runs to assess changes in performance.
- Visualizes time-series metrics like heart rate and pace.
- Designed to support future coaching logic with LLMs.

---
## ✅ Requirements

- Python 3.x
- Dependencies:
  - `numpy`
  - `matplotlib`
  - `xml.etree.ElementTree` (built-in)

Install required libraries:
```bash
pip install numpy matplotlib
