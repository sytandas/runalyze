## 🏃 runalyze
Analyzing run from heart rate, pace and cadence.
Uses .tcx file from (e.g. garmin, coros etc).

TCX File Parser and analyzer (ana.py):
ana.py takes two `.tcx` files as input, parses them and plot cadence, heart rate, pace.

```bash
python3 ana.py <file1.tcx> <file2.tcx>
```

Coacing insight (model.py):
After analyzing attempt to improve on it with AI coaching (todo).

## 📦 install
```
pip install numpy matplotlib
```

## ✅ Todo:
1. Load balancing
2. Dew point effect on pace
