# Run analyzing tool.
Analyzing run from heart rate, pace and cadence.
Uses .tcx file from (e.g. garmin, coros etc).
# TCX File Parser (ana.py)
This is a simple Python script that takes two `.tcx` files as input, parses them.
## 📦 Requirements
- Python 3.x
- Numpy, matplotlib and built-in and xml.etree.ElementTree.
### Command-line format:
```bash
python ana.py <file1.tcx> <file2.tcx>

## Todo:
1. Compare two runs as fitness change metrics.
2. Evaluate easy, threshold pace from overall data (finding it relatively correctly from least amount of data will be the goal).
3. Training load/effect - coaching part using LLMs.
