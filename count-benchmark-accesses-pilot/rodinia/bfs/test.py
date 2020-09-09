import json
import sys

with open('manual.json') as f:
    data = json.load(f)
    json.dump(data, sys.stdout, indent=2)
