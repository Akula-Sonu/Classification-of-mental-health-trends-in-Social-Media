# Hadoop Streaming reducer: sums counts per key (year_month + label)
import sys
from collections import defaultdict
counts = defaultdict(int)

for line in sys.stdin:
    parts = line.strip().split("\t")
    if len(parts) != 3:
        continue
    key = f"{parts[0]}\t{parts[1]}"
    try:
        counts[key] += int(parts[2])
    except ValueError:
        continue

for key, val in counts.items():
    print(f"{key}\t{val}")
