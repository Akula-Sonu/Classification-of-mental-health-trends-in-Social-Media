# Hadoop Streaming mapper: reads predictions CSV lines and emits "year_month\tlabel\t1"
import sys, csv
reader = csv.reader(sys.stdin)
header = None
for row in reader:
    if not row:
        continue
    if header is None:
        header = [c.strip().lower() for c in row]
        # expect "year_month" and "predicted_label"
        continue
    rowmap = {header[i]: row[i] if i < len(row) else "" for i in range(len(header))}
    ym = rowmap.get("year_month","").strip()
    label = rowmap.get("predicted_label","").strip()
    if ym and label:
        print(f"{ym}\t{label}\t1")
