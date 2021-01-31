#/usr/bin/env python3
import csv
import sys 


metrics = [
    # 'SOL L1/TEX Cache',
    # 'SOL L2 Cache',
    # 'SM [%]',
    # 'Achieved Occupancy',
    # 'Achieved Active Warps Per SM',
    'Max Bandwidth',
    'Memory Throughput'
]


if len(sys.argv) < 2:
    raise ValueError("Usage: ./prog [profile.csv]")

fname  = sys.argv[1]
metrics_val = {}
for met in metrics:
    metrics_val[met] = []

with open(fname, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader:
        metric_name, metric_val = lines[9], lines[11]
        if metric_name in metrics:
            val = metric_val.replace(',',"")
            metrics_val[metric_name].append(float(val))

for key in metrics_val.keys():
    val = sum(metrics_val[key]) / len(metrics_val[key])
    print("{:30}".format(key), "{:.2f}".format(val))