import csv
import os

# Read the first CSV file and get a column
with open('1_bench_gcn.csv', 'r') as file1:
    reader = csv.reader(file1)
    next(reader)
    col0 = [row[0] for row in reader]

with open('../1_bench_gcn.csv', 'r') as file1:
    reader = csv.reader(file1)
    next(reader)
    col1 = [float(row[1]) for row in reader]

# Read the second CSV file and get a column
with open('1_bench_gcn.csv', 'r') as file2:
    reader = csv.reader(file2)
    next(reader)
    col2 = [float(row[1]) for row in reader]
# print(col2)

# Compute the ratio of the values in the two columns
ratio = [ col2[i]/col1[i] for i in range(len(col1))]

# Write the result, along with the original two columns, to a third CSV file
with open('Fig_6b_PyG_gcn.csv', 'w', newline='') as result_file:
    writer = csv.writer(result_file)
    writer.writerow(['Dataset', 'PyG(ms)', 'TC-GNN(ms)', 'Speedup (x)'])
    for i in range(len(col0)):
        writer.writerow([col0[i], col2[i], col1[i], "{:.3f}".format(ratio[i])])

print("\n\n=>Please check [Fig_6b_PyG_gcn.csv] for the results.\n\n")
# os.system("mv *.err logs/")




# Read the first CSV file and get a column
with open('1_bench_agnn.csv', 'r') as file1:
    reader = csv.reader(file1)
    next(reader)
    col0 = [row[0] for row in reader]

with open('../1_bench_agnn.csv', 'r') as file1:
    reader = csv.reader(file1)
    next(reader)
    col1 = [float(row[1]) for row in reader]

# Read the second CSV file and get a column
with open('1_bench_agnn.csv', 'r') as file2:
    reader = csv.reader(file2)
    next(reader)
    col2 = [float(row[1]) for row in reader]
# print(col2)

# Compute the ratio of the values in the two columns
ratio = [ col2[i]/col1[i] for i in range(len(col1))]

# Write the result, along with the original two columns, to a third CSV file
with open('Fig_6b_PyG_agnn.csv', 'w', newline='') as result_file:
    writer = csv.writer(result_file)
    writer.writerow(['Dataset', 'PyG(ms)', 'TC-GNN(ms)', 'Speedup (x)'])
    for i in range(len(col0)):
        writer.writerow([col0[i], col2[i], col1[i], "{:.3f}".format(ratio[i])])

print("=>Please check [Fig_6b_PyG_agnn.csv] for the results.\n\n")