# Import Libraries
import cv2
import csv
import os
import numpy as np
import pandas as pd

# Define Filepaths
filename = 'processed_output_fixed.csv'
#filename = 'small_results.csv'
main_dir = r'D:/Data/'

# Parse all the Data
os.chdir(main_dir)

# Read CSV Data
myRows = []
# note: timestamp in nanoseconds.
try:
    print('\nReading File...\n')
    with open(filename, 'r') as myCSV:
        data = csv.reader(myCSV)
        next(data, None)
        row_counter = 0
        for row in data:
            myRows.append(row)

            if row_counter % 1000 == 0:
                print('Row #: ', row_counter)

            row_counter += 1

    myCSV.close()
    print('\nFile Read and Closed!')
except FileNotFoundError:
    print('\nNo File!')

total_packets = 1
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
curr_packet_num = int(myRows[0][1])
num_smoke_clusters = 0
last_known = myRows[0][17]

for row in myRows:
    if int(row[1]) == curr_packet_num:
        if float(row[6]) < 1.95 and int(row[5]) > 4230:
            num_smoke_clusters += 1
        last_known = int(row[17])
    else:
        if num_smoke_clusters >= 1 and last_known == 1:
            true_positives += 1
        elif num_smoke_clusters >= 1 and last_known == 0:
            false_positives += 1
        elif num_smoke_clusters == 0 and last_known == 0:
            true_negatives += 1
        elif num_smoke_clusters == 0 and last_known == 1:
            false_negatives += 1
        total_packets += 1
        curr_packet_num = int(row[1])
        last_known = int(row[17])
        num_smoke_clusters = 0
        if float(row[6]) < 1.95 and int(row[5]) > 4230:
            num_smoke_clusters += 1

precision = (true_positives / (false_positives + true_positives))
recall = (true_positives / (true_positives + false_negatives))
f1_score = 2 * (precision * recall) / (precision + recall)

print('\nReflectivity Threshold: ', reflect_thresh)
print('Total Packets: ', total_packets)
print('Precision: ', recall)
print('Recall: ', precision)
print('F1 Score: ', f1_score, '\n')