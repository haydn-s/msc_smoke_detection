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

# Define Ranges of tested Conditions
reflect_max = 2100
reflect_min = 1880

point_max = 4800
point_min = 3300
point_increment = 10

f1_scores = np.zeros((reflect_max - reflect_min, int((point_max - point_min) / point_increment)))
reflect_count = 0
point_count = 0

# Track Optimal Values
max_f1 = 0
optimal_reflectivity = 0
optimal_points = 0

# Process the Results
for reflect_thresh in range(reflect_min, reflect_max):
    reflect_thresh = reflect_thresh / 1000.0
    point_count = 0

    #for density_thresh in range(100, 220):
        #density_thresh = density_thresh / 100.0

    for point_thresh in range(point_min, point_max, point_increment):
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
                if float(row[6]) < reflect_thresh and int(row[5]) > point_thresh: #and int(row[5]) > 5000:#and float(row[12]) <= density_thresh: #and int(row[9][1]) == 2:
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
                if float(row[6]) < reflect_thresh and int(row[5]) > point_thresh: #and int(row[5]) > 5000:#and float(row[12]) <= density_thresh: #and int(row[9][1]) == 2:
                    num_smoke_clusters += 1

        precision = (true_positives / (false_positives + true_positives))
        recall = (true_positives / (true_positives + false_negatives))
        f1_score = 2 * (precision * recall) / (precision + recall)

        print('\nReflectivity Threshold: ', reflect_thresh)
        print('Total Packets: ', total_packets)
        print('Precision: ', recall)
        print('Recall: ', precision)
        print('F1 Score: ', f1_score, '\n')

        f1_scores[reflect_count][point_count] = f1_score
        point_count += 1

        if f1_score > max_f1:
            max_f1 = f1_score
            optimal_reflectivity = reflect_thresh
            optimal_points = point_thresh
    reflect_count += 1

# Output F1 Scores to File
f1_DF = pd.DataFrame(f1_scores)
f1_DF.to_csv("F1_scores_output.csv")

print('\nMaximum F1 Score: ', max_f1)
print('Optimal Reflectivity Threshold: ', optimal_reflectivity)
print('Optimal Points per Cluster Threshold: ', optimal_points)