# Import Libraries
import numpy as np
import cv2
import csv
import os
from sklearn.cluster import MeanShift as ms
from sklearn.cluster import estimate_bandwidth
import pandas as pd

# Define Constants
packet_size = 100000

# Define Filepaths
data_dir = r'D:/Data/raw_data/'
main_dir = r'D:/Data/'

# Create Output CSV File
os.chdir(main_dir)

with open('processed_output.csv', 'w', newline = '') as finalCSV:
    filewriter = csv.writer(finalCSV, delimiter = ',')#, quotechar = '|', quoting = csv.QUOTE_MINIMAL)
    filewriter.writerow(('File Name', 'Packet Number', 'First Frame in Packet', 'Last Frame in Packet', 'Cluster Number', 'Number of Points in Cluster', \
        'Average Reflectivity', 'Minimum Reflectivity', '25th Percentile Reflectivity', 'Median Reflectivity', '75th Percentile Reflectivity', 'Maximum Reflectivity', \
        'Average Density', 'Cluster Center: X', 'Cluster Center: Y', 'Cluster Center: Z', 'Smoke Cluster', 'Should Packet have Smoke?'))

    # Loop through all Files
    os.chdir(data_dir)

    for file in os.listdir(data_dir):

        # Define Variables
        curr_packet = 0

        # Read CSV Data
        print('\nFilename: ' + file)
        myRows = []
        # note: timestamp in nanoseconds.
        try:
            print('Reading File...')
            with open(file, 'r') as myCSV:
                data = csv.reader(myCSV)
                next(data, None)
                row_counter = 0
                for row in data:
                    myRows.append(row)

                    if row_counter % 100000 == 0:
                        print('Row #: ', row_counter)

                    row_counter += 1

            myCSV.close()
            print('File Read and Closed!\n')
        except FileNotFoundError:
            print('No File!\n')

        # Iterate through Packets
        while (len(myRows) > ((curr_packet + 1) * packet_size)):    
            coords = []
            num_points = 0

            for num in range(packet_size):
                if int(myRows[num + (curr_packet * packet_size)][11]) > 0 and float(myRows[num + (curr_packet * packet_size)][8]) > 1.0 and int(myRows[num + (curr_packet * packet_size)][11]) <= 3:
                    num_points += 1

            curr_coords = np.zeros(shape = (num_points, 3))
            curr_reflectivity = np.zeros(shape = (num_points, 1))

            curr_point = 0
            max_x = 0
            max_y = 0
            max_z = 0

            # Create the Current Packet
            for num in range(packet_size):
                if int(myRows[num + (curr_packet * packet_size)][11]) > 0 and float(myRows[num + (curr_packet * packet_size)][8]) > 1.0 and int(myRows[num + (curr_packet * packet_size)][11]) <= 3:
                    curr_coords[curr_point] = [float(myRows[num + (curr_packet * packet_size)][8]), float(myRows[num + (curr_packet * packet_size)][9]), float(myRows[num + (curr_packet * packet_size)][10])]
                    curr_reflectivity[curr_point] = int(myRows[num + (curr_packet * packet_size)][11])
                    curr_point += 1
                    if float(myRows[num + (curr_packet * packet_size)][8]) > max_x:
                        max_x = float(myRows[num + (curr_packet * packet_size)][8])
                    if float(myRows[num + (curr_packet * packet_size)][9]) > max_y:
                        max_y = float(myRows[num + (curr_packet * packet_size)][9])
                    if float(myRows[num + (curr_packet * packet_size)][10]) > max_z:
                        max_z = float(myRows[num + (curr_packet * packet_size)][10])

            # Perform Mean Shift Clustering on the Current Packet
            bandwidth = estimate_bandwidth(curr_coords, quantile = 0.15, n_samples = num_points // 5)
            curr_cluster = ms(bandwidth = bandwidth, n_jobs = -1).fit(curr_coords)
            cluster_centers = curr_cluster.cluster_centers_
            cluster_labels = curr_cluster.labels_

            # Calculate Average Reflectivity per Cluster
            cluster_set = set(cluster_labels)

            cluster_reflectivities = np.zeros(shape = (len(cluster_set), 1))
            cluster_reflectivities_collection = [[] for _ in range(len(cluster_set))]
            points_per_cluster = np.zeros(shape = (len(cluster_set), 1))
            density_parameters = np.zeros(shape = (len(cluster_set), 6))
            cluster_densities = np.zeros(shape = (len(cluster_set), 1))

            for i in range(len(cluster_set)):
                density_parameters[i][1] = max_x
                density_parameters[i][3] = max_y
                density_parameters[i][5] = max_z

            print("Number of Clusters: ", len(cluster_set), '\n')

            for i in range(len(curr_coords)):
                cluster_reflectivities[cluster_labels[i]] += curr_reflectivity[i]
                cluster_reflectivities_collection[cluster_labels[i]].append(curr_reflectivity[i])
                points_per_cluster[cluster_labels[i]] += 1

                if curr_coords[i][0] > density_parameters[cluster_labels[i]][0]:
                    density_parameters[cluster_labels[i]][0] = curr_coords[i][0]

                if curr_coords[i][0] < density_parameters[cluster_labels[i]][1]:
                    density_parameters[cluster_labels[i]][1] = curr_coords[i][0]

                if curr_coords[i][1] > density_parameters[cluster_labels[i]][2]:
                    density_parameters[cluster_labels[i]][2] = curr_coords[i][1]

                if curr_coords[i][1] < density_parameters[cluster_labels[i]][3]:
                    density_parameters[cluster_labels[i]][3] = curr_coords[i][1]

                if curr_coords[i][2] > density_parameters[cluster_labels[i]][4]:
                    density_parameters[cluster_labels[i]][4] = curr_coords[i][2]

                if curr_coords[i][2] < density_parameters[cluster_labels[i]][5]:
                    density_parameters[cluster_labels[i]][5] = curr_coords[i][2]

            for i in range(len(cluster_set)):
                cluster_densities[i] = ((density_parameters[i][0] - density_parameters[i][1]) *\
                                        (density_parameters[i][2] - density_parameters[i][3]) *\
                                        (density_parameters[i][4] - density_parameters[i][5])) \
                                        / points_per_cluster[i]
                cluster_reflectivities[i] /= float(points_per_cluster[i])
                cluster_reflectivities_collection[i].sort()

            print("Finished the preparation loop!")
            print("\nDensity Parameters:")
            print(density_parameters)
            print("\nSummed Cluster Reflectivities:")
            print(cluster_reflectivities)
            print("\nPoints per Cluster:")
            print(points_per_cluster)
            print("\nCluster Densities:")
            print(cluster_densities)
            print("\nAverage Cluster Reflectivities:")
            print(cluster_reflectivities)
            print("\nFinished cluster calculations!\n")

            # Save the Result
            for i in range(len(cluster_centers)):
                smoke_detected = False
                if cluster_reflectivities[i] <= 1.7 and cluster_reflectivities[i] > 0:
                    if cluster_densities[i] <= 5 and cluster_densities[i] > 0:
                        smoke_detected = True

                if 'no' not in os.path.basename(file) and smoke_detected:
                    filewriter.writerow((os.path.basename(file), curr_packet, (curr_packet * packet_size) + 1, (curr_packet + 1) * packet_size, i, points_per_cluster[i][0], \
                        cluster_reflectivities[i][0], cluster_reflectivities_collection[i][0], cluster_reflectivities_collection[i][int(len(cluster_reflectivities_collection[i]) * 0.25)], \
                        cluster_reflectivities_collection[i][int(len(cluster_reflectivities_collection[i]) * 0.5)], \
                        cluster_reflectivities_collection[i][int(len(cluster_reflectivities_collection[i]) * 0.75)], \
                        cluster_reflectivities_collection[i][-1], cluster_densities[i][0], cluster_centers[i][0], cluster_centers[i][1], cluster_centers[i][2], 1, 1))
                elif 'no' not in os.path.basename(file) and not smoke_detected:
                    filewriter.writerow((os.path.basename(file), curr_packet, (curr_packet * packet_size) + 1, (curr_packet + 1) * packet_size, i, points_per_cluster[i][0], \
                        cluster_reflectivities[i][0], cluster_reflectivities_collection[i][0], cluster_reflectivities_collection[i][int(len(cluster_reflectivities_collection[i]) * 0.25)], \
                        cluster_reflectivities_collection[i][int(len(cluster_reflectivities_collection[i]) * 0.5)], \
                        cluster_reflectivities_collection[i][int(len(cluster_reflectivities_collection[i]) * 0.75)], \
                        cluster_reflectivities_collection[i][-1], cluster_densities[i][0], cluster_centers[i][0], cluster_centers[i][1], cluster_centers[i][2], 0, 1))
                elif 'no' in os.path.basename(file) and not smoke_detected:
                    filewriter.writerow((os.path.basename(file), curr_packet, (curr_packet * packet_size) + 1, (curr_packet + 1) * packet_size, i, points_per_cluster[i][0], \
                        cluster_reflectivities[i][0], cluster_reflectivities_collection[i][0], cluster_reflectivities_collection[i][int(len(cluster_reflectivities_collection[i]) * 0.25)], \
                        cluster_reflectivities_collection[i][int(len(cluster_reflectivities_collection[i]) * 0.5)], \
                        cluster_reflectivities_collection[i][int(len(cluster_reflectivities_collection[i]) * 0.75)], \
                        cluster_reflectivities_collection[i][-1], cluster_densities[i][0], cluster_centers[i][0], cluster_centers[i][1], cluster_centers[i][2], 0, 0))
                elif 'no' in os.path.basename(file) and smoke_detected:
                    filewriter.writerow((os.path.basename(file), curr_packet, (curr_packet * packet_size) + 1, (curr_packet + 1) * packet_size, i, points_per_cluster[i][0], \
                        cluster_reflectivities[i][0], cluster_reflectivities_collection[i][0], cluster_reflectivities_collection[i][int(len(cluster_reflectivities_collection[i]) * 0.25)], \
                        cluster_reflectivities_collection[i][int(len(cluster_reflectivities_collection[i]) * 0.5)], \
                        cluster_reflectivities_collection[i][int(len(cluster_reflectivities_collection[i]) * 0.75)], \
                        cluster_reflectivities_collection[i][-1], cluster_densities[i][0], cluster_centers[i][0], cluster_centers[i][1], cluster_centers[i][2], 1, 0))

            curr_packet += 1

        print('Packet Processed!')

    print('File Processed!')

print('Processing Complete!')