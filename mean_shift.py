# Import Libraries
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import os
from sklearn.cluster import MeanShift as ms
from sklearn.cluster import estimate_bandwidth
import pandas as pd
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

# Define Variables
packet_size = 100000
curr_packet = 3

# Read CSV Data
if __name__ == '__main__':
    foldername = 'smoke_1'
    filename = foldername + '.csv'
    print('Filename: ' + filename)
    print()
    myRows = []
    row_count = 1
    # note: timestamp in nanoseconds.
    try:
        print('Reading File...')
        with open(filename, 'r') as myCSV:
            data = csv.reader(myCSV)
            next(data, None)
            rows_count = 0
            for row in data:
                myRows.append(row)
                if row_count % 1000 == 0:
                    print('Row Imported: ' + str(row_count))
                row_count += 1
                if(row_count > (packet_size * curr_packet) + 100001):
                    break
        myCSV.close()
        print('File Read and Closed!\n')
    except FileNotFoundError:
        print('No File!\n')

# Iterate through Packets
while (len(myRows) > ((curr_packet + 1) * packet_size)):    
    print('I MADE IT')
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
            curr_reflectivity[curr_point] = float(myRows[num + (curr_packet * packet_size)][11])
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
    points_per_cluster = np.zeros(shape = (len(cluster_set), 1))
    euclidean_distances = np.zeros(shape = (len(cluster_set), 1))
    density_parameters = np.zeros(shape = (len(cluster_set), 6))
    cluster_densities = np.zeros(shape = (len(cluster_set), 1))

    for i in range(len(cluster_set)):
        density_parameters[i][1] = max_x
        density_parameters[i][3] = max_y
        density_parameters[i][5] = max_z

    print("Number of Clusters: ", len(cluster_set))
    print()

    for i in range(len(curr_coords)):
        cluster_reflectivities[cluster_labels[i]] += curr_reflectivity[i]
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
        euclidean_distances[i] += (float(cluster_centers[i][0])**2 + float(cluster_centers[i][1])**2 + float(cluster_centers[i][2])**2)**(0.5)

    print("Finished the preparation loop!")
    print("\nDensity Parameters:")
    print(density_parameters)
    print("\nSummed Cluster Reflectivities:")
    print(cluster_reflectivities)
    print("\nPoints per Cluster:")
    print(points_per_cluster)
    print("\nCluster Center Euclidean Distances:")
    print(euclidean_distances)
    print("\nCluster Densities:")
    print(cluster_densities)
    print("\nAverage Cluster Reflectivities:")
    print(cluster_reflectivities)
    print("\nFinished cluster calculations!\n")

    # Plot the Result
    print("Plotting the result...\n")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    cluster_colors = 10 * ['r', 'g', 'b', 'c', 'k', 'y', 'm']

    for i in range(len(curr_coords)):
        ax.scatter(curr_coords[i, 0], curr_coords[i, 1], curr_coords[i, 2], c = cluster_colors[cluster_labels[i]], marker = 'o')

    for i in range(len(cluster_centers)):
        if cluster_reflectivities[i] <= 1.7 and cluster_reflectivities[i] > 0:
            if euclidean_distances[i] <= 30 and euclidean_distances[i] >= 1:
                if cluster_densities[i] <= 5 and cluster_densities[i] > 0:
                    ax.scatter(cluster_centers[i, 0], cluster_centers[i, 1], cluster_centers[i, 2], marker = 'x', color = 'black', s = 300, linewidth = 5, zorder = 10)
    
    plt.show()

    # # Save the Files
    # file_directory = r'D:/Data/mean_shift/' + foldername

    # image_name = 'mean_shift_cluser_' + str(curr_packet + 1) + '.jpg'
    # if not os.path.exists(file_directory):    
    #     os.makedirs(file_directory)
    #     os.chdir(file_directory)
    #     cv2.imwrite(image_name, image_name)
    #     print('Frame Saved')
    # else:    
    #     os.chdir(file_directory)
    #     cv2.imwrite(image_name, image_name)
    #     print('Frame Saved')

    curr_packet += 1