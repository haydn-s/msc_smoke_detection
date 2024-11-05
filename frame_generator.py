# Import Libraries
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import os

# Read CSV Data
if __name__ == '__main__':
    foldername = 'medium smoke, medium flame test 3'
    filename = foldername + '.csv'
    print('Filename: ' + filename)
    myRows = []
    row_count = 1
    # note: timestamp in nanoseconds.
    try:
        print('Reading File...')
        with open(filename, 'r') as myCSV:
            data = csv.reader(myCSV)
            next(data, None)
            for row in data:
                myRows.append(row)
                print('Row Imported: ' + str(row_count))
                row_count += 1
        myCSV.close()
        print('File Read and Closed!')
    except FileNotFoundError:
        print('No File!')

# Define Variables
packet_size = 12500
curr_packet = 0

# Iterate through Packets
while (len(myRows) > ((curr_packet + 1) * packet_size)):    
    x_coords = []
    y_coords = []
    z_coords = []
    point_reflect = []

    for num in range(packet_size):
        x_coords.append(float(myRows[num + (curr_packet * packet_size)][8]))
        y_coords.append(float(myRows[num + (curr_packet * packet_size)][9]))
        z_coords.append(float(myRows[num + (curr_packet * packet_size)][10]))
        point_reflect.append(int(myRows[num + (curr_packet * packet_size)][11]))
    
    # Find the Minimum Coordinate
    min_x = min(x_coords)
    min_y = min(y_coords)
    min_z = min(z_coords)

    # Find the Maximum Coordinate
    max_x = max(x_coords)
    max_y = max(y_coords)
    max_z = max(z_coords)

    # Define Reference Frame Dimensions
    width = 640
    height = 512

    # Normalize the Data -> ROUND TO THE NEAREST INT FOR ALL COORDS
        # X Projection
    horizontal_xproj_scale_factor = width / (max_y - min_y)
    vertical_xproj_scale_factor = height / (max_z - min_z)
    norm_horizontal_x = [int(val * horizontal_xproj_scale_factor - (min_y * horizontal_xproj_scale_factor)) for val in y_coords]
    norm_vertical_x = [int(val * vertical_xproj_scale_factor - (min_z * vertical_xproj_scale_factor)) for val in z_coords]

        # Y Projection
    horizontal_yproj_scale_factor = width / (max_x - min_x)
    vertical_yproj_scale_factor = height / (max_z - min_z)
    norm_horizontal_y = [int(val * horizontal_yproj_scale_factor - (min_x * horizontal_yproj_scale_factor)) for val in x_coords]
    norm_vertical_y = [int(val * vertical_yproj_scale_factor - (min_z * vertical_yproj_scale_factor)) for val in z_coords]

        # Z Projection
    horizontal_zproj_scale_factor = width / (max_x - min_x)
    vertical_zproj_scale_factor = height / (max_y - min_y)
    norm_horizontal_z = [int(val * horizontal_zproj_scale_factor - (min_x * horizontal_zproj_scale_factor)) for val in x_coords]
    norm_vertical_z = [int(val * vertical_zproj_scale_factor - (min_y * vertical_zproj_scale_factor)) for val in y_coords]

    # Create Images to Plot Reflectivity
    x_image = np.full((height, width), 255, np.uint8)
    y_image = np.full((height, width), 255, np.uint8)
    z_image = np.full((height, width), 255, np.uint8)

    for index in range(len(norm_horizontal_x)):
        prev_reflect = x_image[norm_vertical_x[index] - 1][norm_horizontal_x[index] - 1]
        curr_reflect = point_reflect[index]

        if (prev_reflect == 0 or curr_reflect < prev_reflect):
            x_image[norm_vertical_x[index] - 1][norm_horizontal_x[index] - 1] = curr_reflect
            y_image[norm_vertical_y[index] - 1][norm_horizontal_y[index] - 1] = curr_reflect
            z_image[norm_vertical_z[index] - 1][norm_horizontal_z[index] - 1] = curr_reflect
        else:
            pass

    # Save the Images to Respective Folders
    x_directory = r'D:/Data/processed_data/' + foldername + r'/projections_x'
    y_directory = r'D:/Data/processed_data/' + foldername + r'/projections_y'
    z_directory = r'D:/Data/processed_data/' + foldername + r'/projections_z'

        # X Projection
    image_name_x = 'x_projection_packet_' + str(curr_packet + 1) + '.jpg'
    if not os.path.exists(x_directory):    
        os.makedirs(x_directory)
        os.chdir(x_directory)
        cv2.imwrite(image_name_x, x_image)
        print('X Projection Saved')
    else:    
        os.chdir(x_directory)
        cv2.imwrite(image_name_x, x_image)
        print('X Projection Saved')

        # Y Projection
    image_name_y = 'y_projection_packet_' + str(curr_packet + 1) + '.jpg'
    if not os.path.exists(y_directory):    
        os.makedirs(y_directory)
        os.chdir(y_directory)
        cv2.imwrite(image_name_y, y_image)
        print('Y Projection Saved')
    else:    
        os.chdir(y_directory)
        cv2.imwrite(image_name_y, y_image)
        print('Y Projection Saved')

        # Z Projection
    image_name_z = 'z_projection_packet_' + str(curr_packet + 1) + '.jpg'
    if not os.path.exists(z_directory):
        os.makedirs(z_directory)
        os.chdir(z_directory)
        cv2.imwrite(image_name_z, z_image)
        print('Z Projection Saved')
    else:    
        os.chdir(z_directory)
        cv2.imwrite(image_name_z, z_image)
        print('Z Projection Saved')

    curr_packet += 1