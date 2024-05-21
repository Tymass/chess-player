import cv2
import numpy as np
import math
from sklearn.cluster import DBSCAN
import time
from numpy.typing import NDArray
from typing import Any

def line_intersection(line1: list, line2: list) -> int:
    '''
    Calculate points of line intersections.

    args:
        - list: line [[x1,x2],[y1,y2]]
    return:
        - int: coordinates of intersection: x,y
    '''
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def get_edges(img: NDArray[Any], blur_kernel: tuple=(3,3), border_default: int=3) -> NDArray[Any]:
    '''
    Get edges from Canny filter.
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, border_default)
    edges = cv2.Canny(blurred,low_thr_canny,up_thr_canny)
    return edges


def get_lines(edges: NDArray[Any], rho: int, theta: float, threshold: int, min_line_length: int, max_line_gap: int) -> NDArray[Any]:
    '''
    Extract lines from the image edges.
    '''
    return cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

def keep_lines(lines: NDArray[Any], line_length_threshold: int) -> list:
    '''
        Keep only lines with given length.
    '''
    valid_lines = []
    for line in lines:
        #print(np.linalg.norm(line))
        for x1,y1,x2,y2 in line:
            vec_magnitude = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            #print(vec_magnitude)
            if vec_magnitude > line_length_threshold:
                valid_lines.append(line)
                #cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    print(valid_lines)
    return valid_lines


def convert_to_points(line: NDArray[Any]) -> tuple:
    '''
        Take points [x1,y1,x2,y2] and convert them to lines ([x1,y1],[x2,y2]).
    '''
    line = line[0]
    return ([line[0], line[1]], [line[2], line[3]])


def calculate_intersecitons(valid_lines: list) -> list:
    '''
        Calculate intersection points of all lines.
    '''
    intersections = []
    for i in range(len(valid_lines)):
        for j in range(i + 1, len(valid_lines)):
            line1 = convert_to_points(valid_lines[i])
            line2 = convert_to_points(valid_lines[j])
            
            try:
                intersection = line_intersection(line1, line2)
                intersections.append(intersection)
            except Exception as e:
                print(f"Line {i} and Line {j} do not intersect")
    return intersections


def keep_line_intersections(intersections: list) -> list:
    '''
        Keep only intersections that appear on image.
    '''
    x_lim = range(0, grid.shape[0] + 1)
    y_lim = range(0, grid.shape[1] + 1)
    valid_intersections = []
    for intersection in intersections:
        x, y = int(intersection[0]), int(intersection[1])

        if x in x_lim and y in y_lim:
            valid_intersections.append((x,y))
    return valid_intersections


def get_point_centers(valid_intersections: list, grid: NDArray[Any], eps: int=20, min_samples: int=2) -> list:
    '''
        Calculate centroids from intersections points then group them into centers with DBSCAN. 
    '''

    for valid_intersection in valid_intersections:
        x, y = int(valid_intersection[0]), int(valid_intersection[1])
        try:
            grid[x, y] = 255
        except:
            print(f'Intersection: excluded')

    grid = grid.transpose().astype(np.uint8)

    _, labels, _, centroids = cv2.connectedComponentsWithStats(grid, 8, cv2.CV_32S)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(centroids)

    # Get labels (clusters)
    labels = dbscan.labels_

    # Retrieve unique clusters (excluding noise, labeled as -1)
    unique_labels = set(labels) - {-1}

    # Compute centroids for each cluster
    centers = []
    for label in unique_labels:
        cluster_points = centroids[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centers.append(centroid)

    return centers


def calculate_distances(points: list) -> NDArray[Any]:
    '''
        Compute the pairwise Manhattan distances between a list of points and returns a distance matrix. 
    '''

    # Convert points to NumPy array for easier manipulation
    points_array = np.array(points)
    num_points = len(points_array)
    
    # Initialize distance matrix with zeros
    distances = np.zeros((num_points, num_points))
    
    # Calculate pairwise Manhattan distances between points
    for i in range(num_points):
        for j in range(i+1, num_points):
            distances[i, j] = np.sum(np.abs(points_array[i] - points_array[j]))
            distances[j, i] = distances[i, j]  # distances matrix is symmetric

    return distances


def remove_close_points(points: list, threshold: int, image_size: tuple) -> list:
    '''
        If some points are too close to each other, remove points that are farther from image centre. Points are considered 'close' to each other if they are closer than mean distance.
    '''
    distances = calculate_distances(points)
    num_points = len(points)
    indices_to_remove = []

    for i in range(num_points):
        for j in range(i+1, num_points):
            if distances[i, j] < threshold:
                # Calculate the center of the image
                center_x = image_size[0] / 2
                center_y = image_size[1] / 2
                center = (center_x, center_y)
                # Calculate the distances of each point to the center of the image
                distance_to_center_i = np.sum(np.abs(np.array(points[i]) - np.array(center)))
                distance_to_center_j = np.sum(np.abs(np.array(points[j]) - np.array(center)))
                # Remove the point that is farther from the center
                if distance_to_center_i > distance_to_center_j:
                    indices_to_remove.append(i)
                else:
                    indices_to_remove.append(j)

    # Remove duplicates from the list of indices to remove
    indices_to_remove = list(set(indices_to_remove))

    # Remove points from the original list
    points_filtered = [point for idx, point in enumerate(points) if idx not in indices_to_remove]

    return points_filtered


def divide_points_into_rows(points: list, threshold: int) -> list:
    '''
        Group points into rows based on y-coordinates difference.
    '''
    # Sort points based on y-coordinate
    sorted_points = sorted(points, key=lambda point: point[1])

    rows = []
    current_row = [sorted_points[0]]

    for i in range(1, len(sorted_points)):
        # Check if the y-coordinate difference between current point and last point in current row is less than threshold
        if abs(sorted_points[i][1] - current_row[-1][1]) <= threshold:
            current_row.append(sorted_points[i])
        else:
            rows.append(current_row)
            current_row = [sorted_points[i]]

    # Add the last row
    rows.append(current_row)
    return rows


def fill_patterns(rows: list, target_length: int) -> list:
    '''
        Fill every row with steadily distributed points.
    '''
    filled_rows = []

    for points in rows:

        # Separate the points into x and y coordinates
        x_coords, y_coords = zip(*points)
        
        # Create the target x positions to interpolate
        target_x_positions = np.linspace(min(x_coords), max(x_coords), target_length)
        
        # Perform linear interpolation for the y coordinates based on x coordinates
        interpolated_y_values = np.interp(target_x_positions, x_coords, y_coords)
        
        # Combine the interpolated x and y values
        filled_row = list(zip(target_x_positions, interpolated_y_values))
    
        filled_rows.append(filled_row)
    
    return filled_rows
    

def mean_closest_distance(points: list) -> float:
    '''
        Calculate the mean of the closest pairwise Manhattan distances between a set of points. 
    '''
    distances = calculate_distances(points)
    
    # Exclude diagonal and initialize minimum distance to a large value
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    
    # Calculate mean of minimum distances
    mean_min_distance = np.mean(min_distances)
    
    return mean_min_distance


def exclude_small_rows(rows: list, min_row_len: int=2) -> list:
    '''
        Remove rows that are too small.
    '''
    rows_fixed = []
    for row in rows:
        if len(row) < min_row_len:
            pass
        else:
            rows_fixed.append(row)
    return rows_fixed


def print_points(img: NDArray[Any], board_points: list) -> None:
    '''
        Imshow original image with detected chessboard.
    '''
    for p in board_points:
        x,y = int(p[0]), int(p[1])
        cv2.circle(img, (x,y), 1, (255,255,255), 10)

    cv2.imshow('Detected chessboard', img)
    cv2.waitKey()


def convert_rows_to_points(rows: list) -> list:
    '''
        Convert rows [[point1,point2],[point3,point4]] into list of points [point1,point2,point3,point4]
    '''
    return [item for sublist in rows for item in sublist]


#------------------------------------
t1 = time.time()

img_path = "img/img1.png"

img = cv2.imread(img_path)

blur_kernel = (3,3)
border_default = 3
low_thr_canny = 100
up_thr_canny = 150

edges = get_edges(img)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 200  # angular resolution in radians of the Hough grid
threshold = 10  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 100  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments

line_image = np.copy(img) * 0  # creating a blank to draw lines on

lines = get_lines(edges, rho, theta, threshold, min_line_length, max_line_gap)

line_length_threshold = 120

valid_lines = keep_lines(lines, line_length_threshold)

intersections = calculate_intersecitons(valid_lines)

grid = np.zeros((line_image.shape[1], line_image.shape[0]), dtype=np.uint8)

valid_intersections = keep_line_intersections(intersections)

eps = 20  # maximum distance between two samples
min_samples = 2  # minimum number of samples in a neighborhood

#--------------------------------------------------------------------

centers = get_point_centers(valid_intersections, grid)

mean_dist = mean_closest_distance(centers)

incomplete_board = remove_close_points(centers, mean_dist*0.99, grid.shape)

rows = divide_points_into_rows(incomplete_board, 10) 

rows_fixed = exclude_small_rows(rows)

#points_list = convert_rows_to_points(rows_fixed)

number_of_points = 9

filled_points = fill_patterns(rows_fixed, number_of_points)

board_points = convert_rows_to_points(filled_points)

t2 = time.time()

print(f'time{t2-t1}')

print_points(img, board_points)



