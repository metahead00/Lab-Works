import cv2 
import numpy as np
import math

image_path = "images/0.5cm_1.jpg"

def detect_markers_in_image_using_aruco(image_path):
    image = cv2.imread(image_path)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector.detectMarkers(image)

def find_centers(corner_coordinates, ids):
    center_coordinates = []     # id, center_x, center_y
    for i in range(len(corner_coordinates)):    
        corners = corner_coordinates[i][0]
        center = sum(corners) / 4
        center_coordinates.append((ids[i][0], center[0], center[1]))

    # sort by id
    center_coordinates.sort()
    return center_coordinates

def draw_lines_in_a_sequence(centers, image_path):
    #centers -> ids mapped to center_x, center_y
    #centers must be sorted by ids, so the lines will be drawn as: 1 -> 2 -> 3 -> 4...
    image = cv2.imread(image_path)
    for i in range(len(centers)):
        if i == 0:
            prev_point = tuple(map(int, centers[i][1:]))
            continue
        curr_point = tuple(map(int, centers[i][1:]))
        cv2.line(image, prev_point, curr_point, (0, 255, 0))
        prev_point = curr_point

    cv2.imshow('Detected Markers with Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
def calculate_angles_in_order(centers, image_path):
    #centers -> ids mapped to center_x, center_y
    #centers must be sorted by ids, so the angles will be calculated in order. eg 1-3, 3-5, 5-7
    if len(centers) < 3:
        print("There must be minimum 3 points to find an angle\n")
        return
    
    image = cv2.imread(image_path)
    prev_point = np.array(centers[0][1:])
    angle_point = np.array(centers[1][1:])  #calculate the angle here
    for i in range(len(centers)):
        if i < 2:
            continue
        cv2.line(image, tuple(map(int, prev_point)), tuple(map(int, angle_point)), (0, 255, 0))
        next_point = np.array(centers[i][1:])
        # point = x, y
        # linear algebra: theta = arccos(u dot v / |u||v|)
        # u = angle_point - prev, v = next - angle_point
        vector_u = np.array(angle_point) - np.array(prev_point)
        vector_v = np.array(angle_point) - np.array(next_point)
        angle = math.acos(np.dot(vector_u, vector_v) / (np.linalg.norm(vector_u) * np.linalg.norm(vector_v) + 1e-6)) * (180 / math.pi)
        
        angle_point_tuple = tuple(map(int, angle_point))
        cv2.putText(image, f"{int(angle)}", angle_point_tuple, 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)

        prev_point = angle_point
        angle_point = next_point

    
    cv2.line(image, tuple(map(int, prev_point)), tuple(map(int, angle_point)), (0, 255, 0))

    cv2.imshow('Detected Markers with Specific Lines and Angles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


corners, ids, rejected = detect_markers_in_image_using_aruco(image_path)
centers = find_centers(corners, ids)
#draw_lines_in_a_sequence(centers, image_path)
calculate_angles_in_order(centers, image_path)


