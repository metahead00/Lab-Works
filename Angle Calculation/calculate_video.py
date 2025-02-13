import cv2
import numpy as np
import math

def detect_markers_in_frame(frame):
    # change to gray scale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector.detectMarkers(gray)

def find_centers(corner_coordinates, ids):
    centers = []  # id, center_x, center_y
    for i in range(len(corner_coordinates)):
        center = sum(corner_coordinates[i][0]) / 4
        centers.append((ids[i][0], int(center[0]), int(center[1])))

    # sort by marker ID
    # with this approach, the number of markers is not important
    # but the order in the frame is important
    centers.sort()  
    return centers

def draw_lines_and_calculate_angles(centers, frame):
    if len(centers) < 3:
        return frame  # not enough markers for angles

    prev_point = np.array(centers[0][1:])
    angle_point = np.array(centers[1][1:])
    for i in range(2, len(centers)):
        next_point = np.array(centers[i][1:])
        
        # draw line
        cv2.line(frame, tuple(map(int, prev_point)), tuple(map(int, angle_point)), (0, 255, 0), 2)
        
        # calculate angle
        # theta = arccos(u dot v / |u||v|)
        vector_u = angle_point - prev_point
        vector_v = angle_point - next_point
        angle = math.acos(np.dot(vector_u, vector_v) / (np.linalg.norm(vector_u) * np.linalg.norm(vector_v) + 1e-6)) * (180 / math.pi)
        
        # draw angle
        cv2.putText(frame, f"{int(angle)}", tuple(map(int, angle_point)), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
        
        prev_point = angle_point
        angle_point = next_point
    
    # last line...
    cv2.line(frame, tuple(map(int, prev_point)), tuple(map(int, angle_point)), (0, 255, 0), 2)

    return frame

# there is a problem here, idk what
""" def perspective_transform(frame, corners):
    Perform a perspective transformation for the entire frame
    using the corners of a single marker as a reference.
    # Use the corners of the first detected marker (4 points)
    pts1 = np.float32([corner[0] for corner in corners[0]])  # First marker's corners

    # Define the destination points to map to the full frame size
    height, width = frame.shape[:2]
    pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply the transformation to the entire frame
    warped_frame = cv2.warpPerspective(frame, M, (width, height))

    return warped_frame """



def process_video(video_source, output_file):
    cap = cv2.VideoCapture(video_source)  # 0 for webcam, or provide video file path
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return
    
     # Get video resolution (width, height)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize VideoWriter to save the video (output_file is the filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))  # 20 FPS

    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break

        corners, ids, rejected = detect_markers_in_frame(frame)
        # fixed_frame = perspective_transform(frame, corners)

        if ids is not None:
            centers = find_centers(corners, ids)
            frame = draw_lines_and_calculate_angles(centers, frame)

        # Write the processed frame to the video file
        out.write(frame)

        cv2.imshow('USB Webcam Marker Detection', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Release the video writer
    cv2.destroyAllWindows()

# Run the program
video_path = "test8_perspectivetest.avi"
process_video(1, video_path)  # 1 for usb webcam or replace with video file path
                                      # change output path if it already exists