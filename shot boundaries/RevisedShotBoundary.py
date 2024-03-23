import os
import cv2
import json
import numpy as np


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def calculate_frame_difference(frame1, frame2):
    # Example using color histograms
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return diff, hist1, hist2


def detect_shot_boundaries(frames, fps, threshold=0.2):
    shot_boundaries = []
    for i in range(len(frames) - 1):
        diff, hist1, hist2 = calculate_frame_difference(frames[i], frames[i + 1])
        if diff > threshold:
            timestamp = i / fps  # Convert frame number to timestamp
            # Store frame number, timestamp, and histogram of the first frame in the boundary
            shot_boundaries.append((i, timestamp, hist1))
    return shot_boundaries


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def play_video_from_frame(video_path, start_frame):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pause = False  # To control pause/play

    while cap.isOpened():
        if not pause:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Video', frame)

        key = cv2.waitKey(25) & 0xFF

        if key == ord('q'):  # Press 'q' to quit the video playback
            break
        elif key == ord('p'):  # Press 'p' to toggle pause/play
            pause = not pause
        elif key == ord('r'):  # Press 'r' to reset to start frame
            pause = False
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    cap.release()
    cv2.destroyAllWindows()


def extract_frame_details(frame):
    return frame.mean(axis=(0, 1))   


def process_video_for_boundaries(video_path):
    fps = get_video_fps(video_path)
    frames = extract_frames(video_path)
    boundaries = detect_shot_boundaries(frames, fps)
    boundary_details = {}

    for frame_number, _ in boundaries:
        frame_details = extract_frame_details(frames[frame_number])
        boundary_details[frame_number] = frame_details

    return boundary_details


def match_query_video(query_video_path, stored_boundaries):
    query_fps = get_video_fps(query_video_path)
    query_frames = extract_frames(query_video_path)
    query_boundaries = detect_shot_boundaries(query_frames, query_fps)

    for frame_number, _ in query_boundaries:
        query_frame_details = extract_frame_details(query_frames[frame_number])
        match = find_matching_frame(query_frame_details, stored_boundaries)
        if match:
            return match, frame_number

    return None, None


def find_matching_frame(full_video_boundaries, query_video_boundaries):
    min_diff = float('inf')
    matching_frame = None

    for query_frame, _, query_hist in query_video_boundaries:  # Corrected tuple unpacking
        query_hist = np.array(query_hist, dtype=np.float32)

        for full_frame, _, full_hist in full_video_boundaries:  # Corrected tuple unpacking
            full_hist = np.array(full_hist, dtype=np.float32)

            diff = cv2.compareHist(query_hist, full_hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff < min_diff:
                min_diff = diff
                matching_frame = full_frame

    return matching_frame


def store_shot_boundaries(video_path, shot_boundaries):
    # Define the path for the 'boundaries' subfolder within 'shot boundaries' folder
    base_folder = os.path.join(os.getcwd(), 'shot boundaries', 'boundaries')

    # Create the 'boundaries' subfolder if it doesn't exist
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # Construct the JSON file path
    base_filename = os.path.basename(video_path)
    boundary_file_name = os.path.splitext(base_filename)[0] + '_boundaries.json'
    boundary_file = os.path.join(base_folder, boundary_file_name)

    # Convert shot boundaries to a serializable format and write to a JSON file
    boundaries_data = [{"frame": frame, "timestamp": timestamp, "histogram": hist.tolist()} for frame, timestamp, hist in shot_boundaries]
    with open(boundary_file, 'w') as file:
        json.dump(boundaries_data, file)

    print(f"Shot boundaries stored in {boundary_file}")


def process_and_store_boundaries(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):  
            video_path = os.path.join(folder_path, filename)
            print(f"Processing video: {video_path}")

            fps = get_video_fps(video_path)
            frames = extract_frames(video_path)
            boundaries = detect_shot_boundaries(frames, fps)

            store_shot_boundaries(video_path, boundaries)


def load_shot_boundaries(boundary_file):
    try:
        with open(boundary_file, 'r') as file:
            boundaries_data = json.load(file)
            # Correctly unpack the dictionaries
            return [(int(boundary['frame']), boundary['timestamp'], np.array(boundary['histogram'])) for boundary in boundaries_data]
    except FileNotFoundError:
        print(f"No stored boundaries found for {boundary_file}")
        return None


def load_boundaries_for_folder(subfolder_path):
    boundaries_dict = {}
    for filename in os.listdir(subfolder_path):
        if filename.endswith('_boundaries.json'):
            json_path = os.path.join(subfolder_path, filename)
            video_filename = filename.replace('_boundaries.json', '.mp4')

            print(f"Loading boundaries for video: {video_filename}")

            boundaries = load_shot_boundaries(json_path)
            if boundaries:
                boundaries_dict[video_filename] = boundaries

    return boundaries_dict


def find_match_in_videos(query_boundaries, stored_video_boundaries):
    for video_path, video_boundaries in stored_video_boundaries.items():
        matching_frame = find_matching_frame(video_boundaries, query_boundaries)
        if matching_frame is not None:
            print(f"Match found in {video_path} at frame {matching_frame}")
            return video_path, matching_frame
        else:
            print(f"No match found in {video_path}")


def find_best_match(query_video_boundaries, stored_video_boundaries):
    best_match = None
    best_query_match = None
    min_diff = float('inf')
    best_match_video_path = None

    for video_path, video_boundaries in stored_video_boundaries.items():
        for query_frame, _, query_hist in query_video_boundaries:
            query_hist = np.array(query_hist, dtype=np.float32)

            for full_frame, _, full_hist in video_boundaries:
                full_hist = np.array(full_hist, dtype=np.float32)

                diff = cv2.compareHist(query_hist, full_hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff < min_diff:
                    min_diff = diff
                    best_match = full_frame
                    best_query_match = query_frame
                    best_match_video_path = video_path

    return best_match_video_path, best_match, best_query_match


def calculate_start_frame_in_full_video(full_video_match_frame, query_video_match_frame):
    # Calculate the offset in the query video
    offset_in_query_video = query_video_match_frame

    # Apply this offset to the full video's matching frame
    start_frame_in_full_video = full_video_match_frame - offset_in_query_video
    return max(start_frame_in_full_video, 0)  


process_and_store_videos = False
load_videos = True
single_video = False
fps = 30

boundaries_folder_path = os.path.join(os.getcwd(), 'boundaries')
videos_directory = '/Users/anthonypenaflor/Downloads/cs576_final_dataset/final_videos'

# Usage
# Full video
if process_and_store_videos:
    process_and_store_boundaries(videos_directory)

if load_videos:
    loaded_video_boundaries = load_boundaries_for_folder(boundaries_folder_path)

if single_video:
    video_path = '/Users/anthonypenaflor/Downloads/cs576_final_dataset/final_videos/video1.mp4'
    fps = get_video_fps(video_path)
    print("Processing video boundaries")
    video_frames = extract_frames(video_path)
    video_boundaries = detect_shot_boundaries(video_frames, fps)
    print("Finished video boundaries")

# Query Video
query_path = '/Users/anthonypenaflor/Downloads/cs576_final_dataset/final_query_videos/video11_1.mp4'
print("Processing query boundaries")
query_frames = extract_frames(query_path)
query_boundaries = detect_shot_boundaries(query_frames, fps)
print("Finished query boundaries")

# Find matching frame
# path, matching_frame = find_match_in_videos(query_boundaries, loaded_video_boundaries)
path, matching_frame, query_frame = find_best_match(query_boundaries, loaded_video_boundaries)

# matching_frame = find_matching_frame(video_boundaries, query_boundaries)
if matching_frame is not None:
    print(f"Earliest matching frame in full video: {matching_frame}")
    full_path = os.path.join(videos_directory, path)
    actual_frame = calculate_start_frame_in_full_video(matching_frame, query_frame)
    play_video_from_frame(full_path, actual_frame)
else:
    print("No matching frame found")
