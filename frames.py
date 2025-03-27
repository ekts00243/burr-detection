import cv2
import os
import argparse

def video_to_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames; exit loop

        # Construct filename and save the frame as JPEG
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to '{output_folder}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a video file into frames.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output", type=str, default="frames", help="Directory to save the frames")
    args = parser.parse_args()

    video_to_frames(args.video, args.output)
