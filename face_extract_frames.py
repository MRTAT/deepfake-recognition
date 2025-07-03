import cv2
import mediapipe as mp
import os
from pathlib import Path
from tqdm import tqdm


def extract_face_frames(video_path, output_dir):
    """
    Extract ALL frames from video and crop faces using MediaPipe
    Args:
        video_path: path to video file
        output_dir: directory to save frames
    """
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,  # 0 for short videos, 1 for long distance
        min_detection_confidence=0.5
    )

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video info - FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {duration:.2f}s")

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    saved_count = 0
    video_name = Path(video_path).stem  # Get filename without extension

    print(f"Processing video: {video_name}")

    # Initialize progress bar
    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process ALL frames (no skipping)
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(rgb_frame)

        faces_in_frame = 0
        if results.detections:
            for i, detection in enumerate(results.detections):
                # Get face bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape

                # Convert from relative coordinates to pixel coordinates
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)

                # Expand bounding box for more context
                margin = 20
                x = max(0, x - margin)
                y = max(0, y - margin)
                width = min(w - x, width + 2 * margin)
                height = min(h - y, height + 2 * margin)

                # Crop face from frame
                face_crop = frame[y:y + height, x:x + width]

                # Save face crop
                if face_crop.size > 0:
                    filename = f"{video_name}_frame_{frame_count:06d}_face_{i:02d}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    cv2.imwrite(filepath, face_crop)
                    saved_count += 1
                    faces_in_frame += 1

        frame_count += 1

        # Update progress bar with detailed info
        pbar.set_postfix({
            'Faces saved': saved_count,
            'Current frame faces': faces_in_frame
        })
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"Completed! Processed {frame_count} frames and saved {saved_count} face crops from video {video_name}")


def process_dataset(dataset_root):
    """
        dataset_root: path to original dataset folder
    """

    dataset_path = Path(dataset_root)

    # Create new output folder named dataset_std
    output_root = dataset_path.parent / "dataset_std"
    print(f"Output will be saved to: {output_root}")

    # Loop through train and test
    for split in ['train_videos', 'test_videos']:
        split_path = dataset_path / split
        if not split_path.exists():
            continue

        # Loop through fake and real
        for category in ['fake', 'real']:
            category_path = split_path / category
            if not category_path.exists():
                continue

            # Create corresponding output directory in dataset_std
            output_path = output_root / split / category

            print(f"\nProcessing {split}/{category}")
            print(f"Input: {category_path}")
            print(f"Output: {output_path}")

            # Process all mp4 videos in directory
            video_files = list(category_path.glob("*.mp4"))

            # Progress bar for all videos in category
            for video_file in tqdm(video_files, desc=f"Videos in {split}/{category}"):
                extract_face_frames(
                    video_path=str(video_file),
                    output_dir=str(output_path)
                )


# Usage
if __name__ == "__main__":
    # Change this path to your dataset path
    dataset_root = "dataset"

    # Process entire dataset
    process_dataset(dataset_root)
