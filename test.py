import streamlit as st
import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import Compose, Resize, ToTensor
import numpy as np
from cnn_model import Cnn
import tempfile
import os
import argparse


def parse_args():
    """Parse command line arguments with fallback for Streamlit"""
    parser = argparse.ArgumentParser(description='Deepfake Detection System')
    parser.add_argument('--max_frames', type=int, default=10,
                        help='Maximum number of frames to analyze (default: 10)')
    parser.add_argument('--frame_threshold', type=float, default=0.5,
                        help='Probability threshold for individual frame classification (0-1, default: 0.5)')
    parser.add_argument('--video_threshold', type=float, default=0.2,
                        help='Ratio threshold for video classification (0-1, default: 0.2)')
    parser.add_argument('--model_path', type=str, default='trained_model/best_cnn.pt',
                        help='Path to trained model (default: trained_model/best_cnn.pt)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (default: 2)')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--face_margin', type=int, default=20,
                        help='Margin around detected face (default: 20)')

    try:
        args = parser.parse_args()
    except SystemExit:
        # Fallback for Streamlit
        args = argparse.Namespace(
            max_frames=10,
            frame_threshold=0.5,
            video_threshold=0.2,
            model_path='trained_model/best_cnn.pt',
            num_classes=2,
            input_size=224,
            face_margin=20
        )
    return args


@st.cache_resource
def load_model(path, num_classes):
    """Load the trained CNN model"""
    try:
        model = Cnn(num_classes=num_classes)
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


@st.cache_resource
def initialize_face_detector():
    """Initialize MediaPipe face detector"""
    return mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )


def preprocess_frame(frame, size):
    """Preprocess frame for model input"""
    transform = Compose([
        Resize((size, size)),
        ToTensor()
    ])
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame).unsqueeze(0)  # add batch dimension
    return frame


def predict_frame(model, frame_tensor):
    """Predict if frame is fake with probability"""
    with torch.no_grad():
        output = model(frame_tensor)
        probabilities = F.softmax(output, dim=1)
        fake_prob = probabilities[0][1].item()  # probability of FAKE
        return fake_prob


def detect_and_crop_face(frame, face_detector, margin=20):
    """Detect and crop face from frame using MediaPipe"""
    results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.detections:
        detection = results.detections[0]  # Take first face
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = frame.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        # Expand bbox with margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        width = min(w - x, width + 2 * margin)
        height = min(h - y, height + 2 * margin)

        face_crop = frame[y:y + height, x:x + width]
        return face_crop, (x, y, width, height)

    return None, None


def extract_frames_with_faces(video_path, max_frames, face_detector):
    """Extract frames from video that contain faces"""
    cap = cv2.VideoCapture(video_path)
    frames_with_faces = []

    count = 0
    frame_idx = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames to avoid taking frames too close together
        if frame_idx % 5 != 0:  # Take every 5th frame
            continue

        # Only take frames with faces
        results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            frames_with_faces.append((frame, frame_idx))
            count += 1

    cap.release()
    return frames_with_faces


def analyze_video(video_path, model, face_detector, max_frames, frame_threshold, video_threshold, input_size,
                  face_margin):
    """Analyze video for deepfake detection"""

    # Extract frames with faces
    frames_with_faces = extract_frames_with_faces(video_path, max_frames, face_detector)

    if not frames_with_faces:
        return None, "No faces detected in the video!"

    # Analyze frames
    fake_count = 0
    total_frames = len(frames_with_faces)
    all_probabilities = []
    frame_results = []

    for i, (frame, frame_idx) in enumerate(frames_with_faces):
        # Detect and crop face
        face_crop, face_bbox = detect_and_crop_face(frame, face_detector, face_margin)

        if face_crop is not None:
            # Preprocess and predict
            tensor = preprocess_frame(face_crop, input_size)
            fake_prob = predict_frame(model, tensor)

            all_probabilities.append(fake_prob)

            # Determine if frame is FAKE based on probability threshold
            is_fake = fake_prob > frame_threshold
            if is_fake:
                fake_count += 1

            frame_results.append({
                'frame_idx': frame_idx,
                'fake_prob': fake_prob,
                'is_fake': is_fake,
                'frame': frame,
                'face_bbox': face_bbox
            })

    # Calculate final metrics
    fake_percentage = (fake_count / total_frames) * 100
    required_fake_frames = int(total_frames * video_threshold)

    # Final decision based on video threshold
    overall_result = "FAKE" if fake_count > required_fake_frames else "REAL"

    results = {
        'overall_result': overall_result,
        'fake_count': fake_count,
        'total_frames': total_frames,
        'fake_percentage': fake_percentage,
        'required_fake_frames': required_fake_frames,
        'avg_fake_prob': np.mean(all_probabilities) if all_probabilities else 0,
        'max_fake_prob': max(all_probabilities) if all_probabilities else 0,
        'frame_results': frame_results
    }

    return results, None


def display_results(results):
    """Display analysis results"""
    st.success("✅ Analysis Complete!")

    # Main result
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if results['overall_result'] == "FAKE":
            st.error(f"**Result:** {results['overall_result']}")
        else:
            st.success(f"**Result:** {results['overall_result']}")

    with col2:
        st.metric("FAKE Frames", f"{results['fake_count']}/{results['total_frames']}")

    with col3:
        st.metric("FAKE Percentage", f"{results['fake_percentage']:.1f}%")

    with col4:
        st.metric("Required for FAKE", f">{results['required_fake_frames']}")

    # Decision explanation
    st.subheader("📊 Decision Logic")

    if results['fake_count'] > results['required_fake_frames']:
        st.error(
            f"🔴 **FAKE** detected: {results['fake_count']} frames exceed threshold of {results['required_fake_frames']} frames")
    else:
        st.success(
            f"🟢 **REAL** video: Only {results['fake_count']} frames detected, below threshold of {results['required_fake_frames']} frames")

    # Summary statistics
    st.subheader("📈 Statistical Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Average FAKE Probability", f"{results['avg_fake_prob']:.3f}")

    with col2:
        st.metric("Maximum FAKE Probability", f"{results['max_fake_prob']:.3f}")


def main():
    """Main Streamlit application"""
    args = parse_args()

    st.title("🎭 Deepfake Detection System")
    st.markdown("**Detect deepfake videos using AI face analysis**")

    # Initialize components
    face_detector = initialize_face_detector()

    # Runtime controls
    st.subheader("🎛️ Analysis Settings")
    col1, col2, col3 = st.columns(3)

    with col1:
        max_frames = st.number_input("Maximum Frames to Analyze",
                                     min_value=1, max_value=50,
                                     value=args.max_frames)

    with col2:
        frame_threshold = st.slider("Frame Probability Threshold",
                                    min_value=0.0, max_value=1.0,
                                    value=args.frame_threshold, step=0.05,
                                    help="Threshold to classify individual frame as FAKE")

    with col3:
        video_threshold = st.slider("Video Decision Threshold",
                                    min_value=0.0, max_value=1.0,
                                    value=args.video_threshold, step=0.05,
                                    help="Minimum ratio of FAKE frames to classify video as FAKE")

    # Calculate and display logic
    required_fake_frames = int(max_frames * video_threshold)

    st.markdown(f"📊 **Logic:** Frame is FAKE if probability > **{frame_threshold:.2f}**")
    st.markdown(
        f"🎯 **Decision:** Video is FAKE if > **{required_fake_frames}** frames are FAKE (>{video_threshold:.0%} of {max_frames} frames)")

    # File upload
    st.subheader("📹 Upload Video")
    uploaded_video = st.file_uploader("Upload a video file",
                                      type=['mp4', 'avi', 'mov', 'wmv', 'mpeg4'],
                                      help="Limit 200MB per file • MP4, AVI, MOV, WMV, MPEG4")

    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_video.read())
            video_path = tfile.name

        st.video(video_path)

        if st.button("🔍 Analyze Video", type="primary"):
            with st.spinner(f"Analyzing up to {max_frames} frames..."):

                # Load model
                model = load_model(args.model_path, args.num_classes)
                if model is None:
                    os.unlink(video_path)
                    return

                # Analyze video
                results, error = analyze_video(
                    video_path=video_path,
                    model=model,
                    face_detector=face_detector,
                    max_frames=max_frames,
                    frame_threshold=frame_threshold,
                    video_threshold=video_threshold,
                    input_size=args.input_size,
                    face_margin=args.face_margin
                )

                if error:
                    st.error(f"❌ {error}")
                else:
                    display_results(results)

        # Cleanup
        try:
            os.unlink(video_path)
        except:
            pass


if __name__ == "__main__":
    main()