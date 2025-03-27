# Oil Hole Burr Detection System

This project implements an oil hole burr detection system using computer vision and deep learning techniques. It supports both image and video processing, with the option to use a trained RTDETR model for improved burr detection.

## Project Structure

- `data/`: Contains raw images and videos for testing.
- `main.py`: Implements a general burr detection system using traditional computer vision techniques.
- `main2.py`: Implements an oil hole specific burr detection system with RTDETR integration.
- `train.py`: Script for training the RTDETR model.
- `debug_images/`: Directory where debug images are saved when debug mode is enabled.
- `requirements.txt`: Lists the required Python packages.

## Setup

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Install `ultralytics` for RTDETR support:**

    ```bash
    pip install ultralytics
    ```

## Usage

### General Burr Detection (main.py)

```bash
python main.py --input <image_or_video_path> --mode <image|video> --debug --no-gpu
```

*   `--input`: Path to the input image or video.
*   `--mode`: Processing mode, either `image` or `video`.
*   `--debug`: Enables debug mode, saving intermediate images.
*   `--no-gpu`: Disables GPU processing.

### Oil Hole Burr Detection (main2.py)

```bash
python main2.py --input <image_or_video_path> --mode <image|video> --debug --model <path_to_rtdetr_weights> --use_rtdetr --conf <confidence_threshold>
```

*   `--input`: Path to the input image or video.
*   `--mode`: Processing mode, either `image` or `video`.
*   `--debug`: Enables debug mode, saving intermediate images.
*   `--model`: Path to the trained RTDETR model weights (`.pt` file).
*   `--use_rtdetr`: Enables RTDETR-based burr detection.
*   `--conf`: Confidence threshold for RTDETR detections (default: 0.3, suggested: 0.03).

### Training the RTDETR Model (train.py)

1.  Prepare your dataset in the appropriate format (e.g., YOLO format) and create a dataset YAML file.

2.  Run the training script:

    ```bash
    python train.py --data <dataset.yaml> --weights <rtdetr-l.pt> --epochs 50 --batch 16 --imgsz 640
    ```

*   `--data`: Path to the dataset YAML file.
*   `--weights`: Path to the initial RTDETR-L weights (`rtdetr-l.pt`).
*   `--epochs`: Number of training epochs.
*   `--batch`: Batch size.
*   `--imgsz`: Input image size.

## Output

The processed output (image or video) will be displayed in a window, with detected burrs outlined in red. The confidence of each burr detection is logged to the console. For video processing, press `q` to quit the live output.
A result image is also saved to `oil_hole_detection_result.jpg`.
A result video is also saved to `oil_hole_detection_output.avi`.
