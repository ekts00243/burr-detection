import argparse
from ultralytics import RTDETR
import cv2
import numpy as np
from pathlib import Path

def test_model(weights_path: str, source: str, conf_thres: float = 0.25):
    """
    Test the trained RTDETR model on images or video.
    
    Args:
        weights_path (str): Path to trained weights file
        source (str): Path to image/video file or directory
        conf_thres (float): Confidence threshold for detections
    """
    # Load the model
    model = RTDETR(weights_path)
    
    # Run inference
    results = model(source, conf=conf_thres)
    
    # Process results
    for r in results:
        # Get the original image
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        
        # Save results
        output_path = Path('runs/detect')
        output_path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(source, str) and Path(source).is_file():
            filename = Path(source).name
        else:
            filename = f"result_{len(list(output_path.glob('*.jpg')))}.jpg"
            
        cv2.imwrite(str(output_path / filename), im)
        
        # Print detection results
        print(f"\nDetections in {filename}:")
        for box in r.boxes:
            print(f"Burr detected with confidence: {box.conf.item():.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test trained RTDETR model for burr detection")
    parser.add_argument('--weights', type=str, default='runs/train/weights/best.pt',
                      help='Path to trained weights file')
    parser.add_argument('--source', type=str, required=True,
                      help='Path to image/video file or directory')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    test_model(args.weights, args.source, args.conf)