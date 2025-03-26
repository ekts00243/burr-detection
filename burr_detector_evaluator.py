import cv2
import numpy as np
import os
import sys
import time
from loguru import logger
import matplotlib.pyplot as plt
from detect_burrs_hed import load_hed_model, process_frame, DETECTION_CONFIG

# Configure logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("logs/burr_evaluator_{time}.log", rotation="10 MB", level="DEBUG")

def evaluate_methods(video_path, roi, sample_frames=10):
    """Evaluate different burr detection methods on sample frames from a video"""
    logger.info(f"Evaluating burr detection methods on {video_path}")
    
    # Load model
    model = load_hed_model()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample (evenly distributed)
    if sample_frames > frame_count:
        sample_frames = frame_count
    
    sample_indices = [int(i * frame_count / sample_frames) for i in range(sample_frames)]
    
    # Available methods to evaluate
    methods_to_test = [
        ["hed"],
        ["canny"],
        ["hog"],
        ["depth"],
        ["multiscale"],
        ["hed", "canny"],
        ["hed", "canny", "hog"],
        ["hed", "canny", "hog", "depth", "multiscale"]
    ]
    
    # Results storage
    results = {method_key(m): {"burr_counts": [], "processing_time": []} for m in methods_to_test}
    
    # Process sample frames with each method
    for frame_idx in sample_indices:
        logger.info(f"Processing frame {frame_idx}/{frame_count}")
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame {frame_idx}")
            continue
        
        # Apply each method
        for methods in methods_to_test:
            method_name = method_key(methods)
            logger.info(f"Testing method: {method_name}")
            
            # Create configuration for this test
            config = DETECTION_CONFIG.copy()
            config["methods"] = methods
            
            # Process frame and measure time
            start_time = time.time()
            _, _, _, _, burr_count = process_frame(frame.copy(), roi, model, config)
            processing_time = time.time() - start_time
            
            # Store results
            results[method_name]["burr_counts"].append(burr_count)
            results[method_name]["processing_time"].append(processing_time)
            
            logger.info(f"Method: {method_name} - Burrs: {burr_count} - Time: {processing_time:.4f}s")
            
    cap.release()
    
    # Calculate average results
    for method_name in results:
        avg_burrs = np.mean(results[method_name]["burr_counts"])
        avg_time = np.mean(results[method_name]["processing_time"])
        results[method_name]["avg_burrs"] = avg_burrs
        results[method_name]["avg_time"] = avg_time
        logger.info(f"Method: {method_name} - Avg Burrs: {avg_burrs:.2f} - Avg Time: {avg_time:.4f}s")
    
    # Generate visualization of results
    visualize_results(results)
    
    return results

def method_key(methods_list):
    """Convert a list of methods to a string key"""
    return "+".join(methods_list)

def visualize_results(results):
    """Visualize the evaluation results"""
    plt.figure(figsize=(12, 8))
    
    # Extract data
    methods = list(results.keys())
    avg_burrs = [results[m]["avg_burrs"] for m in methods]
    avg_times = [results[m]["avg_time"] for m in methods]
    
    # Create bar chart for burr counts
    plt.subplot(2, 1, 1)
    plt.bar(methods, avg_burrs, color='skyblue')
    plt.ylabel('Average Burr Count')
    plt.title('Average Burr Count by Method')
    plt.xticks(rotation=45, ha='right')
    
    # Create bar chart for processing times
    plt.subplot(2, 1, 2)
    plt.bar(methods, avg_times, color='salmon')
    plt.ylabel('Average Processing Time (s)')
    plt.title('Average Processing Time by Method')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('data/results/method_comparison.png')
    logger.info("Saved method comparison visualization to data/results/method_comparison.png")
    
    # Create scatter plot showing time vs. burr count
    plt.figure(figsize=(10, 8))
    plt.scatter(avg_times, avg_burrs, s=100, alpha=0.7)
    
    # Add labels for each point
    for i, method in enumerate(methods):
        plt.annotate(method, (avg_times[i], avg_burrs[i]), 
                     xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Processing Time (s)')
    plt.ylabel('Average Burr Count')
    plt.title('Burr Detection Methods: Performance vs. Detection Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('data/results/method_performance.png')
    logger.info("Saved method performance visualization to data/results/method_performance.png")

def interactive_method_selector(video_path, roi=None):
    """Interactive tool to visualize the results of different methods on a specific frame"""
    # Load model
    model = load_hed_model()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # If ROI is not provided, create one in the center
    if roi is None:
        roi_w, roi_h = 200, 200
        roi = (frame_width//2 - roi_w//2, frame_height//2 - roi_h//2, roi_w, roi_h)
    
    # Current state
    current_frame_idx = 0
    current_methods = ["hed", "canny"]
    
    # Available methods
    all_methods = ["hed", "canny", "hog", "depth", "multiscale"]
    
    # Control instructions
    instructions = [
        "Controls:",
        "- Left/Right arrow: Previous/Next frame",
        "- Number keys 1-5: Toggle methods",
        "  1: HED, 2: Canny, 3: HOG, 4: Depth, 5: Multiscale",
        "- 'a': Toggle all methods on/off",
        "- 'q': Quit"
    ]
    
    while True:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame {current_frame_idx}")
            current_frame_idx = 0
            continue
        
        # Create configuration for this test
        config = DETECTION_CONFIG.copy()
        config["methods"] = current_methods
        
        # Process frame
        _, _, _, composite, burr_count = process_frame(frame.copy(), roi, model, config)
        
        # Add frame info
        cv2.putText(composite, f"Frame: {current_frame_idx}/{frame_count}", 
                   (10, composite.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add active methods
        methods_text = ", ".join(current_methods)
        cv2.putText(composite, f"Active Methods: {methods_text}", 
                   (10, composite.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display instructions
        for i, line in enumerate(instructions):
            cv2.putText(composite, line, 
                       (10, 50 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 1)
        
        # Display the composite
        cv2.imshow("Burr Detection Method Selector", composite)
        
        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == 81:  # Left arrow
            current_frame_idx = max(0, current_frame_idx - 1)
        elif key == 83:  # Right arrow
            current_frame_idx = min(frame_count - 1, current_frame_idx + 1)
        elif key >= ord('1') and key <= ord('5'):
            # Toggle methods
            method_idx = key - ord('1')
            if method_idx < len(all_methods):
                method = all_methods[method_idx]
                if method in current_methods:
                    current_methods.remove(method)
                else:
                    current_methods.append(method)
                if not current_methods:  # Ensure at least one method is selected
                    current_methods.append(all_methods[0])
        elif key == ord('a'):
            # Toggle all methods
            if len(current_methods) < len(all_methods):
                current_methods = all_methods.copy()
            else:
                current_methods = [all_methods[0]]
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    
    video_path = "data/raw/recording_20250324_111847.avi"
    roi = (100, 100, 200, 200)  # Adjust for your specific video
    
    # Run interactive selector
    interactive_method_selector(video_path, roi)
    
    # Or run evaluation
    # evaluate_methods(video_path, roi, sample_frames=5)
