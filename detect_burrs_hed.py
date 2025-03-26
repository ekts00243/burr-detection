import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Add, Layer, Lambda
import time
from loguru import logger

# Configure logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("logs/burr_detection_{time}.log", rotation="10 MB", level="DEBUG")

# Force CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logger.info("Running in CPU-only mode")

def create_hed_model(input_shape=(224, 224, 3)):
    """Create a simplified HED-like model using TensorFlow/Keras"""
    logger.debug(f"Creating HED model with input shape {input_shape}")
    inputs = Input(shape=input_shape)
    
    # Normalize input
    normalized = Lambda(lambda x: x/255.0)(inputs)
    
    # Use VGG16 as base
    base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=normalized)
    
    # Get intermediate layers
    layer1 = base_model.get_layer('block1_conv2').output
    layer2 = base_model.get_layer('block2_conv2').output
    layer3 = base_model.get_layer('block3_conv3').output
    layer4 = base_model.get_layer('block4_conv3').output
    layer5 = base_model.get_layer('block5_conv3').output
    
    # Side outputs
    side1 = Conv2D(1, (1, 1), activation='sigmoid')(layer1)
    side2 = Conv2D(1, (1, 1), activation='sigmoid')(layer2)
    side3 = Conv2D(1, (1, 1), activation='sigmoid')(layer3)
    side4 = Conv2D(1, (1, 1), activation='sigmoid')(layer4)
    side5 = Conv2D(1, (1, 1), activation='sigmoid')(layer5)
    
    # Use UpSampling2D instead of tf.image.resize
    # Calculate upsampling factors
    side1_up = UpSampling2D(size=(1, 1))(side1)  # Already the right size
    side2_up = UpSampling2D(size=(2, 2))(side2)
    side3_up = UpSampling2D(size=(4, 4))(side3)
    side4_up = UpSampling2D(size=(8, 8))(side4)
    side5_up = UpSampling2D(size=(16, 16))(side5)
    
    # Fuse all outputs
    fused = Add()([side1_up, side2_up, side3_up, side4_up, side5_up])
    fused = Conv2D(1, (1, 1), activation='sigmoid')(fused)
    
    model = Model(inputs=inputs, outputs=fused)
    logger.debug("HED model created successfully")
    return model

def load_hed_model():
    """Load or create HED model"""
    logger.info("Loading HED edge detection model")
    try:
        # Set memory growth to avoid taking all GPU memory
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            logger.debug(f"Set memory growth for device {device}")
    except Exception as e:
        logger.warning(f"Could not configure GPU memory growth: {e}")
    
    try:
        model = create_hed_model()
        logger.success("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load HED model: {e}")
        logger.exception("Error details:")
        raise

def enhance_edges(edges, alpha=1.5, beta=0):
    """Enhance edge detection results for better visibility"""
    # Apply contrast enhancement
    enhanced = cv2.convertScaleAbs(edges, alpha=alpha, beta=beta)
    return enhanced

def process_frame(frame, roi, model):
    """Process a single frame to detect burrs inside the specified ROI using HED"""
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]
    
    # Make a copy for visualization
    visual_frame = frame.copy()
    
    # Create a rectangle around the ROI
    cv2.rectangle(visual_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    try:
        # Preprocessing steps to improve edge detection
        logger.debug("Preprocessing ROI with CLAHE and bilateral filtering")
        # 1. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced_roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 2. Apply bilateral filter to reduce noise while preserving edges
        smoothed = cv2.bilateralFilter(enhanced_roi, 9, 75, 75)
        
        # Prepare input for the model
        input_frame = cv2.resize(smoothed, (224, 224))
        input_frame = np.expand_dims(input_frame, axis=0)
        
        # Run prediction
        logger.debug("Running edge detection model inference")
        start_time = time.time()
        edges = model.predict(input_frame, verbose=0)[0, :, :, 0]
        inference_time = time.time() - start_time
        logger.debug(f"Edge detection inference completed in {inference_time:.4f} seconds")
        
        # Resize to original ROI size
        edges = cv2.resize(edges, (w, h))
        edges = (edges * 255).astype(np.uint8)
        
        # Enhance edges for better visibility
        edges = enhance_edges(edges, alpha=1.5, beta=10)
        
        # Apply adaptive thresholding instead of global thresholding
        logger.debug("Applying adaptive thresholding")
        binary = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Advanced morphological operations
        logger.debug("Applying morphological operations")
        kernel = np.ones((3, 3), np.uint8)
        # Remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        # Connect nearby edges
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Dilate to enhance edge connectivity
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        # Find contours with hierarchy to better handle nested contours
        logger.debug("Finding contours")
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        logger.debug(f"Found {len(contours)} raw contours")
        
        # Filter contours by area and shape
        min_area = 20
        max_area = 500
        filtered_contours = []
        
        logger.debug("Filtering contours")
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
                
            # Additional shape filtering
            # Avoid very circular contours (likely not burrs)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.8:  # Threshold for circularity
                    continue
                    
            # Look for elongated shapes (likely burrs)
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            aspect_ratio = max(w_c, h_c) / (min(w_c, h_c) + 1e-5)  # Avoid division by zero
            if aspect_ratio < 1.2:  # Threshold for elongation
                continue
                
            filtered_contours.append(contour)
        
        logger.debug(f"Detected {len(filtered_contours)} potential burrs after filtering")
        
        # Draw contours with different style
        result = roi_frame.copy()
        cv2.drawContours(result, filtered_contours, -1, (0, 0, 255), 2)
        
        # Add text for each detected burr
        for i, contour in enumerate(filtered_contours):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(result, f"Burr {i+1}", (cx-20, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Replace ROI in both frames
        frame[y:y+h, x:x+w] = result
        visual_frame[y:y+h, x:x+w] = result
        
        # Create a composite visualization
        # First, resize the edge and binary images to match the frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Create full-size versions of the edge and binary outputs for visualization
        edge_full = np.zeros((frame_height, frame_width), dtype=np.uint8)
        edge_full[y:y+h, x:x+w] = edges
        edge_color = cv2.cvtColor(edge_full, cv2.COLOR_GRAY2BGR)
        
        binary_full = np.zeros((frame_height, frame_width), dtype=np.uint8)
        binary_full[y:y+h, x:x+w] = binary
        binary_color = cv2.cvtColor(binary_full, cv2.COLOR_GRAY2BGR)
        
        # Add ROI rectangle to edge and binary visualizations
        cv2.rectangle(edge_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(binary_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add labels to each quadrant for clarity
        cv2.putText(visual_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(edge_color, "Edge Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(binary_color, "Binary Threshold", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Burr Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Now all images have the same dimensions, we can safely stack them
        logger.debug(f"Stacking frames for composite - shapes: {visual_frame.shape}, {edge_color.shape}, {binary_color.shape}, {frame.shape}")
        top_row = np.hstack([visual_frame, edge_color])
        bottom_row = np.hstack([binary_color, frame])
        composite = np.vstack([top_row, bottom_row])
        
        # Resize for display if too large
        if composite.shape[1] > 1280:
            scale = 1280 / composite.shape[1]
            composite = cv2.resize(composite, (0, 0), fx=scale, fy=scale)
        
        return frame, edges, binary, composite, len(filtered_contours)
    
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        logger.exception("Error details:")
        
        # Return a default frame in case of error
        error_frame = frame.copy()
        cv2.putText(error_frame, "Error processing frame", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Create a simple composite for error case
        error_composite = np.vstack([np.hstack([error_frame, error_frame]), 
                                     np.hstack([error_frame, error_frame])])
        
        return error_frame, np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8), error_composite, 0

def main():
    logger.info("Starting burr detection application")
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Load model
        logger.info("Loading HED model...")
        model = load_hed_model()
        
        # Video path
        video_path = "data/raw/recording_20250324_111847.avi"
        logger.info(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Create output directory if it doesn't exist
        os.makedirs("data/results", exist_ok=True)
        logger.debug("Created output directory: data/results")
        
        # Create output video writer
        output_path = "data/results/burr_detection_hed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        logger.debug(f"Created output video writer: {output_path}")
        
        # Create composite video writer (4-up display)
        composite_output_path = "data/results/burr_detection_composite.mp4"
        composite_width = frame_width * 2
        composite_height = frame_height * 2
        composite_out = cv2.VideoWriter(composite_output_path, fourcc, fps, 
                                      (composite_width, composite_height))
        logger.debug(f"Created composite video writer: {composite_output_path}")
        
        # Define ROI (adjust these values)
        roi = (100, 100, 200, 200)
        logger.info(f"Using ROI: {roi}")
        
        frame_count = 0
        burr_counts = []
        
        start_time = time.time()
        logger.info("Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video reached")
                break
            
            frame_count += 1
            
            # Log progress
            if frame_count % 10 == 0:
                elapsed_time = time.time() - start_time
                fps_processing = frame_count / elapsed_time
                eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0
                logger.info(f"Processed {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%) - {fps_processing:.2f} FPS - ETA: {eta:.1f}s")
            
            # Process frame
            processed_frame, edges, binary, composite, burr_count = process_frame(frame, roi, model)
            burr_counts.append(burr_count)
            
            if burr_count > 0:
                logger.info(f"Frame {frame_count}: Detected {burr_count} burrs")
            
            # Add frame number and burr count to the frame
            cv2.putText(processed_frame, f"Frame: {frame_count}, Burrs: {burr_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Write to output videos
            out.write(processed_frame)
            composite_out.write(composite)
            
            # Display
            cv2.imshow("Burr Detection Composite", composite)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Processing stopped by user")
                break
        
        # Calculate total processing time
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        logger.info(f"Processing completed: {frame_count} frames in {total_time:.2f} seconds ({avg_fps:.2f} FPS)")
        
        cap.release()
        out.release()
        composite_out.release()
        cv2.destroyAllWindows()
        
        # Generate a simple report
        if burr_counts:
            frames_with_burrs = sum(1 for c in burr_counts if c > 0)
            max_burrs = max(burr_counts)
            avg_burrs = sum(burr_counts) / len(burr_counts)
            
            logger.info("\n----- Burr Detection Report -----")
            logger.info(f"Total frames processed: {frame_count}")
            logger.info(f"Frames with burrs detected: {frames_with_burrs} ({frames_with_burrs/frame_count*100:.1f}%)")
            logger.info(f"Maximum burrs in a single frame: {max_burrs}")
            logger.info(f"Average burrs per frame: {avg_burrs:.2f}")
            logger.info(f"Output saved to {output_path}")
            logger.info(f"Composite visualization saved to {composite_output_path}")
        
        logger.success("Processing complete")
        
    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")
        logger.exception("Error details:")
        return

if __name__ == "__main__":
    import sys
    main()
