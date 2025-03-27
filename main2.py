import cv2
import numpy as np
import os
from loguru import logger
import argparse
from ultralytics import RTDETR

class OilHoleBurrDetector:
    def __init__(self, debug=False, model_path=None, use_rtdetr=False, conf_threshold=0.3):
        """
        Initialize Oil Hole Burr Detector
        
        :param debug: Enable debug visualization
        :param model_path: Path to the trained RTDETR model
        :param use_rtdetr: Boolean to enable RTDETR-based burr detection
        :param conf_threshold: Confidence threshold for RTDETR detections
        """
        self.debug = debug
        self.use_rtdetr = use_rtdetr
        self.conf_threshold = conf_threshold
        
        # Load RTDETR model if enabled
        if self.use_rtdetr and model_path:
            try:
                self.model = RTDETR(model_path)
                logger.info(f"RTDETR model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Error loading RTDETR model: {e}")
                self.model = None
                self.use_rtdetr = False  # Disable RTDETR if loading fails
        else:
            self.model = None
            if self.use_rtdetr:
                logger.warning("RTDETR is enabled, but no model path provided.")
        
        # Create debug directory if not exists
        if self.debug:
            os.makedirs('debug_images', exist_ok=True)
    
    def preprocess_image(self, image):
        """
        Preprocess image for oil hole detection
        
        :param image: Input image
        :return: Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def detect_oil_holes(self, image):
        """
        Detect potential oil holes using circular Hough Transform
        
        :param image: Preprocessed grayscale image
        :return: List of detected oil hole circles
        """
        # Detect circles using Hough Transform
        try:
            circles = cv2.HoughCircles(
                image, 
                cv2.HOUGH_GRADIENT, 
                dp=1,  # Inverse ratio of accumulator resolution
                minDist=50,  # Minimum distance between detected centers
                param1=50,   # Upper threshold for edge detection
                param2=30,   # Threshold for center detection
                minRadius=10,  # Minimum circle radius
                maxRadius=100  # Maximum circle radius
            )
            
            # If no circles found, return empty list
            if circles is None:
                return []
            
            # Convert to integer coordinates
            circles = np.uint16(np.around(circles))
            
            if self.debug:
                # Draw detected circles for visualization
                debug_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
                for (x, y, r) in circles[0, :]:
                    # Draw the outer circle
                    cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
                    # Draw the center of the circle
                    cv2.circle(debug_img, (x, y), 2, (0, 0, 255), 3)
                
                # Save debug image without using matplotlib
                cv2.imwrite('debug_images/detected_oil_holes.jpg', debug_img)
            
            return circles[0, :]
        except Exception as e:
            logger.error(f"Error detecting oil holes: {e}")
            return []
    
    def detect_burrs_in_oil_holes(self, image, oil_holes):
        """
        Detect burrs specifically within oil holes, using RTDETR if enabled
        
        :param image: Original image
        :param oil_holes: Detected oil hole circles
        :return: List of (burr_mask, x, y, r, confidence) tuples for oil holes with burrs
        """
        burr_results = []
        
        for (x, y, r) in oil_holes:
            try:
                # Ensure coordinates are within image bounds
                if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                    continue
                
                # Create circular mask for the oil hole
                circle_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.circle(circle_mask, (x, y), r, 255, -1)
                
                if self.use_rtdetr and self.model:
                    # Use RTDETR to detect burrs within the oil hole
                    roi_x = max(0, int(x - r * 1.2))  # Increase ROI size
                    roi_y = max(0, int(y - r * 1.2))  # Increase ROI size
                    roi_width = min(image.shape[1] - roi_x, int(2.4*r))  # Increase ROI size
                    roi_height = min(image.shape[0] - roi_y, int(2.4*r))  # Increase ROI size
                    
                    # Skip if ROI is too small
                    if roi_width <= 0 or roi_height <= 0:
                        continue
                    
                    # Extract ROI
                    roi = image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
                    
                    # Run inference on the ROI
                    results = self.model.predict(roi, conf=self.conf_threshold, verbose=False)
                    
                    # Process the results
                    
                    for result in results:
                        if result.boxes:
                            for box in result.boxes:
                                # Get the bounding box coordinates
                                b = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = map(int, b)
                                
                                # Get the confidence score
                                confidence = box.conf[0].cpu().numpy()

                                # Create a mask for the burr using the bounding box
                                burr_mask = np.zeros(image.shape[:2], dtype=np.uint8)

                                # Shift box coordinates to original image space
                                x1 += roi_x
                                y1 += roi_y
                                x2 += roi_x
                                y2 += roi_y
                                
                                cv2.rectangle(burr_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)

                                # Only keep burrs within the circle
                                burr_mask = cv2.bitwise_and(burr_mask, circle_mask)
                    
                                # Add to results if there are any burrs
                                if np.any(burr_mask):
                                    burr_results.append((burr_mask, x, y, r, confidence))
                
                else:
                    # Use traditional image processing for burr detection
                    # Convert to grayscale for edge detection
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Create safe ROI bounds (prevent overflow)
                    roi_x = max(0, int(x - r))
                    roi_y = max(0, int(y - r))
                    roi_width = min(image.shape[1] - roi_x, int(2*r))
                    roi_height = min(image.shape[0] - roi_y, int(2*r))
                    
                    # Skip if ROI is too small
                    if roi_width <= 0 or roi_height <= 0:
                        continue
                    
                    # Extract ROI
                    roi = gray[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
                    
                    # Edge detection within ROI
                    edges = cv2.Canny(roi, 50, 150)
                    
                    # Find contours in the ROI
                    contours, _ = cv2.findContours(
                        edges, 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    # Filter potential burrs
                    potential_burrs = [
                        contour for contour in contours 
                        if cv2.contourArea(contour) > 10  # Minimum area threshold
                    ]
                    
                    # Create a burr mask for this oil hole
                    if potential_burrs:
                        # Create a mask for burrs in original image coordinates
                        burr_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                        
                        # Adjust contour coordinates to original image space
                        adjusted_contours = []
                        for contour in potential_burrs:
                            # Shift contour points to original image coordinates
                            adjusted_contour = contour.copy()
                            adjusted_contour[:,:,0] += roi_x
                            adjusted_contour[:,:,1] += roi_y
                            adjusted_contours.append(adjusted_contour)
                        
                        # Draw contours on the mask
                        cv2.drawContours(burr_mask, adjusted_contours, -1, 255, -1)
                        
                        # Only keep burrs within the circle
                        burr_mask = cv2.bitwise_and(burr_mask, circle_mask)
                        
                        # Add to results if there are any burrs
                        if np.any(burr_mask):
                            burr_results.append((burr_mask, x, y, r, 0.0)) # Add 0.0 confidence for traditional method
                
            except Exception as e:
                logger.error(f"Error processing oil hole at ({x},{y}): {e}")
                continue
        
        return burr_results
    
    def visualize_results(self, image, oil_holes, burr_results):
        """
        Create visualization of detected burrs by outlining them in red closely
        
        :param image: Original image
        :param oil_holes: Detected oil hole circles
        :param burr_results: List of (burr_mask, x, y, r, confidence) tuples for oil holes with burrs
        :return: Annotated image
        """
        result = image.copy()
        
        # Outline each burr in red
        for burr_mask, x, y, r, confidence in burr_results:
            # Find contours in the burr mask
            contours, _ = cv2.findContours(burr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw the contours on the original image
            cv2.drawContours(result, contours, -1, (0, 0, 255), 2)  # Red color
        
        return result
    
    def process_image(self, image_path, display=True):
        """
        Full pipeline for oil hole burr detection on a single image
        
        :param image_path: Path to input image
        :param display: Whether to display the result in a window
        :return: Burr detection results and visualized image
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return None, None
            
            # Preprocess
            preprocessed = self.preprocess_image(image)
            
            # Detect oil holes
            oil_holes = self.detect_oil_holes(preprocessed)
            
            # Detect burrs in oil holes
            burr_results = self.detect_burrs_in_oil_holes(image, oil_holes)
            
            # Log burr confidences
            for burr_mask, x, y, r, confidence in burr_results:
                logger.info(f"Burr detected at ({x}, {y}) with confidence: {confidence:.2f}")
            
            # Create visualization
            result_image = self.visualize_results(image, oil_holes, burr_results)
            
            # Display results in console
            logger.info(f"Detected {len(oil_holes)} oil holes, {len(burr_results)} with burrs")
            
            # Save result image
            output_path = 'oil_hole_detection_result.jpg'
            cv2.imwrite(output_path, result_image)
            logger.info(f"Result saved to {output_path}")
            
            # Display in window if requested
            if display:
                cv2.namedWindow('Oil Hole Burr Detection', cv2.WINDOW_NORMAL)
                cv2.imshow('Oil Hole Burr Detection', result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return burr_results, result_image
            
        except Exception as e:
            logger.exception(f"Error processing image: {e}")
            return None, None
    
    def process_video(self, video_path):
        """
        Process video for oil hole burr detection
        
        :param video_path: Path to input video
        :return: True if processing completed successfully
        """
        try:
            # Open video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return False
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Video writer setup
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('oil_hole_detection_output.avi', fourcc, fps, (width, height))
            
            # Create window for display
            cv2.namedWindow('Oil Hole Burr Detection', cv2.WINDOW_NORMAL)
            
            frame_count = 0
            burr_frames_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Preprocess frame
                    preprocessed = self.preprocess_image(frame)
                    
                    # Detect oil holes
                    oil_holes = self.detect_oil_holes(preprocessed)
                    
                    # Detect burrs in oil holes
                    burr_results = self.detect_burrs_in_oil_holes(frame, oil_holes)
                    
                    # Log burr confidences
                    for burr_mask, x, y, r, confidence in burr_results:
                        logger.info(f"Burr detected at ({x}, {y}) with confidence: {confidence:.2f}")
                    
                    # Create visualization
                    result_frame = self.visualize_results(frame, oil_holes, burr_results)
                    
                    # Count frames with burrs
                    if burr_results:
                        burr_frames_count += 1
                    
                    # Write to output video
                    out.write(result_frame)
                    
                    # Display the frame
                    cv2.imshow('Oil Hole Burr Detection', result_frame)
                    
                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    # Track progress
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log less frequently for better performance
                        logger.info(f"Processed {frame_count} frames, Burr frames: {burr_frames_count}")
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}")
                    # Continue to next frame on error
                    continue
            
            # Release resources
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            logger.info(f"Video processing complete. Total frames: {frame_count}, Burr frames: {burr_frames_count}")
            return True
            
        except Exception as e:
            logger.exception(f"Video processing error: {e}")
            return False


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Oil Hole Burr Detection')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to input image or video')
    parser.add_argument('--mode', choices=['image', 'video'], default='image',
                        help='Processing mode: image or video')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with visualizations')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the trained RTDETR model')
    parser.add_argument('--use_rtdetr', action='store_true',
                        help='Enable RTDETR-based burr detection')
    parser.add_argument('--conf', type=float, default=0.03,
                        help='Confidence threshold for RTDETR detections')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = OilHoleBurrDetector(
        debug=args.debug, 
        model_path=args.model,
        use_rtdetr=args.use_rtdetr,
        conf_threshold=args.conf
    )
    
    # Process based on mode
    if args.mode == 'image':
        detector.process_image(args.input)
    else:
        detector.process_video(args.input)

if __name__ == '__main__':
    main()