import cv2
import numpy as np
import os
import sys
from loguru import logger

class BurrDetector:
    def __init__(self, debug=True, use_gpu=True):
        """
        Initialize the Burr Detector with GPU acceleration and logging
        
        :param debug: Boolean to enable verbose logging and intermediate image saving
        :param use_gpu: Boolean to enable CUDA GPU processing
        """
        self.debug = debug
        self.use_gpu = use_gpu
        
        # Check and initialize GPU support
        self.gpu_available = self.check_gpu_support()
        
        # Create debug directory if not exists
        if self.debug:
            os.makedirs('debug_images', exist_ok=True)
    
    def check_gpu_support(self):
        """
        Check if CUDA GPU is available
        
        :return: Boolean indicating GPU availability
        """
        try:
            # Check if CUDA is available through OpenCV
            if not cv2.cuda.getCudaEnabledDeviceCount():
                logger.warning("OpenCV CUDA support not enabled in the build.")
                return False

            # Try to create a CUDA stream to test GPU support
            cuda_stream = cv2.cuda.Stream()
            logger.info("CUDA GPU is available!")
            return True
        except Exception as e:
            logger.exception(f"CUDA GPU not available: {e}")
            return False
    
    def preprocess_image(self, image):
        """
        Preprocess the input image with optional GPU acceleration
        
        :param image: Input image from camera/video
        :return: Preprocessed grayscale image
        """
        try:
            if self.use_gpu and self.gpu_available:
                # GPU-accelerated preprocessing
                # Upload image to GPU
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(image)
                
                # Convert to grayscale on GPU
                gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
                
                # Apply Gaussian blur on GPU
                gpu_blurred = cv2.cuda.GaussianBlur(
                    gpu_gray, 
                    (5, 5), 
                    0
                )
                
                # Canny edge detection on GPU
                gpu_edges = cv2.cuda.Canny(gpu_blurred, 50, 150)
                
                # Download results back to CPU
                edges = gpu_edges.download()
            else:
                # CPU-based preprocessing
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
            
            # Morphological operations (keeping on CPU for simplicity)
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            
            if self.debug:
                cv2.imwrite('debug_images/preprocessed.jpg', eroded)
            
            return eroded
        
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise
    
    def find_workpiece_contours(self, preprocessed_image):
        """
        Detect and filter contours of the workpiece
        
        :param preprocessed_image: Preprocessed binary image
        :return: Largest contour (assumed to be the workpiece)
        """
        try:
            # Find contours
            contours, _ = cv2.findContours(
                preprocessed_image, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Find the largest contour (main workpiece)
            if not contours:
                raise ValueError("No contours found")
            
            largest_contour = max(contours, key=cv2.contourArea)
            
            return largest_contour
        
        except Exception as e:
            logger.error(f"Contour detection error: {e}")
            raise
    
    def detect_burrs(self, image, contour):
        """
        Detect burrs by comparing edge irregularities
        
        :param image: Original input image
        :param contour: Main workpiece contour
        :return: Burr mask and burr contours
        """
        try:
            # Create a mask of the workpiece
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Compute convex hull to get ideal smooth boundary
            hull = cv2.convexHull(contour)
            
            # Create hull mask
            hull_mask = np.zeros(image.shape[:2], np.uint8)
            cv2.drawContours(hull_mask, [hull], -1, 255, -1)
            
            # Identify potential burr regions
            diff_mask = cv2.bitwise_xor(mask, hull_mask)
            
            # Find burr contours
            burr_contours, _ = cv2.findContours(
                diff_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter out very small contours (noise)
            significant_burrs = [
                burr for burr in burr_contours 
                if cv2.contourArea(burr) > 5  # Tightened threshold for smaller resolution
            ]
            
            if self.debug:
                debug_burr_img = image.copy()
                cv2.drawContours(debug_burr_img, significant_burrs, -1, (0,255,0), 2)
                cv2.imwrite('debug_images/burr_contours.jpg', debug_burr_img)
            
            return diff_mask, significant_burrs
        
        except Exception as e:
            logger.error(f"Burr detection error: {e}")
            raise
    
    def calculate_burr_confidence(self, image, burr_contours):
        """
        Calculate a confidence score based on the size and number of burr contours.
        
        :param image: Original image
        :param burr_contours: List of burr contours
        :return: Confidence score (0-1)
        """
        if not burr_contours:
            return 0.0
        
        total_burr_area = sum(cv2.contourArea(burr) for burr in burr_contours)
        image_area = image.shape[0] * image.shape[1]
        
        # Normalize burr area by image area
        confidence = total_burr_area / image_area
        
        # Clip to ensure the value is within the range of 0 to 1
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return confidence

    def visualize_burrs(self, image, burr_mask, burr_contours, confidence):
        """
        Create a visualization of detected burrs with confidence score.
        
        :param image: Original input image
        :param burr_mask: Mask of burr regions
        :param burr_contours: Contours of burrs
        :param confidence: Confidence score
        :return: Image with burrs highlighted
        """
        try:
            # Create a semi-transparent red overlay for burr regions
            overlay = image.copy()
            red_mask = np.zeros_like(image)
            red_mask[burr_mask > 0] = [0, 0, 255]  # Red color
            
            # Blend original image with red overlay
            output = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
            
            # Draw contours around burrs
            cv2.drawContours(output, burr_contours, -1, (0, 255, 0), 2)
            
            # Display confidence score
            text = f"Burr Confidence: {confidence:.2f}"
            cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return output
        
        except Exception as e:
            logger.error(f"Burr visualization error: {e}")
            raise
    
    def process_image(self, image_path):
        """
        Complete burr detection pipeline for a single image with real-time display.
        
        :param image_path: Path to input image
        :return: Image with burrs highlighted
        """
        try:
            # Read image and resize
            image = cv2.imread(image_path)
            image = cv2.resize(image, (640, 480))
            
            # Preprocess
            preprocessed = self.preprocess_image(image)
            
            # Find workpiece contour
            workpiece_contour = self.find_workpiece_contours(preprocessed)
            
            # Detect burrs
            burr_mask, burr_contours = self.detect_burrs(image, workpiece_contour)
            
            # Calculate confidence
            confidence = self.calculate_burr_confidence(image, burr_contours)
            
            # Visualize burrs
            result = self.visualize_burrs(image, burr_mask, burr_contours, confidence)
            
            # Display the image in a resizable window
            cv2.namedWindow('Burr Detection', cv2.WINDOW_NORMAL)
            cv2.imshow('Burr Detection', result)
            cv2.waitKey(0)  # Wait until a key is pressed
            cv2.destroyAllWindows()
            
            logger.info("Burr detection completed. Displayed in window.")
            
            return result
        
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            raise
    
    def process_video(self, video_path):
        """
        Process video for burr detection with GPU support and real-time display.
        
        :param video_path: Path to input video
        :return: Processed video writer object
        """
        try:
            # Open video capture
            cap = cv2.VideoCapture(video_path)
            
            # Ensure consistent resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Video writer setup
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('burr_detection_output.avi', fourcc, 20.0, 
                                   (640, 480))
            
            # GPU-specific setup for video processing
            gpu_frame = cv2.cuda_GpuMat() if self.use_gpu and self.gpu_available else None
            
            frame_count = 0
            burr_frames_count = 0
            
            # Create a named window for displaying the video
            cv2.namedWindow('Burr Detection', cv2.WINDOW_NORMAL)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize to ensure consistent 640x480
                frame = cv2.resize(frame, (640, 480))
                
                try:
                    # Preprocess
                    preprocessed = self.preprocess_image(frame)
                    
                    # Find workpiece contour
                    workpiece_contour = self.find_workpiece_contours(preprocessed)
                    
                    # Detect burrs
                    burr_mask, burr_contours = self.detect_burrs(frame, workpiece_contour)
                    
                    # Calculate confidence
                    confidence = self.calculate_burr_confidence(frame, burr_contours)
                    
                    # Check if burrs detected
                    if burr_contours:
                        burr_frames_count += 1
                        
                        # Visualize burrs
                        result = self.visualize_burrs(frame, burr_mask, burr_contours, confidence)
                    else:
                        result = frame
                    
                    # Write to output video
                    out.write(result)
                    
                    # Display the image in the window
                    cv2.imshow('Burr Detection', result)
                    
                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    # Optional: print progress
                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.info(f"Processed {frame_count} frames, Burr frames: {burr_frames_count}")
                    
                except Exception as e:
                    logger.warning(f"Error processing frame {frame_count}: {e}")
            
            # Release resources
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            logger.info(f"Video processing complete. Total frames: {frame_count}, Burr frames: {burr_frames_count}")
            
            return True
        
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            raise

def main():
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Burr Detection System')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to input image or video')
    parser.add_argument('--mode', choices=['image', 'video'], default='image',
                        help='Processing mode: image or video')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU processing')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = BurrDetector(
        debug=args.debug, 
        use_gpu=not args.no_gpu
    )
    
    # Process based on mode
    if args.mode == 'image':
        detector.process_image(args.input)
    else:
        detector.process_video(args.input)

if __name__ == '__main__':
    main()