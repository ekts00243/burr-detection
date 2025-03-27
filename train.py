import argparse
from ultralytics import RTDETR

def train_rtdetr(dataset_yaml: str, weights: str, epochs: int, batch: int, imgsz: int):
    """
    Train the RTDETR-L model on the specified dataset.
    
    Args:
        dataset_yaml (str): Path to dataset YAML file.
        weights (str): Path to initial weights (e.g., "rtdetr-l.pt").
        epochs (int): Number of training epochs.
        batch (int): Batch size.
        imgsz (int): Input image size.
    """
    # Initialize the model (ensure that rtdetr-l.pt is available)
    model = RTDETR(weights)
    
    # Start training
    model.train(data=dataset_yaml, epochs=epochs, batch=batch, imgsz=imgsz)
    
    # Optionally, you can also export the best checkpoint after training:
    # model.export(format="onnx")  # or other formats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RTDETR-L model on burr dataset.")
    parser.add_argument('--data', type=str, default='dataset.yaml', help="Path to dataset YAML file")
    parser.add_argument('--weights', type=str, default='rtdetr-l.pt', help="Path to initial weights (RTDETR-L checkpoint)")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch', type=int, default=16, help="Batch size")
    parser.add_argument('--imgsz', type=int, default=640, help="Input image size")
    
    args = parser.parse_args()
    
    train_rtdetr(args.data, args.weights, args.epochs, args.batch, args.imgsz)
