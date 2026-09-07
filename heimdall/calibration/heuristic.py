import logging
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)

def estimate_gsd(image_path: str, assumed_car_length_m: float = 4.5) -> float:
    """
    Estimates the Ground Sample Distance (GSD) in meters/pixel by detecting cars.
    Falls back to a default if no cars are found.
    """
    if YOLO is None:
        logger.warning("ultralytics is not installed. Using default GSD.")
        return 0.3

    logger.info("Running YOLOv8 to heuristically estimate scale...")
    
    # Load YOLOv8n (nano) - will auto-download if not present
    # COCO dataset classes: 2 = car, 7 = truck
    model = YOLO("yolov8n.pt")
    
    # Run inference
    results = model(image_path, classes=[2, 7], verbose=False)
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        logger.warning("No cars/trucks detected. Using default GSD of 0.3 m/px.")
        return 0.3

    boxes = results[0].boxes
    
    # Calculate the max dimension (length) of each detected car box
    # boxes.xywh has format [x_center, y_center, width, height]
    lengths_px = []
    for box in boxes.xywh:
        w, h = box[2].item(), box[3].item()
        lengths_px.append(max(w, h))
        
    # Use the median length to avoid outliers
    median_length_px = float(np.median(lengths_px))
    
    gsd = assumed_car_length_m / median_length_px
    
    logger.info(f"Detected {len(lengths_px)} vehicles. Median length: {median_length_px:.1f} px.")
    logger.info(f"Estimated GSD: {gsd:.3f} meters/pixel (assuming {assumed_car_length_m}m per vehicle).")
    
    return gsd
