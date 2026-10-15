"""vision.py - extracts location of objects from RGB and depth frames

The main function is `get_positions_from_video_frames`, which recieves
RGB frames(really in BGR format for opencv compatiability) and depth
frames to extract exact location data for objects found by segmentation
model. Currently, FastSAM(small) by Ultralytics is being used with the
ByteTrack multi-object tracking algorithm.
"""

from ultralytics import YOLO, FastSAM
import cv2
import numpy as np
import time


INTRINSIC_MATRIX_VALUES = {
    "fx": 545.1777954101562,
    "fy": 545.1777954101562,
    "cx": 325.6365051269531,
    "cy": 237.0912322998047
}

VISION_MODEL_FILE = "FastSAM-s.pt"
VISION_TRACKER = "bytetrack.yaml"
_vision_model = FastSAM(VISION_MODEL_FILE)

def frames_to_video(frames, file_name):
    frame_width = len(frames[0][0])
    frame_height = len(frames[0])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10 # ARBITRARY, should be the same as topic publishing frequency
    out = cv2.VideoWriter(file_name, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        out.write(frame)

    out.release()

def get_3D_coordinates(coord, z):
    cx, cy = coord

    x = z * (cx - INTRINSIC_MATRIX_VALUES["cx"]) / INTRINSIC_MATRIX_VALUES["fx"]
    y = z * (cy - INTRINSIC_MATRIX_VALUES["cy"]) / INTRINSIC_MATRIX_VALUES["fy"]
    return (x/1000, y/1000, z/1000) # Convert mm to m

def get_avg_3D_pos(mask, depth_frame):
    pixels = np.argwhere(mask == 1)
    avg_coord = np.array([0.0, 0.0, 0.0])
    
    for (row, col) in pixels:
        depth = depth_frame[row][col]
        x, y, = col, row
        real_coord = np.array(get_3D_coordinates((x, y), depth))
        avg_coord += real_coord

    avg_coord = avg_coord / len(pixels)

    return avg_coord

def get_positions_from_video_frames(frames, depth_frames):
    temp_file = "output.mp4"
    frames_to_video(frames, temp_file)
    print("Starting tracking...")
    start = time.perf_counter()
    results = _vision_model.track(source=temp_file, tracker=VISION_TRACKER, conf=0.5, iou=0.9, retina_masks=True, stream=True)
    print(f"Tracking took {time.perf_counter() - start}")
    detections = {}
    class_names = None
    for i, result in enumerate(results):
        class_names = result.names
        img = result.plot()
        cv2.imshow("Image", img)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()

        #print(depth_frames[i].dtype, depth_frames[i].max(), depth_frames[i].min())
        #cv2.imshow("Image", depth_frames[i] / depth_frames[i].max())
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        if result.boxes.id is None:
            continue

        print("Stating to obtain cutouts...")
        start = time.perf_counter()
        for j, _id in enumerate(result.boxes.id):
            mask = result.masks.data[j].numpy()
            bbox = result.boxes.xyxy[j].to(int)
            img = img.orig_img 
            cutout = img[bbox[1]:bbox[3], bbox[0]:bbox[2]] # Get the bounding boxes to make it easier for VLM
            # Use new mask if it is bigger(image clearer) than the old
            if _id not in detections or np.sum(mask) > np.sum(detections[_id]["mask"]):
                detections[_id] = {
                    "mask": mask,
                    "bbox": bbox,
                    "cutout": cutout, # Cutout of object
                    "pos": get_avg_3D_pos(mask, depth_frames[i])
                }

        print("Obaining cutouts took {time.perf_counter() - start}")

    print(class_names)
    return detections, class_names
