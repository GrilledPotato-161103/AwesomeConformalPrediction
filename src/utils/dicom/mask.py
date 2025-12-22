import numpy as np 
import cv2
from typing import Tuple

# Helper
def find_bbox(mask: np.array, ratio: bool = False, ssize = [1, 1]):
    """_summary_
    Args:
        mask (np.ndarray): (h, w) - np.uint8
    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes
    """
    W, H = ssize
    mask = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_bboxes = []
    for contour in contours: 
        (x, y, w, h) = cv2.boundingRect(contour)
        if ratio:
            cnt_bboxes.append((x / W, y / H, w / W, h / H))
        else:
            cnt_bboxes.append((x, y, w, h))
    return contours, cnt_bboxes

def get_bbox(mask: np.ndarray, min_area: int):
    def check_tiny_nodule(bbox: Tuple[int, int, int, int], min_area: int = 25) -> bool:
        """_summary_
        Args:
            bbox (Tuple[int, int, int, int]): (x, y, w, h)
            min_area (int, optional): Defaults to 25.
        Returns:
            bool: True if bbox area < min_area
        """
        return bbox[2] * bbox[3] < min_area

    _, bboxes = find_bbox(mask[0])
    filtered_bboxes = []
    for bbox in bboxes:
        # filter tiny nodule: min_area=25 for image (512x512)
        if check_tiny_nodule(bbox, min_area):
            x, y, w, h = bbox
            mask[0, y: y+h, x: x+w] = 0
        else:
            filtered_bboxes.append(bbox)

    return mask, filtered_bboxes

def lung_loc_postprocess(lung_loc_mask):
    for i in range(lung_loc_mask.shape[-1]):
        mask = lung_loc_mask[:, :, i]
        contours, _ = find_bbox(mask)
        if len(contours):
            max_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [max_contour], -1,
                             (255), thickness=cv2.FILLED)
            lung_loc_mask[:, :, i] = (
                np.array(mask) / 255.0).astype(lung_loc_mask.dtype)

    return lung_loc_mask
