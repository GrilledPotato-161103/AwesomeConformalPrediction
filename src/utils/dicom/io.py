import os
import zipfile
import numpy as np
import pydicom
from typing import List

# Deployment functions
def unzip_study(study_zip_path: str, src_path: str):
    study_name = os.path.basename(study_zip_path).split(".zip")[0]
    study_path = os.path.join(src_path, study_name)
    if not os.path.exists(study_path):
        os.makedirs(study_path)

    with zipfile.ZipFile(study_zip_path, 'r') as zip_ref:
        zip_ref.extractall(study_path)

    return study_path

def get_dicoms(study_path: str):
    '''
    Get all dicoms path from study_path, return a full image paths list.
    '''

    dicom_paths = []
    for root, _, files in os.walk(study_path):
        for file in files:
            dicom_path = os.path.join(root, file)
            dicom_paths.append(dicom_path)

    return dicom_paths

def sorted_dicom(dicom_paths: List[str], is_filter: bool = True):
    paths = []
    for dicom_path in dicom_paths:
        raw_data = pydicom.dcmread(dicom_path, force=True)

        valid = True
        if is_filter:
            try:
                wc = min(raw_data.WindowCenter)
                ww = max(raw_data.WindowWidth)
                valid = -200 >= wc and wc >= -650 and 1000 <= ww and ww <= 1600
            except:
                valid = False

            if not valid:
                continue

        try:
            img_data = np.array(raw_data.pixel_array)
            if img_data.shape != (512, 512):
                valid = False

            series_number = raw_data.SeriesNumber
            instance_number = raw_data.InstanceNumber
        except:
            valid = False

        if not valid:
            continue

        paths.append(
            (dicom_path, f"{series_number:02d}_{instance_number:04d}"))

    if is_filter and len(paths) <= 100:
        return sorted_dicom(dicom_paths, is_filter=False)

    def key(element):
        return element[1]

    return sorted(paths, key=key)

def check_dicom(dicom_path):
    '''
    Check if dicom_path is ct image dicom without any wrong tag. Return
    '''
    try:
        dcm = pydicom.dcmread(dicom_path)
        dcm_tag = str(dcm.dir)

        contains = ''
        status = True

        # Check "Patient Protocol" tag
        if "Patient Protocol" in dcm_tag:
            status = False
            contains += "Patient Protocol"

        # Check "Dose Report" tag
        if "Dose Report" in dcm_tag:
            status = False
            if len(contains) != 0:
                contains += ", "
            contains += "Dose Report"

        # Check "pdf" tag
        if "pdf" in dcm_tag or "PDF" in dcm_tag:
            status = False
            if len(contains) != 0:
                contains += ", "
            contains += "PDF"

        if not status:
            return False, "Contain " + contains

        # Check "Pixel Array" length
        pixel_array = dcm.pixel_array
        if len(pixel_array) != 512:
            return False, "Length of pixel array is not 512"

        # Check "PixelSpacing"
        pixel_spacing = dcm.get("PixelSpacing", "-1")
        if pixel_spacing == "-1":
            return False, "Don't have pixel spacing"

        # Check "Modality" tag
        modality = dcm.Modality
        if modality != "CT":
            return False, "Is not CT Modality"

        return True, ""

    except Exception as e:
        return False, str(e)

def read_input(dicom_path: str):
    '''
    Read raw dicom data from dicom_path, return pixel_array of that dicom, an unique_id and pixel spacing tag.
    '''
    raw_data = pydicom.dcmread(dicom_path)

    # series_number = raw_data.SeriesNumber
    # instance_number = raw_data.InstanceNumber

    wc = raw_data.WindowCenter
    ww = raw_data.WindowWidth
    if isinstance(wc, pydicom.multival.MultiValue):
        wc = wc[0]
    if isinstance(ww, pydicom.multival.MultiValue):
        ww = ww[0]

    intercept = raw_data.RescaleIntercept
    slope = raw_data.RescaleSlope

    pixel_spacing = raw_data.get('PixelSpacing', np.array([1, 1]))
    raw_data = np.array(raw_data.pixel_array)

    return raw_data, pixel_spacing, intercept, slope, wc, ww