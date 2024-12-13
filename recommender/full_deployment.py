import warnings
import pandas as pd
import re
import json
from Object_Detection.utils.object_localization import ocr_receipt
from Object_Detection.utils.vertex_extract_dict import extract_dict as ved
from recommender.utils.product_recommender import recommend as pr
from recommender.utils.cheap_close import cheap_proximity_rec as cc
import shutil
import os

def full_deployment(key_path: str, test_path: str, dataset_path: str, uid: str, email: str, model, lon: float, lat: float):
    """
    Full deployment pipeline for processing OCR, validating data, and generating recommendations.

    Args:
        key_path (str): Path to the key file.
        test_path (str): Path to the test receipt image.
        dataset_path (str): Path to the dataset CSV file.
        uid (str): User ID.
        email (str): User's email address.
        model: OCR model for processing the receipt.
        lon (float): Longitude for proximity recommendations.
        lat (float): Latitude for proximity recommendations.

    Returns:
        dict: Recommendations for the user.

    Raises:
        ValueError: If input data or files are invalid.
        RuntimeError: If JSON decoding fails after retries.
    """
    # Ensure dataset exists and is readable
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise ValueError(f"Failed to read dataset: {e}")

    # Validate dataset structure
    if df.empty:
        raise ValueError("DataFrame is empty. Please check the dataset file.")
    required_columns = {'uid', 'long', 'lat'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset missing required columns: {required_columns - set(df.columns)}")
    if df[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns.")

    # Validate email format
    email_regex = r"^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,})+$"
    if re.fullmatch(email_regex, email) is None:
        raise ValueError("Email is not valid.")

    # Validate model presence
    if model is None:
        raise ValueError("The model parameter is required but not provided.")

    # Suppress warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Perform OCR and process data
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            struk = ocr_receipt(test_path, model)
            data = ved(struk, key_path, uid, email)
            if not data:
                raise ValueError("VED function returned no data.")
            data = pd.DataFrame(data)
            break
        except json.JSONDecodeError as e:
            if attempt == max_retries:
                raise RuntimeError(f"Failed to decode JSON after {max_retries + 1} attempts: {e}")
            else:
                print(f"JSONDecodeError encountered on attempt {attempt + 1}: {e}. Retrying...")

    # Validate new data structure
    if not all(df.columns == data.columns):
        raise ValueError("Columns of the existing dataset and new data do not match.")

    # Backup dataset
    backup_path = f"{dataset_path}.backup"
    try:
        shutil.copyfile(dataset_path, backup_path)
        print(f"Backup created at {backup_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to create backup: {e}")

    # Append new data and save
    try:
        df = pd.concat([df, data], ignore_index=True)
        df.to_csv(dataset_path, index=False)
        print(f"Dataset updated and saved at {dataset_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save updated dataset: {e}")

    # Generate recommendations
    try:
        test_rec = pr(dataset_path, uid)
        end_rec = cc(
            dataset=dataset_path,
            uid=uid,
            product_list=test_rec,
            lon=lon,
            lat=lat
        )
    except Exception as e:
        raise RuntimeError(f"Failed to generate recommendations: {e}")

    return end_rec
