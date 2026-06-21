"""
scanner/cloudinary_helper.py
"""

import os
import cv2
import numpy as np


def is_cloudinary_configured() -> bool:
    return all([
        os.environ.get('CLOUDINARY_CLOUD_NAME'),
        os.environ.get('CLOUDINARY_API_KEY'),
        os.environ.get('CLOUDINARY_API_SECRET'),
    ])


def upload_and_save_local(image_bgr: np.ndarray,
                           filename: str,
                           local_path: str,
                           media_url_prefix: str) -> str:
    """
    Try Cloudinary first. If not configured or fails,
    save to local disk and return local media URL.
    """
    if is_cloudinary_configured():
        url = _upload_to_cloudinary(image_bgr, filename)
        if url:
            return url

    # Fallback: save locally
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    cv2.imwrite(local_path, image_bgr)
    return media_url_prefix + filename


def _upload_to_cloudinary(image_bgr: np.ndarray, filename: str) -> str:
    try:
        import cloudinary
        import cloudinary.uploader

        cloudinary.config(
            cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME'),
            api_key    = os.environ.get('CLOUDINARY_API_KEY'),
            api_secret = os.environ.get('CLOUDINARY_API_SECRET'),
            secure     = True
        )

        success, buffer = cv2.imencode('.jpg', image_bgr,
                                       [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            return None

        result = cloudinary.uploader.upload(
            buffer.tobytes(),
            public_id     = f'inframonitor/results/{filename}',
            resource_type = 'image',
            overwrite     = True,
            format        = 'jpg',
        )

        return result.get('secure_url')

    except Exception as e:
        print(f"[Cloudinary] Upload failed for {filename}: {e}")
        return None