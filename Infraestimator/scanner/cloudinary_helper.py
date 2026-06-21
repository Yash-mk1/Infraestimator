"""
scanner/cloudinary_helper.py — NEW FILE

Handles uploading images to Cloudinary.
If Cloudinary is not configured, falls back to local disk storage.
"""

import os
import tempfile
import cv2
import numpy as np


def is_cloudinary_configured() -> bool:
    """Check if Cloudinary credentials are set in environment."""
    return all([
        os.environ.get('CLOUDINARY_CLOUD_NAME'),
        os.environ.get('CLOUDINARY_API_KEY'),
        os.environ.get('CLOUDINARY_API_SECRET'),
    ])


def upload_image(image_bgr: np.ndarray, filename: str) -> str:
    """
    Upload a BGR numpy image to Cloudinary.
    Returns the secure URL of the uploaded image.
    Falls back to None if upload fails.
    """
    if not is_cloudinary_configured():
        return None

    try:
        import cloudinary
        import cloudinary.uploader

        # Configure cloudinary from environment variables
        cloudinary.config(
            cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME'),
            api_key    = os.environ.get('CLOUDINARY_API_KEY'),
            api_secret = os.environ.get('CLOUDINARY_API_SECRET'),
            secure     = True
        )

        # Encode image to jpg bytes in memory
        success, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            return None

        # Upload from bytes
        result = cloudinary.uploader.upload(
            buffer.tobytes(),
            public_id       = f'inframonitor/results/{filename}',
            resource_type   = 'image',
            overwrite       = True,
            format          = 'jpg',
        )

        return result.get('secure_url')

    except Exception as e:
        print(f"[Cloudinary] Upload failed for {filename}: {e}")
        return None


