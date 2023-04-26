"""
This class contains static paths used by the invoice classifier web application.

Attributes:
- UPLOAD_PATH (str): The directory path where uploaded files are stored.
- PREDICT_PATH (str): The directory path where predicted output files are stored.
- MODEL_PATH (str): The file path of the invoice classifier model.
"""


class Path:
    UPLOAD_PATH = './app/static/upload'
    PREDICT_PATH = './app/static/predict'
    MODEL_PATH = './app/model/invoice_classifier.pt'
