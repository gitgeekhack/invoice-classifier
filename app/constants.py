"""
This class contains static paths used by the invoice classifier web application.

Class Name: `Path`

Attributes:
- `UPLOAD_PATH` (str): The directory path where uploaded files are stored.
- `PREDICT_PATH` (str): The directory path where predicted output files are stored.
- `MODEL_PATH` (str): The file path of the invoice classifier model.

Class Name: `Dimensions`

Attributes:
- `IMAGE_WIDTH` (int): The width of the images used by the application.
- `IMAGE_HEIGHT` (int): The height of the images used by the application.

Class Name: `Regex`

Attributes:
- `PATTERN` (list of str): A list of regex patterns used to extract dates from text.
- `CLEANED_TEXT` (str): A regex pattern used to clean text data.
- `CAPTURES_PATTERN` (str): A regex pattern used to capture numerical data from text.
- `DATE_FORMAT` (str): The format for dates parsed from text.

Class Name: `Labels`

Attributes:
- `DATE_LABEL` (str): The label used for dates.
- `AMOUNT_LABEL` (list of str): A list of labels used for monetary amounts, including 'MONEY' and 'CARDINAL'.
"""


class Path:
    UPLOAD_PATH = './app/static/upload'
    PREDICT_PATH = './app/static/predict'
    MODEL_PATH = './app/model/invoice_classifier.pt'
    DISTILBERT_PATH = './app/model/distilbert-model'
    TR_OCR_PATH = './app/model/tr-ocr-model'
    XGB_PATH = './app/model/embed-XGB-DOC-CLF.pkl'
    PPOCR_RECN = '/home/darshan/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_slim_infer'
    PPOCR_DETN = '/home/darshan/.paddleocr/whl/det/en/en_PP-OCRv3_det_slim_infer'


class Dimensions:
    IMAGE_WIDTH = 500
    IMAGE_HEIGHT = 700


class Regex:
    PATTERN = [
        "[0-9]{2}/{1}[0-9]{2}/{1}[0-9]{4}",
        "\\d{1,2}-(January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sept|October|Oct|November|Nov|December|Dec)-\\d{4}",
        "\\d{4}-\\d{1,2}-\\d{1,2}",
        "[0-9]{1,2}\\s(January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sept|October|Oct|November|Nov|December|Dec)\\s\\d{4}",
        "\\d{1,2}-\\d{1,2}-\\d{2,4}"]
    CLEANED_TEXT = r'[a-z]:\s'
    CAPTURES_PATTERN = r'\d+'
    DATE_FORMAT_1 = "%Y-%m-%d"
    DATE_FORMAT_2 = "%Y-%d-%m"


class Labels:
    DATE_LABEL = 'DATE'
    AMOUNT_LABEL = ['MONEY', 'CARDINAL']
    INVOICE = 0
    OTHER = 1


class Threshold:
    THRESHOLD_VALUE = 0.96

