"""
The OCRService class is responsible for performing OCR on an image using the microsoft/trocr-base-printed model from
the Hugging Face Transformers library.

Attributes:
processor: TrOCRProcessor object, used for preprocessing images before feeding them to the OCR model.
model_ocr: VisionEncoderDecoderModel object, the OCR model that generates text from an image.

Methods:
tr_ocr(img_crop, label_id): An asynchronous method that takes an image crop and a label ID as input.
The img_crop argument should be a NumPy array representing an image crop, and label_id is an integer representing
the label ID of the corresponding OCR label. This method performs OCR on the image crop by first preprocessing it using
the processor, and then feeding it to the OCR model. The generated text is returned along with the label_id.
"""
import time

import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
from app.constants import Path

class OCRService:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained(Path.TR_OCR_PATH)
        self.model_ocr = VisionEncoderDecoderModel.from_pretrained(Path.TR_OCR_PATH)

    async def tr_ocr(self, img_crop, label_id):
        start = time.time()
        image = np.asarray(img_crop)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if len(image.shape) > 2 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model_ocr.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text, label_id



