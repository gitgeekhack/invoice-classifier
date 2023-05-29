"""
The ImageCropMaker class provides a method to crop the image based on the detected coordinates and label ids.
It uses the OCRService and TextHandler classes to extract text from the cropped images.

Methods:

expand(cords, margin): Returns a list of coordinates by expanding the given coordinates by the given margin value.
img_crop_maker(results, img): Takes the result of object detection and the image as input and returns a dictionary
containing extracted text and its label name and confidence score.
Attributes:

ocr_service: An instance of the OCRService class.
text_handler: An instance of the TextHandler class.
"""

import asyncio
import time

from app.services.helper.ocr import OCRService
from app.services.helper.text_handler import TextHandler


class ImageCropMaker:
    def __init__(self):
        self.ocr_service = OCRService()
        self.text_handler = TextHandler()

    async def expand(self, cords, margin):
        # suppose cords is x1, y1, x2, y2
        return [
            cords[0] - margin,
            cords[1] - margin,
            cords[2] + margin,
            cords[3] + margin]

    async def img_crop_maker(self, results, img):
        dic = {}
        df = results.pandas().xyxy[0]
        df = df.loc[df.groupby('name')['confidence'].idxmax()]
        tasks = []

        for ind in range(len(df)):
            cords = df.iloc[ind][0:4]
            label_id = df.iloc[ind][5]

            if label_id == 2:
                cords_exp = await self.expand(cords, margin=4)
                img_crop = img.crop((tuple(cords_exp)))
                tasks.append(self.ocr_service.tr_ocr(img_crop=img_crop, label_id=label_id))

            elif label_id == 0:
                cords_exp = await self.expand(cords, margin=5)
                img_crop = img.crop((tuple(cords_exp)))
                tasks.append(self.ocr_service.tr_ocr(img_crop=img_crop, label_id=label_id))

            elif label_id == 1:
                cords_exp = await self.expand(cords, margin=6)
                img_crop = img.crop((tuple(cords_exp)))
                tasks.append(self.ocr_service.tr_ocr(img_crop=img_crop, label_id=label_id))

        results = await asyncio.gather(*tasks)

        for ind in range(len(results)):
            text, _ = results[ind]
            label_id = df.iloc[ind][5]
            label_name = df.iloc[ind][6]
            label_confidence = float(round(df.iloc[ind][4], 2))

            if label_id == 2:
                merchant_text = await self.text_handler.text_handle_func(text, label_id)
                dic[label_name] = [merchant_text, label_confidence]

            elif label_id == 0:
                invoice_text = await self.text_handler.text_handle_func(text, label_id)
                dic[label_name] = [invoice_text, label_confidence]

            elif label_id == 1:
                amount_text = await self.text_handler.text_handle_func(text, label_id)
                dic[label_name] = [amount_text, label_confidence]
        return dic


