"""
The ImagePredictor class contains methods for predicting the content of an uploaded image file. It uses a YOLOv5 model
to detect various regions of interest (ROIs) within the image, and then uses an ImageCropMaker object to extract the
relevant text from each ROI.

Attributes:
model: The YOLOv5 model used for image detection.
img_crop: An instance of the ImageCropMaker class for text extraction.

Methods:
predict_image(file): Given a file, this method saves it to the Path.UPLOAD_PATH directory, and uses the YOLOv5 model to
predict the ROIs within the image. It then saves the resulting image with the predicted ROIs to the Path.PREDICT_PATH
directory, and extracts the relevant text from each ROI using the ImageCropMaker object.

handle_image_prediction_request(request): Given a Flask request object, this method handles a request to predict the
content of one or more uploaded image files.

"""

import os
from PIL import Image
import torch
from app.services.helper.img_cropper import ImageCropMaker
import asyncio
from app.constants import Path, Dimensions


class ImagePredictor:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path=Path.MODEL_PATH)
        self.img_cropper = ImageCropMaker()

    async def predict_image(self, file):
        img_path = os.path.join(Path.UPLOAD_PATH, file.filename)
        file.save(img_path)

        img = Image.open(img_path)
        results = self.model(img)
        results.save(Path.PREDICT_PATH)

        image_name = os.path.splitext(file.filename)[0]
        pred_img_path = os.path.join(Path.PREDICT_PATH, image_name + '.jpg')
        pred_image = Image.open(pred_img_path).resize((Dimensions.IMAGE_WIDTH, Dimensions.IMAGE_HEIGHT))
        pred_image.save(pred_img_path)

        extracted_invoice_info = await self.img_cropper.img_crop_maker(results=results, img=img)

        img = img.resize((Dimensions.IMAGE_WIDTH, Dimensions.IMAGE_HEIGHT))
        img.save(img_path)

        return image_name, extracted_invoice_info

    async def handle_image_prediction_request(self, request):
        files = request.files.getlist('file')
        try:
            tasks = []
            for file in files:
                tasks.append(self.predict_image(file))
            results = await asyncio.gather(*tasks)
            image_names = [name[0] for name in results]
            prediction_results = [info[1] for info in results]
            return image_names, prediction_results

        except Exception as error:
            return type(error).__name__
