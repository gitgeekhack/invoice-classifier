import os
from PIL import Image
import torch
from app.services.helper.img_cropper import img_crop_maker
import asyncio
from app.constants import Path


"""
This module contains functions for making predictions on images using a custom YOLOv5 model.

Functions:
    - predict_image: Asynchronously predicts on a single image file and returns a dictionary.
    - handle_image_prediction_request: Predicts on multiple image files using asyncio and returns a list of base names
      and a list of dictionaries.

Args:
    - file: A file object containing an image file to handle_image_prediction_request on.
    - request: An HTTP request containing multiple image files to handle_image_prediction_request on.

Returns:
    - predict_image: A tuple containing the base name of the image file and a dictionary.
      Example: {'Amount': ['892', 0.65], 'Invoice Date': ['2018-04-03', 0.9], 'Merchant Name': ['NITYANAND', 0.91]}
      
    - handle_image_prediction_request: A tuple containing a list of base names for each predicted image and a list of dictionaries.
      Example: [{'Amount': ['1708.00', 0.84], 'Invoice Date': ['2014-09-24', 0.9], 'Merchant Name': ['MOCAMBO RESTAURANT', 0.9]},
      {'Amount': ['3715', 0.86], 'Invoice Date': ['2016-02-28', 0.91], 'Merchant Name': ['MIRCHI & MIME', 0.88]},
      {'Amount': ['892', 0.65], 'Invoice Date': ['2018-04-03', 0.9], 'Merchant Name': ['NITYANAND', 0.91]}]
"""


model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path=Path.MODEL_PATH)


async def predict_image(file):
    img_path = os.path.join(Path.UPLOAD_PATH, file.filename)
    file.save(img_path)

    img = Image.open(img_path)
    global model
    results = model(img)
    results.save(Path.PREDICT_PATH)

    base_name = os.path.splitext(file.filename)[0]
    pred_img_path = os.path.join(Path.PREDICT_PATH, base_name + '.jpg')
    pred_image = Image.open(pred_img_path).resize((500, 700))
    pred_image.save(pred_img_path)

    my_dict = await img_crop_maker(results=results, img=img)

    img = img.resize((500, 700))
    img.save(img_path)

    return base_name, my_dict


async def handle_image_prediction_request(request):
    files = request.files.getlist('file')
    try:
        tasks = []
        for file in files:
            tasks.append(asyncio.create_task(predict_image(file)))
        results = await asyncio.gather(*tasks)

        base_name_list = [x[0] for x in results]
        my_dict_list = [x[1] for x in results]
        return base_name_list, my_dict_list

    except Exception as error:
        return type(error).__name__

        # Pass the paths of the original and predicted images to the HTML template
        # return render_template('handle_image_prediction_request.html', base_name=base_name_list, my_dict=my_dict_list)



