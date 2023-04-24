from flask import Flask, render_template, request
import os
from PIL import Image
import torch
from img_cropper import img_crop_maker
import asyncio

app = Flask(__name__)
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, '/home/darshan/Documents/GIt/invoice-classifier/Invoice_Classifier_WebApp_OCR_AsyncIO/static/upload')
PREDICT_PATH = os.path.join(BASE_PATH, '/home/darshan/Documents/GIt/invoice-classifier/Invoice_Classifier_WebApp_OCR_AsyncIO/static/predict')


@app.route('/')
def home():
    return render_template('index.html')


model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path='model/best.pt')


async def predict_async(file):
    img_path = os.path.join(UPLOAD_PATH, file.filename)
    file.save(img_path)

    img = Image.open(img_path)
    global model
    results = model(img)
    results.save('static/predict')

    base_name = os.path.splitext(file.filename)[0]
    pred_img_path = os.path.join(PREDICT_PATH, base_name + '.jpg')
    pred_image = Image.open(pred_img_path).resize((500, 700))
    pred_image.save(pred_img_path)

    my_dict = await img_crop_maker(results=results, img=img)

    img = img.resize((500, 700))
    img.save(img_path)

    return base_name, my_dict


@app.route('/predict', methods=['POST'])
async def predict():
    try:
        files = request.files.getlist('file')
        if not files[0]:
            error = 'IsADirectoryError'
            return render_template('index.html', error=error)

        tasks = []
        for file in files:
            tasks.append(asyncio.ensure_future(predict_async(file)))

        results = await asyncio.gather(*tasks)

        base_name_list = [x[0] for x in results]
        my_dict_list = [x[1] for x in results]

        # Pass the paths of the original and predicted images to the HTML template
        return render_template('predict.html', base_name=base_name_list, my_dict=my_dict_list)

    except Exception as error:
        # error = "Oops!! Invalid Image"
        return render_template('index.html', error=error)


if __name__ == '__main__':
    asyncio.run(app.run())
