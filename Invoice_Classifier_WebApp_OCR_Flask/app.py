from flask import Flask, render_template, request
import os
from PIL import Image
import torch
from img_cropper import img_crop_maker

app = Flask(__name__)
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, '/home/kelvin/Desktop/Flask/Invoice_Classifier_WebApp_OCR_Flask/static/upload')
PREDICT_PATH = os.path.join(BASE_PATH, '/home/kelvin/Desktop/Flask/Invoice_Classifier_WebApp_OCR_Flask/static/predict')


@app.route('/')
def home():
    return render_template('index.html')


model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path='model/best.pt')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        files = request.files.getlist('file')
        if not files[0]:
            error = 'IsADirectoryError'
            return render_template('index.html', error=error)

        dict_list=[]
        base_name_list=[]
        for file in files:
            img_path = os.path.join(UPLOAD_PATH, file.filename)
            file.save(img_path)

            img = Image.open(img_path)
            global model
            results = model(img)
            results.save('static/predict')

            base_name = os.path.splitext(file.filename)[0]
            base_name_list.append(base_name)
            pred_img_path = os.path.join(PREDICT_PATH, base_name + '.jpg')
            pred_image = Image.open(pred_img_path).resize((500, 700))
            pred_image.save(pred_img_path)

            dic = img_crop_maker(results=results, img=img)
            dict_list.append(dic)
            img = img.resize((500, 700))
            img.save(img_path)
        print(base_name_list)
        print(dict_list)
            # Pass the paths of the original and predicted images to the HTML template
        return render_template('predict.html', base_name=base_name_list, dict=dict_list)

    except Exception as error:
        # error = "Oops!! Invalid Image"
        return render_template('index.html', error=error)


if __name__ == '__main__':
    app.run(debug=True)
