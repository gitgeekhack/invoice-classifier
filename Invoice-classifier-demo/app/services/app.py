import os
from PIL import Image
import torch
from app.services.helper.img_cropper import img_crop_maker
import asyncio
from app.model import model as md

model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path=md.model_path)

async def predict_async(file):
    img_path = os.path.join(md.UPLOAD_PATH, file.filename)
    file.save(img_path)

    img = Image.open(img_path)
    global model
    results = model(img)
    results.save('/home/darshan/Desktop/Invoice-classifier-demo/app/static/predict')

    base_name = os.path.splitext(file.filename)[0]
    pred_img_path = os.path.join(md.PREDICT_PATH, base_name + '.jpg')
    pred_image = Image.open(pred_img_path).resize((500, 700))
    pred_image.save(pred_img_path)

    my_dict = await img_crop_maker(results=results, img=img)

    img = img.resize((500, 700))
    img.save(img_path)

    return base_name, my_dict


async def predict(request):
    files = request.files.getlist('file')
    try:
        tasks = []
        for file in files:
            tasks.append(asyncio.create_task(predict_async(file)))
        results = await asyncio.gather(*tasks)

        base_name_list = [x[0] for x in results]
        my_dict_list = [x[1] for x in results]
        # print(tasks)
        return base_name_list, my_dict_list
    except:
        if not files[0]:
            error = 'IsADirectoryError'
            return error

        # Pass the paths of the original and predicted images to the HTML template
        # return render_template('predict.html', base_name=base_name_list, my_dict=my_dict_list)



