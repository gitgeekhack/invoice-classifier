import asyncio
from app.services.helper.ocr import tr_ocr
from app.services.helper.text_handler import text_handle_func


async def expand(cords, margin):
    # suppose cords is x1, y1, x2, y2
    return [
        cords[0] - margin,
        cords[1] - margin,
        cords[2] + margin,
        cords[3] + margin]


async def img_crop_maker(results, img):

    """
        Crops the invoice image into various regions of interest (ROIs) based on the detected bounding boxes.
        The ROIs are passed to the OCR (Optical Character Recognition) function to extract the text information from them.
        The extracted texts are then passed to the text handling function to process the texts for each ROI.
        Args:
            results: detected bounding boxes, confidence scores, and class labels returned by the object detection model
            img: the invoice image to be cropped

        Returns:
            A dictionary containing the processed text information for each ROI, where the keys are the ROI labels and the
            values are lists containing the extracted text and its corresponding confidence score.
        """

    dic = {}
    df = results.pandas().xyxy[0]
    df = df.loc[df.groupby('name')['confidence'].idxmax()]
    tasks = []

    for ind in range(len(df)):
        cords = df.iloc[ind][0:4]
        label_id = df.iloc[ind][5]

        if label_id == 2:
            cords_exp = await expand(cords, margin=4)
            img_crop = img.crop((tuple(cords_exp)))
            tasks.append(asyncio.create_task(tr_ocr(img_crop=img_crop, label_id=label_id)))

        elif label_id == 0:
            cords_exp = await expand(cords, margin=5)
            img_crop = img.crop((tuple(cords_exp)))
            tasks.append(asyncio.create_task(tr_ocr(img_crop=img_crop, label_id=label_id)))

        elif label_id == 1:
            cords_exp = await expand(cords, margin=6)
            img_crop = img.crop((tuple(cords_exp)))
            tasks.append(asyncio.create_task(tr_ocr(img_crop=img_crop, label_id=label_id)))

    results = await asyncio.gather(*tasks)

    for ind in range(len(results)):
        text, _ = results[ind]
        label_id = df.iloc[ind][5]
        label_name = df.iloc[ind][6]
        label_confidence = float(round(df.iloc[ind][4], 2))

        if label_id == 2:
            merchant_text = await text_handle_func(text, label_id)
            dic[label_name] = [merchant_text, label_confidence]

        elif label_id == 0:
            invoice_text = await text_handle_func(text, label_id)
            dic[label_name] = [invoice_text, label_confidence]

        elif label_id == 1:
            amount_text = await text_handle_func(text, label_id)
            dic[label_name] = [amount_text, label_confidence]

    return dic
