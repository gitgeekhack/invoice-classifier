from OCR import tr_ocr
from text_handler import text_handle_func


def expand(cords, margin):
    # suppose cords is x1, y1, x2, y2
    return [
        cords[0] - margin,
        cords[1] - margin,
        cords[2] + margin,
        cords[3] + margin]


def img_crop_maker(results,img):
    dic = {}
    df = results.pandas().xyxy[0]
    df = df.loc[df.groupby('name')['confidence'].idxmax()]
    # lw, lh = img.size
    for ind in range(len(df)):
        cords = df.iloc[ind][0:4]
        label_id = df.iloc[ind][5]
        label_name = df.iloc[ind][6]
        label_confidence = float(round(df.iloc[ind][4],2))

        if label_id == 2:
            cords_exp = expand(cords, margin=4)
            img_crop = img.crop((tuple(cords_exp)))
            # denoised = cv2.fastNlMeansDenoisingColored(np.asarray(img_crop), None, 10, 10, 7, 15)
            text, _ = tr_ocr(img_crop=img_crop,label_id=label_id)
            merchant_text = text_handle_func(text,label_id)
            dic[label_name] = [merchant_text, label_confidence]

        elif label_id == 0:
            cords_exp = expand(cords, margin=5)
            img_crop = img.crop((tuple(cords_exp)))
            text, _ = tr_ocr(img_crop=img_crop, label_id=label_id)
            invoice_text = text_handle_func(text, label_id)
            dic[label_name] = [invoice_text, label_confidence]

        elif label_id == 1:
            cords_exp = expand(cords, margin=6)
            img_crop = img.crop((tuple(cords_exp)))
            # denoised = cv2.fastNlMeansDenoisingColored(np.asarray(img_crop), None, 10, 10, 7, 15)
            text, _ = tr_ocr(img_crop=img_crop, label_id=label_id)
            amount_text = text_handle_func(text, label_id)
            dic[label_name] = [amount_text, label_confidence]

    return dic

