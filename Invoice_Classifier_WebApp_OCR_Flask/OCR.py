import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model_ocr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')


def tr_ocr(img_crop, label_id):
    image = np.asarray(img_crop)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if len(image.shape) > 2 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    global model_ocr, processor
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model_ocr.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text, label_id
