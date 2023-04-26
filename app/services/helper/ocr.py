import asyncio
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model_ocr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')


async def tr_ocr(img_crop, label_id):

    """
    Takes an image crop and label ID as input, and returns the generated text and label ID after running optical
    character recognition (OCR) on the image using the TrOCR model.

    Args:
        img_crop (numpy.ndarray): The image crop on which OCR should be performed.
        label_id (int): The label ID corresponding to the image crop.

    Returns:
        tuple: A tuple containing the generated text and the label ID. The generated text is a string, and the label
        ID is an integer.
    """

    image = np.asarray(img_crop)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if len(image.shape) > 2 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    global model_ocr, processor
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = await asyncio.get_running_loop().run_in_executor(None, model_ocr.generate, pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text, label_id
