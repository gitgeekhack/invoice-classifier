import time

from paddleocr import PaddleOCR
import pickle
from app.services.helper.embedding_generator import EmbeddingsGenerator
from app.constants import Path, Threshold, Labels

class Classifier:
    def __init__(self):
        self.embeddings = EmbeddingsGenerator()
        self.ocr = PaddleOCR(lang='en', rec_batch_num=2,
                             rec_model_dir=Path.PPOCR_RECN,
                             det_model_dir=Path.PPOCR_DETN,
                             rec_algorithm='CRNN')
        self.xgb_model = pickle.load(open(Path.XGB_PATH, 'rb'))

    def doc_classifier(self, uploaded_image_path):
        result = self.ocr.ocr(uploaded_image_path)
        result = result[0]
        text_list = [line[1][0] for line in result]
        text = " ".join(text_list)

        text_embd = self.embeddings.generate_text_embeddings([text])
        invoice_prob = (self.xgb_model.predict_proba(text_embd).tolist()[0][0])
        if invoice_prob >= Threshold.THRESHOLD_VALUE:
            return Labels.INVOICE
        else:
            return Labels.OTHER


