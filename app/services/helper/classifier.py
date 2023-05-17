from paddleocr import PaddleOCR
import pickle
from app.services.helper.embedding_generator import EmbeddingsGenerator
from app.constants import Path


class Classifier:
    def __init__(self):
        self.embeddings = EmbeddingsGenerator()
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.xgb_model = pickle.load(open(Path.XGB_PATH, 'rb'))

    def doc_classifier(self, uploaded_image_path):
        result = self.ocr.ocr(uploaded_image_path, cls=True)
        result = result[0]
        text_list = [line[1][0] for line in result]
        text = " ".join(text_list)

        text_embd = self.embeddings.generate_text_embeddings([text])
        invoice_prob = (self.xgb_model.predict_proba(text_embd).tolist()[0][0])
        if invoice_prob >= 0.96:
            return 0
        else:
            return 1
