"""
This code defines a Flask web application that serves an invoice prediction model.
The InvoiceWebApp class contains a ImagePredictor object that is used to handle the prediction of the invoice data.
The invoice_router function creates a Flask blueprint object with two routes:

The '/' route renders the home page of the web application, which is an HTML template file named index.html.
The '/predict' route is used to handle the invoice prediction request.
"""

import flask
from flask import render_template, request, Blueprint
from app.services.invoice_app import ImagePredictor


class InvoiceWebApp:

    def __init__(self):
        self.image_predictor = ImagePredictor()

    def invoice_router(self):
        invoice_web_app = Blueprint('router', __name__)

        @invoice_web_app.route('/')
        async def home():
            return render_template('index.html')

        @invoice_web_app.route('/predict', methods=['POST'])
        async def predict():
            try:
                base_name_list, my_dict_list = await self.image_predictor.handle_image_prediction_request(
                    request=request)
                return render_template('predict.html', base_name=base_name_list, my_dict=my_dict_list)
            except:
                error = await self.image_predictor.handle_image_prediction_request(request=flask.request)
                return render_template('index.html', error=error)

        return invoice_web_app




