import flask
from flask import render_template
from app.services import invoice_app
from flask import Blueprint

"""
This script defines a Flask blueprint for the invoice classifier web application router.

Attributes:
- invoice_web_app (Blueprint): The Flask blueprint for the invoice classifier web application router.

Routes:
- / (GET): Returns the home page for the web application.
- /handle_image_prediction_request (POST): Processes an image prediction request from the user, and returns a page
  displaying the predicted results.
"""


invoice_web_app = Blueprint('router', __name__)


@invoice_web_app.route('/')
async def home():
    return render_template('index.html')


@invoice_web_app.route('/predict', methods=['POST'])
async def pred():
    try:
        base_name_list,  my_dict_list = await invoice_app.handle_image_prediction_request(request=flask.request)
        return render_template('predict.html', base_name=base_name_list, my_dict=my_dict_list)
    except:
        error = await invoice_app.handle_image_prediction_request(request=flask.request)
        return render_template('index.html', error=error)
