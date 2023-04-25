import flask
from flask import Flask, render_template
from app.services import app as apk

def create_app(debug=False):
    app = Flask(__name__, template_folder="./templates", static_folder="./static")
    app.debug = debug

    @app.route('/')
    async def home():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    async def pred():
        try:
            base_name_list,  my_dict_list = await apk.predict(request=flask.request)
            return render_template('predict.html', base_name=base_name_list, my_dict=my_dict_list)
        except:
            # error = "Oops!! Invalid Image"
            error = await apk.predict(request=flask.request)
            return render_template('index.html', error=error)

    return app
