"""
This function creates a Flask app with specified configurations.

Parameters:
- debug (bool): Flag to set the Flask app's debug mode. Defaults to False.

Returns:
- app (Flask): The configured Flask app instance.
"""

from flask import Flask


def create_app(debug=False):
    app = Flask(__name__, template_folder="./templates", static_folder="./static")
    app.debug = debug
    return app
