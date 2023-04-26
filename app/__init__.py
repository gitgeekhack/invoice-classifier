from app.manage import create_app
from app.resources.router import invoice_web_app

"""
This script creates and configures a Flask app instance and registers a blueprint for invoice web application.

Returns:
- app (Flask): The configured Flask app instance with the registered blueprint.
"""


app = create_app(debug=True)
app.register_blueprint(invoice_web_app)
