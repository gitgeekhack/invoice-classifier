"""
This code snippet creates an instance of the Flask app using the create_app function, and an instance of
the InvoiceWebApp class from app.resources.router. It then registers the routes defined in the invoice_router method of
InvoiceWebApp with the Flask app instance.
"""
from app.manage import create_app
from app.resources.router import InvoiceWebApp

app = create_app(debug=True)
web_app = InvoiceWebApp()
app.register_blueprint(web_app.invoice_router())

