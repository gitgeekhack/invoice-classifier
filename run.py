"""
This script runs the Flask application for the invoice classifier web application.

Attributes:
- app (Flask): The Flask application instance for the invoice classifier web application.
"""

from app import app

if __name__ == "__main__":
    app.run('0.0.0.0')
