"""Flask application for the Interview Copilot API."""
from flask import Flask
from flask_cors import CORS
from api.routes import api_bp
from config import API_HOST, API_PORT, API_DEBUG

# Create Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Register blueprints
app.register_blueprint(api_bp)

# Root route
@app.route('/')
def index():
    """Root endpoint with API information."""
    return {
        "name": "Interview Copilot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "stats": "/api/stats",
            "match": "/api/match (POST)",
            "generate_questions": "/api/generate-questions (POST)",
            "evaluate_answer": "/api/evaluate-answer (POST)"
        },
        "documentation": "See /api/health for service status"
    }, 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return {"error": "Endpoint not found", "available_endpoints": "/api/health, /api/stats, /api/match, /api/generate-questions, /api/evaluate-answer"}, 404

@app.errorhandler(500)
def internal_error(error):
    return {"error": "Internal server error"}, 500

if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)

# Export app for Flask CLI
__all__ = ['app']

