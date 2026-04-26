"""
Real-Time News RAG — Multi-Agent Research & Credibility System
Entry point — starts the Flask application.
"""


def main():
    """Start the News RAG application."""
    from web.app import create_app
    app = create_app()
    app.run(debug=True, threaded=True, port=5000)


if __name__ == "__main__":
    main()
