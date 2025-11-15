import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.app.server import app
from src.config import settings

def main():
    # Dash ≥2.17
    if hasattr(app, "run"):
        app.run(host=settings.app_host, port=settings.app_port, debug=True)
    else:  # fallback for older Dash
        app.run_server(host=settings.app_host, port=settings.app_port, debug=True)

if __name__ == "__main__":
    main()
