import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.app.server import app
from src.config import settings

def main():
    # Disable Dash dev tools so the debug toolbar never renders.
    if hasattr(app, "run"):
        app.run(host=settings.app_host, port=settings.app_port, debug=False)
    else:  # fallback for older Dash
        app.run_server(host=settings.app_host, port=settings.app_port, debug=False)

if __name__ == "__main__":
    main()
