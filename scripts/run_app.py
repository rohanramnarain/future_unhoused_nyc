from src.app.server import app
from src.config import settings


if __name__ == "__main__":
    app.run_server(host=settings.app_host, port=settings.app_port, debug=True)