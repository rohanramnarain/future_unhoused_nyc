import os
import sys
from typing import Iterable

from firebase_admin import initialize_app
from firebase_functions import https_fn
from firebase_functions.options import set_global_options

# Prevent cold-start OOMs by asking for more memory and cap concurrency
set_global_options(max_instances=10, memory=512)

# Initialize Firebase Admin SDK (needed if we later read secrets or Firestore)
initialize_app()

# Make sure the project modules (src/, scripts/, etc.) are importable inside the
# Cloud Function bundle. When deploying you must include these directories in the
# same artifact (e.g., copy them under functions/ or set the functions source to
# the repo root) so this path points to real files.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from src.app.server import server as dash_server  # noqa: E402


def _flask_to_https_response(resp):
	"""Convert a Flask response object into firebase_functions.Response."""
	headers: Iterable[tuple[str, str]] = resp.headers.items()
	return https_fn.Response(
		response=resp.get_data(),
		status=resp.status_code,
		headers=dict(headers),
		mimetype=resp.mimetype,
	)


def _dispatch_request(req: https_fn.Request) -> https_fn.Response:
	with dash_server.request_context(req.environ):
		flask_response = dash_server.full_dispatch_request()
	return _flask_to_https_response(flask_response)


@https_fn.on_request(invoker="public")
def future_unhoused_app(req: https_fn.Request) -> https_fn.Response:
	"""Serve the Dash visualization as an HTTPS-triggered Firebase Function."""
	return _dispatch_request(req)