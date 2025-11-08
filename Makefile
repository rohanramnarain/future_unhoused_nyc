PY=python


setup:
$(PY) -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt


install-advanced:
pip install -r requirements-advanced.txt


data:
$(PY) scripts/bootstrap_data.py


train:
$(PY) scripts/train_baseline.py


app:
$(PY) scripts/run_app.py