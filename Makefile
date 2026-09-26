.PHONY: test demo evaluate ablate api dashboard docker-build docker-up clean

# --- These four targets are REAL and TESTED: each just runs a script that
# has already been executed successfully in this repo's build environment. ---

test:
	PYTHONPATH=src python3 -m pytest -q

demo:
	PYTHONPATH=src python3 scripts/run_demo.py

evaluate:
	PYTHONPATH=src python3 evaluation/evaluate.py

ablate:
	PYTHONPATH=src python3 evaluation/ablations.py

# --- These four targets are UNTESTED: they depend on fastapi/streamlit/
# docker, none of which are installable in the sandbox this Makefile was
# written in (no network access). Syntax/structure only. ---

api:
	PYTHONPATH=src python3 -m uvicorn vigil.api.main:app --reload

dashboard:
	PYTHONPATH=src streamlit run src/vigil/ui/dashboard.py

docker-build:
	docker build -t vigil .

docker-up:
	docker compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f data/audit_log.jsonl data/audit.db data/state.db data/api_audit.db
