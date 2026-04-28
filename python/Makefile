# Note: mypy is currenlty on CI tested without petsc4py!

PYTHON = python$(py)

lint-mypy:
	$(PYTHON) -m venv dolfinx-venv-mypy
	. dolfinx-venv-mypy/bin/activate && pip install mypy types-cffi scipy-stubs
	. dolfinx-venv-mypy/bin/activate && mypy -p dolfinx
	. dolfinx-venv-mypy/bin/activate && mypy test
	. dolfinx-venv-mypy/bin/activate && mypy demo

lint-mypy-clean:
	rm -rf dolfinx-venv-mypy

lint: lint-mypy

clean: lint-mypy-clean
