# Building the DOLFINx Python documentation

To build the documentation:

1. Install DOLFINx Python interface using the ``docs`` optional dependency set, e.g.

       python -m pip install .[docs]

   It must be possible to import the module ``dolfinx`` to build the documentation.

2. Run in this directory:
 
       python -m sphinx -W -b html source/ build/html/

## Processing of the demo programs

Python demo programs are written in Python, with MyST-flavoured Markdown syntax
for comments.

1. `jupytext` reads the Python demo code, then converts it to a Markdown file and
   Jupyter notebook.
2. `myst_parser` allows Sphinx to process the Markdown file.
