# Building the DOLFINx Python documentation

To build the documentation:

1. Install DOLFINx (Python interface). It must be possible to import
   the module ``dolfinx``.
2. Run in this directory:
 
       python -m sphinx -W -b html source/ build/html/

## Processing demo programs

Python demo programs are written in Python, with MyST-flavoured
Markdown syntax for comments.

1. `jupytext` reads the Python demo code, then converts to and writes a
   Markdown file.
2. `myst_parser` allows Sphinx to process the Markdown file.
