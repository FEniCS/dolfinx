# Building the DOLFINx Python documentation

To build the documentation:

1. Install DOLFINx Python interface using the ``docs`` dependency group, e.g.

       python -m pip install --group docs .

   It must be possible to import the module ``dolfinx`` to build the documentation.

2. Run in this directory:
 
       python -m sphinx -W -b html source/ build/html/

## Rebuilding without regenerating demo images

Demos that save figures are executed during the build so that the images
they reference exist. This dominates the build time. To reuse the images
from an earlier build instead of regenerating them, set
``DOLFINX_DOC_SKIP_DEMO_IMAGES``:

       DOLFINX_DOC_SKIP_DEMO_IMAGES=1 python -m sphinx -W -b html source/ build/html/

The demos are still converted to Markdown and notebooks, so edits to
their text take effect; only the figures are left as they are. The first
build after a clean checkout must run without this set, otherwise the
images are missing and ``-W`` turns the resulting warnings into errors.

## Processing of the demo programs

Python demo programs are written in Python, with MyST-flavoured Markdown syntax
for comments.

1. `jupytext` reads the Python demo code, then converts it to a Markdown file and
   Jupyter notebook.
2. `myst_parser` allows Sphinx to process the Markdown file.
