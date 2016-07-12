============
Contributing
============

This page provides guidance on how to contribute to DOLFIN.


Adding a demo
=============

The below instructions are for adding a Python demo program to DOLFIN.
DOLFIN demo programs are written in reStructuredText, and converted to
Python/C++ code using ``pylit``. The process for C++ demos is similar.
The documented demo programs are displayed at
http://fenics-dolfin.readthedocs.io/.


Creating the demo program
-------------------------

1. Create a directory for the demo under ``demo/documented/``,
   e.g. ``demo/documented/foo/python/``.
2. Write the demo in reStructuredText (rst), with the actual code in
   'code blocks' (see other demos for guidance). The demo file should
   be named ``demo_foo-bar.py.rst``.
3. Convert the rst file to to a Python file using ``pylit`` (pylit is
   distributed with DOLFIN in ``utils/pylit``)

   .. code-block:: ruby

      ../../../../utils/pylit/pylit.py demo_foo-bar.py.rst

   This will create a file ``demo_foo-bar.py``. Test that the Python
   script can be run.


Adding the demo to the documentation system
-------------------------------------------

1. Add the demo to the list in ``doc/source/demos.rst``.
2. To check how the documentation will be displayed on the web, in
   ``doc/`` run ``make html`` and open the file
   ``doc/build/html/index.html`` in a browser.


Make a pull request
-------------------

1. Create a git branch and add the ``demo_foo-bar.py.rst`` file to the
   repository. Do not add the ``demo_foo-bar.py`` file.
2. If there is no C++ version, edit ``test/regression/test.py`` to
   indicate that there is no C++ version of the demo.
3. Make a pull request at
   https://bitbucket.org/fenics-project/dolfin/pull-requests/ for your
   demo to be considered for addition to DOLFIN. Add the
   ``demo_foo-bar.py.rst`` file to the repository, but do not add the
   ``demo_foo-bar.py`` file.
