Dolfin documentation
====================

Run "make html" to build the web pages. This will run doxygen to generate xml
files documenting the C++ code. These files are used by Sphinx extension
"breathe" to autogenerate documentation. A python script, generate_api_rst.pym
splits this into multiple RST pages.

You can also run "ONLY_SPHINX=1 make html" to skip the doxygen part if you have
not changed any C++ file since the last time you built the documentation. This
is usefull if you are just working on the *.rst files and want fast documentation
generation to check your work.

TODO: fix Sphinx API docs for the Python code and describe how it all fits
together with the mocking of .so files and everything here.

