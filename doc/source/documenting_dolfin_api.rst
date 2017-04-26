
.. _styleguides_sphinx_documenting_interface:

==================================================
Documenting the interface (Programmer's reference)
==================================================

The DOLFIN :ref:`Programmer's Reference <documentation>` is generated
for the DOLFIN C++ library and Python module from the source code
using the documentation tool `Sphinx
<http://sphinx.pocoo.org/index.html>`_. This page describes how to
generate the DOLFIN documentation locally and how to extend the
contents of the Programmer's Reference.


.. _generate_dolfin_documentation_locally:

How to locally build the DOLFIN documentation
---------------------------------------------

The DOLFIN documentation can be generated and built from the DOLFIN
source directly as follows:

* Make sure that `Sphinx <http://sphinx.pocoo.org/index.html>`_ is
  installed.

* Go to the doc/ directory

* Build the documentation by running:

    make html

* Study the results in doc/build/html or dolfin/swig/*/docstrings.i


How to improve and extend the DOLFIN Programmer's reference
-----------------------------------------------------------

The documentation contents are extracted from specially formatted
comments (docstring comments) in the source code, converted to
`reStructuredText <http://docutils.sourceforge.net/rst.html>`_, and
formatted using `Sphinx <http://sphinx.pocoo.org/index.html>`_. The
syntax used for these specially formatted comments is described below.

To document a feature,

#. Add appropriate docstring comments to source files (see
   :ref:`syntax_for_docstring_comments`).

#. Build the documentation as described in
   :ref:`generate_dolfin_documentation_locally` to check the result.

.. _syntax_for_docstring_comments:

Syntax for docstring comments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Doxygen is used to parse the C++ header files for special comments,
we generally use comments starting with ``///``. For information on
the required format, see the doxygen documentation
http://www.doxygen.org/manual/index.html

In addition we try to support some Sphinx syntax like
``:math:`\lambda_3```. We may add some more special tricks like raw
ReStructuredText passthrough, but in general you should stick to 
normal doxygen syntax

