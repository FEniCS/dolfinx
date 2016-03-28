.. _developers_styleguide_cpp:

FEniCS C++ coding style guide
=============================

Naming conventions
------------------

Class names
^^^^^^^^^^^
Use camel caps for class names:

.. code-block:: c++

    class FooBar
    {
      ...
    };

Function names
^^^^^^^^^^^^^^

Use lower-case for function names and underscore to separate words:

.. code-block:: c++

    foo();
    bar();
    foo_bar(...);

Functions returning a value should be given the name of that value,
for example:

.. code-block:: c++

    class Array:
    {
    public:

      /// Return size of array (number of entries)
      std::size_t size() const;

    };

In the above example, the function should be named ``size`` rather
than ``get_size``. On the other hand, a function not returning a value
but rather taking a variable (by reference) and assigning a value to
it, should use the ``get_foo`` naming scheme, for example:

.. code-block:: c++

    class Parameters:
    {
    public:

      /// Retrieve all parameter keys
      void get_parameter_keys(std::vector<std::string>& parameter_keys) const;

    };


Variable names
^^^^^^^^^^^^^^

Use lower-case for variable names and underscore to separate words:

.. code-block:: c++

    Foo foo;
    Bar bar;
    FooBar foo_bar;

Enum variables and constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enum variables should be lower-case with underscore to separate words:

.. code-block:: c++

    enum Type {foo, bar, foo_bar};

We try to avoid using ``#define`` to define constants, but when
necessary constants should be capitalized:

.. code-block:: c++

    #define FOO 3.14159265358979

File names
^^^^^^^^^^

Use camel caps for file names if they contain the
declaration/definition of a class. Header files should have the suffix
``.h`` and implementation files should have the suffix ``.cpp``:

.. code-block:: c++

    FooBar.h
    FooBar.cpp

Use lower-case for file names that contain utilities/functions (not
classes).

Miscellaneous
-------------

.. _styleguides_cpp_coding_style_indentation:

Indentation
^^^^^^^^^^^

Indentation should be two spaces and it should be spaces. Do **not**
use tab(s).

Comments
^^^^^^^^

Comment your code, and do it often. Capitalize the first letter and
don't use punctuation (unless the comment runs over several
sentences). Here's a good example from ``TopologyComputation.cpp``:

.. code-block:: c++

    // Check if connectivity has already been computed
    if (connectivity.size() > 0)
      return;

    // Invalidate ordering
    mesh._ordered = false;

    // Compute entities if they don't exist
    if (topology.size(d0) == 0)
      compute_entities(mesh, d0);
    if (topology.size(d1) == 0)
      compute_entities(mesh, d1);

    // Check if connectivity still needs to be computed
    if (connectivity.size() > 0)
      return;

    ...

Always use ``//`` for comments and ``///`` for documentation (see
:ref:`styleguides_sphinx_documenting_interface`). Never use ``/*
... */``, not even for comments that runs over multiple lines.

Integers and reals
^^^^^^^^^^^^^^^^^^

Use ``std::size_t`` instead of ``int`` (unless you really want to use
negative integers or memory usage is critical).

.. code-block:: c++

    std::size_t i = 0;
    double x = 0.0;

Placement of brackets and indent style
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the `BSD/Allman <http://en.wikipedia.org/wiki/Indent_style>`_
style when formatting blocks of code, i.e., curly brackets following
multiline control statements should appear on the next line and should
not be indented:

.. code-block:: c++

    for (std::size_t i = 0; i < 10; i++)
    {
      ...
    }

For one line statements, omit the brackets:

.. code-block:: c++

    for (std::size_t i = 0; i < 10; i++)
      foo(i);

Header file layout
^^^^^^^^^^^^^^^^^^

Header files should follow the below template:

.. code-block:: c++

    // Copyright (C) 2008 Foo Bar
    //
    // This file is part of DOLFIN.
    //
    // DOLFIN is free software: you can redistribute it and/or modify
    // it under the terms of the GNU Lesser General Public License as published by
    // the Free Software Foundation, either version 3 of the License, or
    // (at your option) any later version.
    //
    // DOLFIN is distributed in the hope that it will be useful,
    // but WITHOUT ANY WARRANTY; without even the implied warranty of
    // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    // GNU Lesser General Public License for more details.
    //
    // You should have received a copy of the GNU Lesser General Public License
    // along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
    //
    // Modified by Bar Foo 2008

    #ifndef __FOO_H
    #define __FOO_H

    namespace dolfin
    {

      class Bar; // Forward declarations here

      /// Documentation of class

      class Foo
      {
      public:

        ...

      private:

        ...

      };

    }

    #endif

Implementation file layout
^^^^^^^^^^^^^^^^^^^^^^^^^^

Implementation files should follow the below template:

.. code-block:: c++

    // Copyright (C) 2008 Foo Bar
    //
    // This file is part of DOLFIN.
    //
    // DOLFIN is free software: you can redistribute it and/or modify
    // it under the terms of the GNU Lesser General Public License as published by
    // the Free Software Foundation, either version 3 of the License, or
    // (at your option) any later version.
    //
    // DOLFIN is distributed in the hope that it will be useful,
    // but WITHOUT ANY WARRANTY; without even the implied warranty of
    // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    // GNU Lesser General Public License for more details.
    //
    // You should have received a copy of the GNU Lesser General Public License
    // along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
    //
    // Modified by Bar Foo 2008

    #include <dolfin/Foo.h>

    using namespace dolfin;

    //-----------------------------------------------------------------------------
    Foo::Foo() : // variable initialization here
    {
      ...
    }
    //-----------------------------------------------------------------------------
    Foo::~Foo()
    {
      // Do nothing
    }
    //-----------------------------------------------------------------------------

The horizontal lines above (including the slashes) should be exactly
79 characters wide.

Including header files and using forward declarations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Do not use ``#include <dolfin.h>`` or ``#include``
``<dolfin/dolfin_foo.h>`` inside the DOLFIN source tree. Only include
the portions of DOLFIN you are actually using.

Include as few header files as possible and use forward declarations
whenever possible (in header files). Put the ``#include`` in the
implementation file.  This reduces compilation time and minimizes the
risk of cyclic dependencies.

Explicit constructors
^^^^^^^^^^^^^^^^^^^^^

Make all one argument constructors (except copy constructors)
explicit:

.. code-block:: c++

    class Foo
    {
      explicit Foo(std::size_t i);
    };

Virtual functions
^^^^^^^^^^^^^^^^^

Always declare inherited virtual functions as virtual in the
subclasses.  This makes it easier to spot which functions are virtual.

.. code-block:: c++

    class Foo
    {
      virtual void foo();
      virtual void bar() = 0;
    };

    class Bar : public Foo
    {
      virtual void foo();
      virtual void bar();
    };

Use of libraries
----------------

Prefer C++ strings and streams over old C-style ``char*``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``std::string`` instead of ``const char*`` and use
``std::istream`` and ``std::ostream`` instead of ``FILE``. Avoid
``printf``, ``sprintf`` and other C functions.

There are some exceptions to this rule where we need to use old
C-style function calls. One such exception is handling of command-line
arguments (``char* argv[]``).

Prefer smart pointers over plain pointers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``std::shared_ptr`` and ``std::unique_ptr`` in favour of plain
pointers. Smart pointers reduce the likelihood of memory leaks and
make ownership clear. Use ``unique_ptr`` for a pointer that is not
shared and ``shared_ptr`` when multiple pointers point to the same
object.
