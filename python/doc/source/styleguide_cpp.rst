.. _developers_styleguide_cpp:

C++ style guide
===============

Formatting
----------

`clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_ is used
to format files. A `.clang-format` file is included in the root
directory. Editors can be configured to apply the style using
`clang-format`.



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

Comments
^^^^^^^^

Capitalize the first letter of a comment and don't use punctuation
(unless the comment runs over several sentences). Here's an example:

.. code-block:: c++

    // Check if connectivity has already been computed
    if (!connectivity.empty())
      return;

    // Invalidate ordering
    mesh._ordered = false;

    // Compute entities if they don't exist
    if (topology.size(d0) == 0)
      compute_entities(mesh, d0);
    if (topology.size(d1) == 0)
      compute_entities(mesh, d1);

    // Check if connectivity still needs to be computed
    if (!connectivity.empty())
      return;

    ...

Always use ``//`` for comments and ``///`` for documentation. Never
use ``/* foo */``, not even for comments that runs over multiple
lines.


Header file layout
^^^^^^^^^^^^^^^^^^

Header files should follow the below template:

.. code-block:: c++

    // Copyright (C) 2018 Foo Bar
    //
    // This file is part of DOLFINx (https://www.fenicsproject.org)
    //
    // SPDX-License-Identifier:    LGPL-3.0-or-later

    #pragma once

    namespace dolfinx
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

Implementation file layout
^^^^^^^^^^^^^^^^^^^^^^^^^^

Implementation files should follow the below template:

.. code-block:: c++

    // Copyright (C) 2018 Foo Bar
    //
    // This file is part of DOLFINx (https://www.fenicsproject.org)
    //
    // SPDX-License-Identifier:    LGPL-3.0-or-later

    #include <dolfinx/Foo.h>

    using namespace dolfinx;

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


Including header files and using forward declarations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Do not use ``#include <dolfinx.h>`` or ``#include``
``<dolfinx/dolfin_foo.h>`` inside the DOLFINx source tree. Only include
the portions of DOLFINx you are actually using.

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

Avoid plain pointers
^^^^^^^^^^^^^^^^^^^^

Use C++11 smart pointer and avoid plain pointers.
