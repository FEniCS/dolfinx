.. _developers_styleguide_python:

Python style guide
==================

Formatting
----------

`ruff <https://docs.astral.sh/ruff/>`_ is used to format and lint the
Python interface, the test suite and the demos. It is configured in
``python/pyproject.toml``. Run ``ruff format`` and ``ruff check`` before
submitting a pull request; the continuous integration runs ``ruff format
--check`` and ``ruff check``.


.. _developers_python_wrappers:

Wrapping C++ objects
--------------------

The nanobind bindings in ``python/dolfinx/wrappers`` expose the C++
library as ``dolfinx.cpp``. The user-facing interface is the pure-Python
layer built on top of it, in which each class holds the bound C++ object
as ``self._cpp_object``. Users and developers should not use
``dolfinx.cpp`` directly.

A C++ accessor that returns another C++ object therefore has to be
turned into the matching Python wrapper somewhere. Do this consistently,
following the rules below, so that wrapper identity is predictable:
``V.dofmap is V.dofmap`` holds, and the wrapper a user holds stays in
step with the C++ object the library would hand back.

Python objects supplied by the caller
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a Python object is passed to an initialiser, store it and return it
unchanged. Re-wrapping the corresponding C++ object would hand back a
different Python object than the caller supplied, discarding any state
that lives only in the Python layer, such as a UFL element:

.. code-block:: python

    @property
    def function_space(self) -> FunctionSpace:
        """Function space on which the boundary condition is defined."""
        return self._V

Accessors that take no arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :func:`functools.cached_property`. The wrapper is built on first
access and the same object is returned thereafter, so no wrapper is
built for a property that is never read and none is built twice:

.. code-block:: python

    @cached_property
    def dofmap(self) -> DofMap:
        """Degree-of-freedom map associated with the function space."""
        return DofMap(self._cpp_object.dofmap)

Accessors that take arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`functools.cached_property` does not apply. Use
``dolfinx._wrapper.cached``, which memoises on the C++ object held in a
per-instance ``dict``. Keying on the C++ object rather than on the
arguments means that a wrapper is rebuilt if the C++ layer replaces the
object it returns, as ``SparsityPattern::finalize`` does for the column
index map:

.. code-block:: python

    def index_map(self, dim: int) -> IndexMap:
        """Index map for the parallel distribution of the mesh entities."""
        return _cached(self._wrappers, IndexMap, self._cpp_object.index_map(dim))

Accessors that build a new C++ object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Methods such as ``FunctionSpace.sub``, ``MatrixCSR.transpose`` and
``CoordinateElement.create_dof_layout`` build a new C++ object on each
call. Wrap the result directly; there is nothing to cache, and caching
would keep every intermediate alive.

Equality and hashing
^^^^^^^^^^^^^^^^^^^^

Do not rely on wrapper identity in library code or tests. Wrappers that
users compare define ``__eq__``, and ``__hash__`` alongside it where the
wrapped type admits a hash consistent with that equality. Note that
defining ``__eq__`` without ``__hash__`` sets ``__hash__`` to ``None``
and makes the class unhashable, which for a class with a hashable base,
such as :class:`dolfinx.fem.FunctionSpace`, silently removes behaviour
the base provided.
