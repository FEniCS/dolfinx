
.. title:: Quick Programmer's Reference (C++)

##################################
Quick Programmer's Reference (C++)
##################################

The complete :ref:`DOLFIN Programmer's Reference <genindex>` documents
**all** the classes and functions in DOLFIN. As a **quicker**
reference, here are some of the most commonly used classes and
functions, organized by topic:

.. tabularcolumns:: |c|c|c|c|

.. list-table::
    :widths: 20, 20, 20, 20
    :header-rows: 1
    :class: center

    * - Finite elements
      - Meshes
      - Variational problems
      - Linear systems

    * - :cpp:class:`FunctionSpace`
      - :cpp:class:`Mesh`
      - :cpp:class:`LinearVariationalProblem`
      - :cpp:func:`assemble`

    * - :cpp:class:`Function`
      - :cpp:class:`MeshFunction`
      - :cpp:class:`LinearVariationalSolver`
      - :cpp:class:`Vector`

    * - :cpp:class:`Form`
      - :cpp:class:`SubDomain`
      - :cpp:class:`NonlinearVariationalProblem`
      - :cpp:class:`Matrix`

    * - :cpp:class:`Expression`
      -
      - :cpp:class:`NonlinearVariationalSolver`
      - :cpp:class:`LUSolver`

    * - :cpp:class:`DirichletBC`
      -
      - :cpp:func:`solve`
      - :cpp:class:`KrylovSolver`


