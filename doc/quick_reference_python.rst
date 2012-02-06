
.. title:: Quick Programmer's Reference (Python)

#####################################
Quick Programmer's Reference (Python)
#####################################

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

    * - :py:class:`FunctionSpace <dolfin.functions.functionspace.FunctionSpace>`
      - :py:class:`Mesh <dolfin.cpp.Mesh>`
      - :py:func:`solve <dolfin.fem.solving.solve>`
      - :py:func:`assemble <dolfin.fem.assembling.assemble>`

    * - :py:class:`TestFunction <dolfin.functions.function.TestFunction>`/:py:class:`TrialFunction <dolfin.functions.function.TrialFunction>`
      - :py:class:`MeshFunction <dolfin.cpp.MeshFunction>`
      - :py:class:`LinearVariationalProblem <dolfin.fem.solving.LinearVariationalProblem>`
      - :py:class:`Vector <dolfin.cpp.Vector>`

    * - :py:class:`Function <dolfin.functions.function.Function>`
      - :py:class:`SubDomain <dolfin.cpp.SubDomain>`
      - :py:class:`LinearVariationalSolver <dolfin.fem.solving.LinearVariationalSolver>`
      - :py:class:`Matrix <dolfin.cpp.Matrix>`

    * - :py:class:`Expression <dolfin.functions.expression.Expression>`
      -
      - :py:class:`NonlinearVariationalProblem <dolfin.fem.solving.NonlinearVariationalProblem>`
      - :py:class:`LUSolver <dolfin.cpp.LUSolver>`

    * - :py:class:`DirichletBC <dolfin.fem.bcs.DirichletBC>`
      -
      - :py:class:`NonlinearVariationalSolver <dolfin.fem.solving.NonlinearVariationalSolver>`
      - :py:class:`KrylovSolver <dolfin.cpp.KrylovSolver>`
