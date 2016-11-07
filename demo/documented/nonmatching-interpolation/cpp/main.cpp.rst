Interpolation from a non-matching mesh
======================================

This example demonstrates how to interpolate functions between
finite element spaces on non-matching meshes.

.. note::

   Interpolation on non-matching meshes is not presently support in
   parallel. See
   https://bitbucket.org/fenics-project/dolfin/issues/162.


Implementation
--------------

The implementation is split in three files: two UFL form files
containing the definition of finite elements, and a C++ file
containing the runtime code.

Running this demo requires the files: :download:`main.cpp`,
:download:`P1.ufl`, :download:`P3.ufl` and :download:`CMakeLists.txt`.


UFL form file
^^^^^^^^^^^^^

The UFL files are implemented in :download:`P1.ufl` and
:download:`P1.ufl`, and the explanations of the UFL files can be found
at :doc:`here (P1) <P1.ufl>` and :doc:`here (P3) <P3.ufl>`.

At the top we include the DOLFIN header file and the generated header
files "P1.h" and "P2.h". For convenience we also include the DOLFIN
namespace.

.. code-block:: cpp

   #include <dolfin.h>
   #include "P1.h"
   #include "P3.h"

   using namespace dolfin;

We then define an ``Expression``:

.. code-block:: cpp

   class MyExpression : public Expression
   {
   public:

     void eval(Array<double>& values, const Array<double>& x) const
     {
       values[0] = sin(10.0*x[0])*sin(10.0*x[1]);
     }

   };

Next, the ``main`` function is started and we create two unit square
meshes with a differing number of vertices in each direction:

.. code-block:: cpp

   int main()
   {
     // Create meshes
     auto mesh0 = std::make_shared<UnitSquareMesh>(16, 16);
     auto mesh1 = std::make_shared<UnitSquareMesh>(64, 64);

We create a linear Lagrange finite element space on the coarser mesh,
and a cubic Lagrange space on the finer mesh:

.. code-block:: cpp

     // Create function spaces
     auto P1 = std::make_shared<P1::FunctionSpace>(mesh0);
     auto P3 = std::make_shared<P3::FunctionSpace>(mesh1);

One each space we create a finite element function:

.. code-block:: cpp

     // Create functions
     Function v1(P1);
     Function v3(P3);

We create an instantiation of ``MyExpression``, and interpolate it
into ``P3``:

.. code-block:: cpp

     // Interpolate expression into P3
     MyExpression e;
     v3.interpolate(e);

Now, we interpolate ``v3`` into the linear finite element space on a
coarser grid:

.. code-block:: cpp

     v1.interpolate(v3);

Finally, we can visualise the function on the two meshes:

.. code-block:: cpp

     plot(v3);
     plot(v1);
     interactive();

     return 0;
   }
