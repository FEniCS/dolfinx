A simple eigenvalue solver (C++)
================================

We recommend that you are familiar with the demo for the Poisson
equation before looking at this demo.


Implementation
--------------

Running this demo requires the files: :download:`main.cpp`,
:download:`StiffnessMatrix.ufl` and :download:`CMakeLists.txt`.


Under construction

.. code-block:: cpp


   #include <dolfin.h>
   #include "StiffnessMatrix.h"

   using namespace dolfin;

   int main()
   {
     #ifdef HAS_SLEPC

     // Create mesh
     auto mesh = std::make_shared<Mesh>("../box_with_dent.xml.gz");

     // Build stiffness matrix
     auto A = std::make_shared<PETScMatrix>();
     auto V = std::make_shared<StiffnessMatrix::FunctionSpace>(mesh);
     StiffnessMatrix::BilinearForm a(V, V);
     assemble(*A, a);

     // Create eigensolver
     SLEPcEigenSolver esolver(A);

     // Compute all eigenvalues of A x = \lambda x
     esolver.solve();

     // Extract largest (first, n =0) eigenpair
     double r, c;
     PETScVector rx, cx;
     esolver.get_eigenpair(r, c, rx, cx, 0);

     cout << "Largest eigenvalue: " << r << endl;

     // Initialize function with eigenvector
     Function u(V);
     *u.vector() = rx;

     // Plot eigenfunction
     plot(u);
     interactive();

     #else

     dolfin::cout << "SLEPc must be installed to run this demo." << dolfin::endl;

     #endif

     return 0;
   }
