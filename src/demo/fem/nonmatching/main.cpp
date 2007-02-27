// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-07
// Last changed: 2006-09-05
//

#include <dolfin.h>
#include "Projection.h"
  
using namespace dolfin;

int main()
{
  // Define function fA

  UnitSquare meshA(16, 16);

  FiniteElement* K = new P1tri();
  int NA = FEM::size(meshA, *K);

  Vector xA(NA);

  for(VertexIterator vi(meshA); !vi.end(); ++vi)
  {
    int id = vi->index();
    Vertex v(meshA, id);
    Point p = v.point();

    xA(id) = 3.0 * sin(2.0 * p.x()) + pow(p.y(), 3);
  }

  Function fA(xA, meshA, *K);

  // Save function to file
  File fileA("fA.pvd");
  fileA << fA;



  // Define meshB

  Mesh meshB(meshA);
  MeshFunction<bool> marked_cells;
  marked_cells.init(meshB, 2);

  for(CellIterator ci(meshB); !ci.end(); ++ci)
  {
    int id = ci->index();
    Cell c(meshB, id);
    Point p = c.midpoint();

    if((p.x() > 0.3 && p.x() < 0.7) && (p.y() > 0.4 && p.y() < 0.6))
    {
      marked_cells.set(c, true);
    }
  }

  meshB.refine(marked_cells);

  // Define and compute projection

  Function fN(fA, meshB);

  Projection::BilinearForm a;
  Projection::LinearForm L(fN);

  Matrix A;
  Vector b;
  Vector xB;

  FEM::assemble(a, A, meshB);
  FEM::assemble(L, b, meshB);

  KrylovSolver solver;

  solver.solve(A, xB, b);

  // Define function fB
  Function fB(xB, meshB, *K);

  // Save solution to file
  File fileB("fB.pvd");
  fileB << fB;
  
  return 0;
}
