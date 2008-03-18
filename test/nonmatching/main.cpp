// Licensed under the GNU LGPL Version 2.1.

#include <dolfin.h>
#include "Simple.h"
  
using namespace dolfin;

int main(int argc, char* argv[])
{
  Mesh meshA;
  MeshEditor editor;
  editor.open(meshA, "triangle", 2, 2);
  editor.initVertices(4);
  editor.addVertex(0, 0.0, 0.0);
  editor.addVertex(1, 1.0, 0.0);
  editor.addVertex(2, 1.0, 1.0);
  editor.addVertex(3, 0.0, 1.0);
  editor.initCells(2);
  editor.addCell(0, 0, 1, 3);
  editor.addCell(1, 1, 2, 3);
  editor.close();

  Function fA;
  Vector xA;

  Form* M = new SimpleFunctional(fA);

  fA.init(meshA, xA, *M, 0);

  int NA = meshA.numVertices();
  int d = meshA.topology().dim();

  // Generate some values for fA
  real* arr = new real[NA];
  for (VertexIterator v(meshA); !v.end(); ++v)
  {
    int id = v->index();
    Point p = v->point();

    arr[0 * NA + id] = 2.0 * (p[1] - 0.5) * pow(p[0] - 0.0, 2); 
  }

  fA.vector().set(arr);
  delete arr;

  File file_fA("fA.pvd");
  file_fA << fA;

  UnitSquare meshB(11, 11);

  Function fB;

  ufc::finite_element* element = M->form().create_finite_element(0);

  projectL2NonMatching(meshB, fA, fB, *element);

  File file_fB("fB.pvd");
  file_fB << fB;

  return 0;
}
