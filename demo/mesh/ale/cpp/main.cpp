// Copyright (C) 2008 Solveig Bruvoll and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-02
// Last changed: 2008-05-05
//
// This demo demonstrates how to move the vertex coordinates
// of a boundary mesh and then updating the interior vertex
// coordinates of the original mesh by suitably interpolating
// the vertex coordinates (useful for implementation of ALE
// methods).

#include <dolfin.h>

using namespace dolfin;

int main()
{
  // Create mesh
  UnitSquare mesh(20, 20);
  //UnitCube mesh(3, 3, 3);
  plot(mesh);

  // Create boundary mesh
  MeshFunction<unsigned int> vertex_map;
  MeshFunction<unsigned int> cell_map;
  BoundaryMesh boundary(mesh, vertex_map, cell_map);

  // Move vertices in boundary
  for (VertexIterator v(boundary); !v.end(); ++v)
  {
    real* x = v->x();
    x[0] *= 3.0;
    x[1] += 0.1*sin(5.0*x[0]);
  }
  plot(boundary);
 
  // Move mesh
  //mesh.move(boundary, vertex_map, cell_map, lagrange);
  mesh.move(boundary, vertex_map, cell_map, hermite);
  plot(mesh);

  return 0;
}
