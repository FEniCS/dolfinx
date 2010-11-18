// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010.
//
// First added:  2010-11-16
// Last changed: 2010-11-17
//
// This demo colors the cells of a mesh such that cells with the same
// color are not neighbors. 'Neighbors' can be in the sense of shared
// vertices, edges or facets.

#include <dolfin.h>

using namespace dolfin;

int main()
{
   #ifdef HAS_TRILINOS

  // Create mesh
  UnitCube mesh(24, 24, 24);

  // Compute vertex-based coloring
  const MeshFunction<dolfin::uint>& colors_vertex = mesh.color("vertex");
  plot(colors_vertex, "Vertex-based cell coloring");

  // Compute edge-based coloring
  const MeshFunction<dolfin::uint>& colors_edge = mesh.color("edge");
  plot(colors_edge, "Edge-based cell coloring");

  // Compute facet-based coloring
  const MeshFunction<dolfin::uint>& colors_facet = mesh.color("facet");
  plot(colors_facet, "Facet-based cell coloring");

  #else

  cout << "Trilinos (with Zoltan enabled) must be installed to run this demo." << endl;

  #endif


  return 0;
}
