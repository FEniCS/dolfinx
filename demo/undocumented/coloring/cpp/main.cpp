// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-16
// Last changed:
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

  // Create a vertex-based coloring object and color cells
  CellColoring coloring_vertex(mesh, "vertex");
  CellFunction<dolfin::uint> colors_vertex = coloring_vertex.compute_local_cell_coloring();
  plot(colors_vertex, "Vertex-based cell coloring");

  // Create an edge-based coloring object and color cells
  CellColoring coloring_edge(mesh, "edge");
  CellFunction<dolfin::uint> colors_edge = coloring_edge.compute_local_cell_coloring();
  plot(colors_edge, "Edge-based cell coloring");

  // Create a facet-based coloring object and color cells
  CellColoring coloring_facet(mesh, "facet");
  CellFunction<dolfin::uint> colors_facet = coloring_facet.compute_local_cell_coloring();
  plot(colors_facet, "Facet-based cell coloring");

  #else

  cout << "Trilinos must be installed to run this demo." << endl;

  #endif


  return 0;
}
