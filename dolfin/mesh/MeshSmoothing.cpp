// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-16
// Last changed: 2008-07-16

#include <dolfin/common/constants.h>
#include "Mesh.h"
#include "BoundaryMesh.h"
#include "Vertex.h"
#include "Cell.h"
#include "Facet.h"
#include "MeshData.h"
#include "MeshSmoothing.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshSmoothing::smooth(Mesh& mesh)
{
  // Make sure we have cell-facet connectivity
  mesh.init(mesh.topology().dim(), mesh.topology().dim() - 1);
  
  // Make sure the mesh is ordered
  mesh.order();

  // Mark vertices on the boundary so we may skip them
  BoundaryMesh boundary(mesh);
  MeshFunction<uint>* vertex_map = boundary.data().meshFunction("vertex map");
  dolfin_assert(vertex_map);
  MeshFunction<bool> on_boundary(mesh, 0);
  on_boundary = false;
  for (VertexIterator v(boundary); !v.end(); ++v)
    on_boundary.set((*vertex_map)(*v), true);

  // Iterate over all vertices
  const uint d = mesh.geometry().dim();
  Array<double> xx(d);
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    // Skip vertices on the boundary
    if (on_boundary(*v))
      continue;
    
    // Get coordinates of vertex
    double* x = v->x();
    const Point p = v->point();
    
    // Compute center of mass of neighboring vertices
    for (uint i = 0; i < d; i++) xx[i] = 0.0;
    uint num_neighbors = 0;
    for (VertexIterator vn(*v); !vn.end(); ++vn)
    {
      // Skip the vertex itself
      if (v->index() == vn->index())
        continue;
      num_neighbors += 1;

      // Compute center of mass
      const double* xn = vn->x();
      for (uint i = 0; i < d; i++)
        xx[i] += xn[i];
    }
    for (uint i = 0; i < d; i++)
      xx[i] /= static_cast<double>(num_neighbors);

    // Compute closest distance to boundary of star
    double rmin = 0.0;
    for (CellIterator c(*v); !c.end(); ++c)
    {
      // Get local number of vertex relative to facet
      const uint local_vertex = c->index(*v);

      // Get normal of corresponding facet
      Point n = c->normal(local_vertex);
      
      // Get first vertex in facet
      Facet f(mesh, c->entities(mesh.topology().dim() - 1)[local_vertex]);
      VertexIterator fv(f);

      // Compute length of projection of v - fv onto normal
      const double r = std::abs(n.dot(p - fv->point()));
      if (rmin == 0.0)
        rmin = r;
      else
        rmin = std::min(rmin, r);
    }

    // Move vertex at most a distance rmin / 2
    double r = 0.0;
    for (uint i = 0; i < d; i++)
    {
      const double dx = xx[i] - x[i];
      r += dx*dx;
    }
    r = std::sqrt(r);
    if (r < DOLFIN_EPS)
      continue;
    rmin = std::min(0.5*rmin, r);
    for (uint i = 0; i < d; i++)
      x[i] += rmin*(xx[i] - x[i])/r;
  }
}
//-----------------------------------------------------------------------------
