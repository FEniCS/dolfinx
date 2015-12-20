// Copyright (C) 2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2010
//
// First added:  2008-07-16
// Last changed: 2011-03-17

#include <dolfin/ale/ALE.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/constants.h>
#include "Mesh.h"
#include "BoundaryMesh.h"
#include "Vertex.h"
#include "Edge.h"
#include "Facet.h"
#include "Cell.h"
#include "MeshData.h"
#include "SubDomain.h"
#include "MeshSmoothing.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MeshSmoothing::smooth(Mesh& mesh, std::size_t num_iterations)
{
  log(PROGRESS, "Smoothing mesh");

  // Make sure we have cell-facet connectivity
  mesh.init(mesh.topology().dim(), mesh.topology().dim() - 1);

  // Make sure we have vertex-edge connectivity
  mesh.init(0, 1);

  // Make sure the mesh is ordered
  mesh.order();

  // Mark vertices on the boundary so we may skip them
  BoundaryMesh boundary(mesh, "exterior");
  const MeshFunction<std::size_t> vertex_map = boundary.entity_map(0);
  MeshFunction<bool> on_boundary(mesh, 0);
  on_boundary = false;
  if (boundary.num_vertices() > 0)
  {
    for (VertexIterator v(boundary); !v.end(); ++v)
      on_boundary[vertex_map[*v]] = true;
  }

  // Iterate over all vertices
  const std::size_t d = mesh.geometry().dim();
  std::vector<double> xx(d);
  for (std::size_t iteration = 0; iteration < num_iterations; iteration++)
  {
    for (VertexIterator v(mesh); !v.end(); ++v)
    {
      // Skip vertices on the boundary
      if (on_boundary[*v])
        continue;

      // Get coordinates of vertex
      const Point p = v->point();

      // Compute center of mass of neighboring vertices
      for (std::size_t i = 0; i < d; i++) xx[i] = 0.0;
      std::size_t num_neighbors = 0;
      for (EdgeIterator e(*v); !e.end(); ++e)
      {
        // Get the other vertex
        dolfin_assert(e->num_entities(0) == 2);
        std::size_t other_index = e->entities(0)[0];
        if (other_index == v->index())
          other_index = e->entities(0)[1];

        // Create the vertex
        Vertex vn(mesh, other_index);

        // Skip the vertex itself
        if (v->index() == vn.index())
          continue;
        num_neighbors += 1;

        // Compute center of mass
        const double* xn = vn.x();
        for (std::size_t i = 0; i < d; i++)
          xx[i] += xn[i];
      }
      for (std::size_t i = 0; i < d; i++)
        xx[i] /= static_cast<double>(num_neighbors);

      // Compute closest distance to boundary of star
      double rmin = 0.0;
      for (CellIterator c(*v); !c.end(); ++c)
      {
        // Get local number of vertex relative to facet
        const std::size_t local_vertex = c->index(*v);

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
      for (std::size_t i = 0; i < d; i++)
      {
        const double dx = xx[i] - p[i];
        r += dx*dx;
      }
      r = std::sqrt(r);
      if (r < DOLFIN_EPS)
        continue;
      rmin = std::min(0.5*rmin, r);

      std::vector<double> new_vertex(mesh.geometry().x(v->index()),
                                     mesh.geometry().x(v->index()) + d);
      for (std::size_t i = 0; i < d; i++)
        new_vertex[i] += rmin*(xx[i] - p[i])/r;
      mesh.geometry().set(v->index(), new_vertex.data());

    }
  }

  if (num_iterations > 1)
    log(PROGRESS, "Mesh smoothing repeated %d times.", num_iterations);
}
//-----------------------------------------------------------------------------
void MeshSmoothing::smooth_boundary(Mesh& mesh,
                                    std::size_t num_iterations,
                                    bool harmonic_smoothing)
{
  cout << "Smoothing boundary of mesh: " << mesh << endl;

  // Extract boundary of mesh
  BoundaryMesh boundary(mesh, "exterior");

  // Smooth boundary
  smooth(boundary, num_iterations);

  // Move interior vertices
  move_interior_vertices(mesh, boundary, harmonic_smoothing);
}
//-----------------------------------------------------------------------------
void MeshSmoothing::snap_boundary(Mesh& mesh,
                                  const SubDomain& sub_domain,
                                  bool harmonic_smoothing)
{
  cout << "Snapping boundary of mesh: " << mesh << endl;

  // Extract boundary of mesh
  BoundaryMesh boundary(mesh, "exterior");

  const std::size_t dim = mesh.geometry().dim();

  // Smooth boundary
  MeshGeometry& geometry = boundary.geometry();
  for (std::size_t i = 0; i < boundary.num_vertices(); i++)
  {
    Point p = geometry.point(i);
    Array<double> x(dim, p.coordinates());
    sub_domain.snap(x);
    geometry.set(i, p.coordinates());
  }

  // Move interior vertices
  move_interior_vertices(mesh, boundary, harmonic_smoothing);
}
//-----------------------------------------------------------------------------
void MeshSmoothing::move_interior_vertices(Mesh& mesh,
                                           BoundaryMesh& boundary,
                                           bool harmonic_smoothing)
{
  // Select smoothing of interior vertices
  if (harmonic_smoothing)
  {
    std::shared_ptr<Mesh> _mesh(&mesh, [](Mesh*){});
    ALE::move(_mesh, boundary);
  }
  else
  {
    // Use vertex map to update boundary coordinates of original mesh
    const MeshFunction<std::size_t>& vertex_map = boundary.entity_map(0);
    for (VertexIterator v(boundary); !v.end(); ++v)
      mesh.geometry().set(vertex_map[*v], v->x());
  }
}
//-----------------------------------------------------------------------------
