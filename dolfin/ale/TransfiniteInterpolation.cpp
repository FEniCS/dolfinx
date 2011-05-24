// Copyright (C) 2008-2009 Solveig Bruvoll and Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2008-05-02
// Last changed: 2009-10-08

#include <string.h>
#include <dolfin/common/constants.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Point.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include "TransfiniteInterpolation.h"
#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
void TransfiniteInterpolation::move(Mesh& mesh, const BoundaryMesh& new_boundary,
                                    InterpolationType method)
{
  // Only implemented in 2D and 3D
  if (mesh.topology().dim() < 2 || mesh.topology().dim() > 3 )
    error("Mesh interpolation only implemented in 2D and 3D so far.");

  // Get vertex and cell maps
  const MeshFunction<unsigned int>& vertex_map = new_boundary.vertex_map();
  const MeshFunction<unsigned int>& cell_map   = new_boundary.cell_map();

  // Extract old coordinates
  const uint gdim = mesh.geometry().dim();
  const uint num_vertices = mesh.num_vertices();
  std::vector<Point> new_x(num_vertices);
  std::vector<Point> ghat(new_boundary.num_vertices());

  // If hermite, create dgdn
  if (method == interpolation_hermite)
    hermite_function(ghat, gdim, new_boundary, mesh, vertex_map, cell_map);

  // Iterate over coordinates in mesh
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    mean_value(new_x[v->index()], gdim, new_boundary, mesh, vertex_map,
               *v, ghat, method);
  }

  // Update mesh coordinates
  MeshGeometry& geometry = mesh.geometry();
  for (uint i = 0; i < geometry.size(); i++)
    std::copy(new_x[i].coordinates(), new_x[i].coordinates() + gdim, geometry.x(i));
}
//-----------------------------------------------------------------------------
void TransfiniteInterpolation::mean_value(Point& new_x, uint dim,
                                          const BoundaryMesh& new_boundary,
                                          Mesh& mesh,
                                          const MeshFunction<uint>& vertex_map,
                                          const Vertex& vertex,
                                          const std::vector<Point>& ghat,
                                          InterpolationType method)
{
  // Check if the point is on the boundary (no need to compute new coordinate)
  for (VertexIterator v(new_boundary); !v.end(); ++v)
  {
    if (vertex_map[*v] == vertex.index())
    {
      std::copy(v->x(), v->x() + dim, new_x.coordinates());
      return;
    }
  }

  const uint num_boundary_vertices = new_boundary.num_vertices();
  std::vector<double> d(num_boundary_vertices);
  std::vector<Point> u(num_boundary_vertices);

  // Compute distance d and direction vector u from x to all p
  for (VertexIterator v(new_boundary); !v.end(); ++v)
  {
    // Old position of point x
    const Point x = vertex.point();

    // Old position of vertex v in boundary
    const Point p = mesh.geometry().point(vertex_map[*v]);

    // Distance from x to each point at the boundary
    d[v->index()] = dist(p, x, dim);

    // Compute direction vector for p - x
    for (uint i = 0; i < dim; i++)
      u[v->index()][i] = (p[i] - x[i]) / d[v->index()];
  }

  // FIXME: explain this
  const uint num_vertices = new_boundary.topology().dim() + 1;

  // Local arrays
  std::vector<double> w(num_vertices, 0.0);
  std::vector<double> d_cell(num_vertices, 0.0);
  std::vector<double> herm(num_vertices, 0.0);
  std::vector<Point> new_p(num_vertices);
  std::vector<Point> u_cell(num_vertices);
  std::vector<Point> ghat_cell(num_vertices);

  // Iterate over all cells in boundary
  double totalW = 0.0;
  for (CellIterator c(new_boundary); !c.end(); ++c)
  {
    // Get local data
    for (VertexIterator v(*c); !v.end(); ++v)
    {
      const uint ind = v.pos();
      new_p[ind] = v->point();
      u_cell[ind] = u[v->index()];
      d_cell[ind] = d[v->index()];
      if (method == interpolation_hermite)
        ghat_cell[ind] = ghat[v->index()];
    }

    // Compute weights w
    if (mesh.topology().dim() == 2)
      computeWeights2D(w, u_cell, d_cell, num_vertices);
    else
      computeWeights3D(w, u_cell, d_cell, dim, num_vertices);

    // Compute sum of weights
    for (uint i = 0; i < num_vertices; i++)
      totalW += w[i];

    // Compute new position
    for (uint j=0; j < dim; j++)
    {
      for (uint i=0; i < num_vertices; i++)
      {
        new_x[j] += w[i]*new_p[i][j];
        if (method == interpolation_hermite)
	        herm[j] += w[i]*ghat_cell[i][j];
      }
    }
  }
  // Scale by totalW
  if (method == interpolation_lagrange)
  {
    for (uint i = 0; i < dim; i++)
      new_x[i] /= totalW;
  }
  else
  {
    for (uint i = 0; i < dim; i++)
      new_x[i] = new_x[i]/totalW + herm[i]/(totalW*totalW);
  }
}
//-----------------------------------------------------------------------------
void TransfiniteInterpolation::computeWeights2D(std::vector<double>& w,
                                                const std::vector<Point>& u,
                                                const std::vector<double>& d,
                                                uint num_vertices)
{
  for (uint i = 0; i < num_vertices; i++)
    w[i] = sgn(det(u[0], u[1]))*std::tan(std::asin(u[0][0]*u[1][1] - u[0][1]*u[1][0])/2) / d[i];
}
//-----------------------------------------------------------------------------
void TransfiniteInterpolation::computeWeights3D(std::vector<double>& w,
                                                const std::vector<Point>& u,
                                                const std::vector<double>& d,
                                                uint dim, uint num_vertices)
{
  std::vector<double> ell(num_vertices);
  std::vector<double> theta(num_vertices);
  double h = 0.0;
  for (uint i = 0; i < num_vertices; i++)
  {
    const uint ind1 = next(i, num_vertices);
    const uint ind2 = previous(i, num_vertices);

    ell[i] = dist(u[ind1], u[ind2], dim);

    theta[i] = 2*std::asin(ell[i] / 2.0);
    h += theta[i]/2.0;
  }

  std::vector<double> c(num_vertices);
  std::vector<double> s(num_vertices);
  for (uint i = 0; i < num_vertices; i++)
  {
    const uint ind1 = next(i, num_vertices);
    const uint ind2 = previous(i, num_vertices);
    c[i] = (2*std::sin(h)*std::sin(h - theta[i])) / (std::sin(theta[ind1])*std::sin(theta[ind2])) - 1;
    const double sinus= 1.0 - c[i]*c[i];
    if (sinus < 0 || sqrt(sinus) < DOLFIN_EPS)
    {
      std::fill(w.begin(), w.end(), 0.0);
      return;
    }
    s[i] = sgn(det(u[0], u[1], u[2]))*std::sqrt(sinus);
  }

  for (uint i = 0; i < num_vertices; i++)
  {
    const uint ind1 = next(i, num_vertices);
    const uint ind2 = previous(i, num_vertices);
    w[i] = (theta[i]-c[ind1]*theta[ind2] - c[ind2]*theta[ind1])/(d[i]*sin(theta[ind1])*s[ind2]);
  }
}
//-----------------------------------------------------------------------------
void TransfiniteInterpolation::hermite_function(std::vector<Point>& ghat, uint dim,
                                                const BoundaryMesh& new_boundary,
		  	                                        Mesh& mesh,
                                                const MeshFunction<uint>& vertex_map,
			                                          const MeshFunction<uint>& cell_map)
{
  std::vector<Point> dfdn(new_boundary.num_vertices());
  normals(dfdn, dim, new_boundary, mesh, vertex_map, cell_map);

  double c = 0.0;
  if (dim == 2)
    c = 2.0;
  else
    c = DOLFIN_PI;

  // FIXME *All* comments should be in English
  //FAKTOREN c før dfdn, HVA VELGER VI DER?
  for (VertexIterator v(new_boundary); !v.end(); ++v)
  {
    integral(ghat[v->index()], dim, new_boundary, mesh, vertex_map, *v);
    for (uint i = 0; i < dim; i++)
      ghat[v->index()][i] = c*dfdn[v->index()][i] - ghat[v->index()][i];
  }
}
//-----------------------------------------------------------------------------
void TransfiniteInterpolation::normals(std::vector<Point>& dfdn, uint dim,
                                       const Mesh& new_boundary,
		                                   Mesh& mesh,
                                       const MeshFunction<uint>& vertex_map,
		                                   const MeshFunction<uint>& cell_map)
{
  for (VertexIterator v(new_boundary); !v.end(); ++v)
  {
    const uint ind = v.pos();
    std::fill(dfdn[ind].coordinates(), dfdn[ind].coordinates() + 3, 0.0);
    for (CellIterator c(new_boundary); !c.end(); ++c)
    {
      for (VertexIterator w(*c); !w.end(); ++w)
      {
        if(v->index() == w->index())
        {
          Facet mesh_facet(mesh, cell_map[*c]);
          Cell mesh_cell(mesh, mesh_facet.entities(mesh.topology().dim())[0]);
          const uint facet_index = mesh_cell.index(mesh_facet);
          const Point n = mesh_cell.normal(facet_index);

          for (uint j = 0; j < dim; j++)
            dfdn[ind][j] -= n[j];
          break;
        }
      }
    }
    const double len = length(dfdn[ind]);

    for (uint i = 0; i < dim; i++)
      dfdn[ind][i] /= len;
  }
}
//-----------------------------------------------------------------------------
void TransfiniteInterpolation::integral(Point& new_x, uint dim,
                                        const BoundaryMesh& new_boundary,
                                        Mesh& mesh,
                                        const MeshFunction<uint>& vertex_map,
                                        const Vertex& vertex)
{
  const uint num_boundary_vertices = new_boundary.num_vertices();
  std::vector<double> d(num_boundary_vertices);
  std::vector<Point> u(num_boundary_vertices);

  // Compute distance d and direction vector u from x to all p
  for (VertexIterator v(new_boundary);  !v.end(); ++v)
  {
    const uint ind = v->index();
    if(ind != vertex.index())
    {
      // Old position of point x
      const Point x = mesh.geometry().point(vertex_map[vertex]);

      // Old position of vertex v in boundary
      const Point p = mesh.geometry().point(vertex_map[*v]);

      // Distance from x to each point at the boundary
      d[ind] = dist(p, x, dim);

      // Compute direction vector for p-x
      for (uint i = 0; i < dim; i++)
        u[ind][i] = (p[i] - x[i]) / d[ind];
    }
    else
    {
      d[ind] = 0.0;
      for (uint i = 0; i < dim; i++)
        u[ind][i] = 0.0;
    }
  }

  // Local arrays
  const uint num_vertices = new_boundary.topology().dim() + 1;
  std::vector<double> w(num_vertices, 0.0);
  std::vector<Point> new_p(num_vertices);
  std::vector<double> d_cell(num_vertices, 0.0);
  std::vector<Point> u_cell(num_vertices);

  // Iterate over all cells in boundary
  for (CellIterator c(new_boundary); !c.end(); ++c)
  {
    // Get local data
    uint in_cell = 0;
    for (VertexIterator v(*c); !v.end(); ++v)
    {
      const uint ind = v.pos();
      if (v->index()==vertex.index())
        in_cell = 1;
      else
      {
        new_p[ind] = v->point();
        u_cell[ind] = u[v->index()];
        d_cell[ind] = d[v->index()];
      }
    }

    if (!in_cell)
    {
      // Compute weights w
      if (mesh.topology().dim() == 2)
        computeWeights2D(w, u_cell, d_cell, num_vertices);
      else
        computeWeights3D(w, u_cell, d_cell, dim, num_vertices);

      // Compute new position
      for (uint j = 0; j < dim; j++)
      {
        for (uint i=0; i < num_vertices; i++)
          new_x[j] += w[i]*(new_p[i][j]-vertex.x()[j]);
      }
    }
  }
}
//-----------------------------------------------------------------------------
double TransfiniteInterpolation::dist(const Point& x, const Point& y, uint dim)
{
  double s = 0.0;
  for (uint i = 0; i < dim; i++)
    s += (x[i] - y[i])*(x[i] - y[i]);
  return std::sqrt(s);
}
//-----------------------------------------------------------------------------
double TransfiniteInterpolation::length(const Point& x)
{
  return x.norm();
}
//-----------------------------------------------------------------------------
