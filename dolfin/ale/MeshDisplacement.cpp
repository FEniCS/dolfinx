// Copyright (C) 2013 Jan Blechta
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
// First added:  2013-03-05
// Last changed: 2013-03-05

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Mesh.h>
#include "Poisson1D.h"
#include "Poisson2D.h"
#include "Poisson3D.h"
#include "MeshDisplacement.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshDisplacement::MeshDisplacement(std::shared_ptr<const Mesh> mesh)
  : Expression(mesh->geometry().dim()), _dim(mesh->geometry().dim())
{
  dolfin_assert(mesh);
  const std::size_t D = mesh->topology().dim();

  // Choose form and function space
  std::shared_ptr<FunctionSpace> V;
  switch (D)
  {
  case 1:
    V.reset(new Poisson1D::FunctionSpace(mesh));
    break;
  case 2:
    V.reset(new Poisson2D::FunctionSpace(mesh));
    break;
  case 3:
    V.reset(new Poisson3D::FunctionSpace(mesh));
    break;
  default:
    dolfin_error("MeshDisplacement.cpp",
                 "create instance if MeshDisplacement",
                 "Illegal mesh dimension (%d)", D);
  }

  // Store displacement functions
  _displacements = std::vector<Function>(_dim, Function(V));
}
//-----------------------------------------------------------------------------
MeshDisplacement::MeshDisplacement(const MeshDisplacement& mesh_displacement)
  : Expression(mesh_displacement._dim), _dim(mesh_displacement._dim),
    _displacements(mesh_displacement._displacements)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshDisplacement::~MeshDisplacement()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Function& MeshDisplacement::operator[] (const std::size_t i)
{
  dolfin_assert(i < _dim);
  return _displacements[i];
}
//-----------------------------------------------------------------------------
const Function& MeshDisplacement::operator[] (const std::size_t i) const
{
  dolfin_assert(i < _dim);
  return _displacements[i];
}
//-----------------------------------------------------------------------------
void MeshDisplacement::eval(Array<double>& values, const Array<double>& x,
                            const ufc::cell& cell) const
{
  for (std::size_t i = 0; i < _dim; i++)
  {
    Array<double> _values(1, &values[i]);
    _displacements[i].eval(_values, x, cell);
  }
}
//-----------------------------------------------------------------------------
void MeshDisplacement::compute_vertex_values(std::vector<double>& vertex_values,
                                             const Mesh& mesh) const
{
  // TODO: implement also computation on current mesh by
  //       _displacements[i].vector()->get_local(block, num_vertices,
  //       all_dofs) which would be merely moving code from
  //       HarmonicSmoothing.cpp here.  This would save some
  //       computation performed by compute_vertex_values()

  vertex_values.clear();

  std::size_t num_vertices = mesh.num_vertices();
  vertex_values.reserve(_dim*num_vertices);

  for (std::size_t i = 0; i < _dim; i++)
  {
    std::vector<double> _vertex_values;
    _displacements[i].compute_vertex_values(_vertex_values, mesh);
    vertex_values.insert(vertex_values.end(),
                         _vertex_values.begin(),
                         _vertex_values.end());
  }
}
//-----------------------------------------------------------------------------
