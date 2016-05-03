// Copyright (C) 2013 Kent-Andre Mardal, Mikael Mortensen, Johan Hake
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
// First added:  2013-04-02

#include <dolfin/common/Array.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/mesh/Cell.h>
#include "Probe.h"

using namespace dolfin;

//----------------------------------------------------------------------------
Probe::Probe(const Array<double>& x, const FunctionSpace& V)
  : _element(V.element())
{
  const Mesh& mesh = *V.mesh();
  const std::size_t gdim = mesh.geometry().dim();

  // Store position of probe
  for (std::size_t i = 0; i < 3; i++)
    _x[i] = (i < gdim ? x[i] : 0.0);

  // Compute in tensor (one for scalar function, . . .)
  value_size_loc = 1;
  for (uint i = 0; i < _element->value_rank(); i++)
    value_size_loc *= _element->value_dimension(i);

  _probes.resize(value_size_loc);

  // Find the cell that contains probe
  const Point point(gdim, x.data());
  std::shared_ptr<BoundingBoxTree> tree = mesh.bounding_box_tree();
  const unsigned int id = tree->compute_first_entity_collision(point);

  // If the cell is on this process, then create an instance
  // of the Probe class. Otherwise raise a dolfin_error.
  if (id != std::numeric_limits<unsigned int>::max())
  {
    // Create cell that contains point
    dolfin_cell.reset(new Cell(mesh, id));
    dolfin_cell->get_cell_data(ufc_cell);
    dolfin_cell->get_coordinate_dofs(_coordinate_dofs);

    // Create work vector for basis
    std::vector<double> basis(value_size_loc);

    _coefficients.resize(_element->space_dimension());

    // Create work vector for basis
    _basis_matrix.resize(value_size_loc);
    for (std::size_t i = 0; i < value_size_loc; ++i)
      _basis_matrix[i].resize(_element->space_dimension());

    for (std::size_t i = 0; i < _element->space_dimension(); ++i)
    {
      _element->evaluate_basis(i, basis.data(), x.data(),
                               _coordinate_dofs.data(),
                               dolfin_cell->orientation());
      for (std::size_t j = 0; j < value_size_loc; ++j)
        _basis_matrix[j][i] = basis[j];
    }
  }
  else
    dolfin_error("Probe.cpp","set probe","Probe is not found on processor");
}
//----------------------------------------------------------------------------
std::size_t Probe::value_size() const
{
  return value_size_loc;
}
//----------------------------------------------------------------------------
std::size_t Probe::number_of_evaluations() const
{
  return _probes[0].size();
}
//----------------------------------------------------------------------------
void Probe::eval(const Function& u)
{
  // Restrict function to cell
  u.restrict(&_coefficients[0], *_element, *dolfin_cell,
             _coordinate_dofs.data(), ufc_cell);

  // Make room for one more evaluation
  for (std::size_t j = 0; j < value_size_loc; j++)
    _probes[j].push_back(0.0);

  const std::size_t n = _probes[0].size() - 1;

  // Compute linear combination
  for (std::size_t i = 0; i < _element->space_dimension(); i++)
  {
    for (std::size_t j = 0; j < value_size_loc; j++)
      _probes[j][n] += _coefficients[i]*_basis_matrix[j][i];
  }
}
//----------------------------------------------------------------------------
void Probe::erase(std::size_t i)
{
  for (std::size_t j = 0; j < value_size_loc; j++)
    _probes[j].erase(_probes[j].begin()+i);
}
//----------------------------------------------------------------------------
void Probe::clear()
{
  for (std::size_t j = 0; j < value_size_loc; j++)
    _probes[j].clear();
}
//----------------------------------------------------------------------------
std::vector<double> Probe::get_probe(std::size_t i) const
{
  return _probes[i];
}
//----------------------------------------------------------------------------
std::vector<double> Probe::coordinates() const
{
  std::vector<double> x(3);
  x.assign(_x, _x + 3);
  return x;
}
//----------------------------------------------------------------------------
