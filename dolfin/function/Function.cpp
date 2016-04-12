// Copyright (C) 2003-2012 Anders Logg
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
// Modified by Garth N. Wells 2005-2010
// Modified by Martin Sandve Alnes 2008-2014
// Modified by Andre Massing 2009

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include <dolfin/adaptivity/Extrapolation.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/utils.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/io/File.h>
#include <dolfin/io/XMLFile.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/DefaultFactory.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include "Expression.h"
#include "FunctionSpace.h"
#include "Function.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V)
  : Hierarchical<Function>(*this), _function_space(V),
  _allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Check that we don't have a subspace
  if (!V->component().empty())
  {
    dolfin_error("Function.cpp",
                 "create function",
                 "Cannot be created from subspace. Consider collapsing the function space");
  }

  // Initialize vector
  init_vector();
}
//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V,
                   std::shared_ptr<GenericVector> x)
  : Hierarchical<Function>(*this), _function_space(V), _vector(x),
  _allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // We do not check for a subspace since this constructor is used for
  // creating subfunctions

  // Assertion uses '<=' to deal with sub-functions
  dolfin_assert(V->dofmap());
  dolfin_assert(V->dofmap()->global_dimension() <= x->size());
}
//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V,
                   std::string filename) : Hierarchical<Function>(*this),
  _function_space(V),
  _allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Check that we don't have a subspace
  if (!V->component().empty())
  {
    dolfin_error("Function.cpp",
                 "create function",
                 "Cannot be created from subspace. Consider collapsing the function space");
  }

  // Initialize vector
  init_vector();

  // Check size of vector
  if (_vector->size() != _function_space->dim())
  {
    dolfin_error("Function.cpp",
                 "read function from file",
                 "The number of degrees of freedom (%d) does not match dimension of function space (%d)",
                 _vector->size(), _function_space->dim());
  }

  // Read function data from file
  MPI_Comm comm = _function_space->mesh()->mpi_comm();
  File file(comm, filename);
  file >> *this;
}
//-----------------------------------------------------------------------------
Function::Function(const Function& v) : Hierarchical<Function>(*this),
  _allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Assign data
  *this = v;
}
//-----------------------------------------------------------------------------
Function::Function(const Function& v, std::size_t i)
  : Hierarchical<Function>(*this),
  _allow_extrapolation(dolfin::parameters["allow_extrapolation"])
{
  // Copy function space pointer
  this->_function_space = v[i]._function_space;

  // Copy vector pointer
  this->_vector = v[i]._vector;
}
//-----------------------------------------------------------------------------
Function::~Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const Function& Function::operator= (const Function& v)
{
  dolfin_assert(v._vector);

  // Make a copy of all the data, or if v is a sub-function, then we
  // collapse the dof map and copy only the relevant entries from the
  // vector of v.
  if (v._vector->size() == v._function_space->dim())
  {
    // Copy function space
    _function_space = v._function_space;

    // Copy vector
    _vector = v._vector->copy();

    // Clear subfunction cache
    _sub_functions.clear();
  }
  else
  {
    // Create new collapsed FunctionSpace
    std::unordered_map<std::size_t, std::size_t> collapsed_map;
    _function_space = v._function_space->collapse(collapsed_map);

    // Get row indices of original and new vectors
    std::unordered_map<std::size_t, std::size_t>::const_iterator entry;
    std::vector<dolfin::la_index> new_rows(collapsed_map.size());
    std::vector<dolfin::la_index> old_rows(collapsed_map.size());
    std::size_t i = 0;
    for (entry = collapsed_map.begin(); entry != collapsed_map.end(); ++entry)
    {
      new_rows[i]   = entry->first;
      old_rows[i++] = entry->second;
    }
    MPI::barrier(MPI_COMM_WORLD);

    // Gather values into a vector
    dolfin_assert(v.vector());
    std::vector<double> gathered_values(collapsed_map.size());
    v.vector()->get_local(gathered_values.data(), gathered_values.size(),
                          old_rows.data());

    // Initial new vector (global)
    init_vector();
    dolfin_assert(_function_space->dofmap());
    dolfin_assert(_vector->size()
                  == _function_space->dofmap()->global_dimension());

    // FIXME (local): Check this for local or global
    // Set values in vector
    this->_vector->set_local(gathered_values.data(), collapsed_map.size(),
                             new_rows.data());
    this->_vector->apply("insert");
  }

  // Call assignment operator for base class
  Hierarchical<Function>::operator=(v);

  return *this;
}
//-----------------------------------------------------------------------------
const Function& Function::operator= (const Expression& v)
{
  interpolate(v);
  return *this;
}
//-----------------------------------------------------------------------------
Function& Function::operator[] (std::size_t i) const
{
  // Check if sub-Function is in the cache, otherwise create and add
  // to cache
  auto sub_function = _sub_functions.find(i);
  if (sub_function != _sub_functions.end())
    return *(sub_function->second);
  else
  {
    // Extract function subspace
    std::vector<std::size_t> component(1, i);
    std::shared_ptr<const FunctionSpace>
      sub_space(_function_space->extract_sub_space(component));

    // Insert sub-Function into map and return reference
    _sub_functions.insert(i, new Function(sub_space, _vector));
    return *(_sub_functions.find(i)->second);
  }
}
//-----------------------------------------------------------------------------
FunctionAXPY Function::operator+(const Function& other) const
{
  return FunctionAXPY(*this, other, FunctionAXPY::Direction::ADD_ADD);
}
//-----------------------------------------------------------------------------
FunctionAXPY Function::operator+(const FunctionAXPY& axpy) const
{
  return FunctionAXPY(axpy, *this, FunctionAXPY::Direction::ADD_ADD);
}
//-----------------------------------------------------------------------------
FunctionAXPY Function::operator-(const Function& other) const
{
  return FunctionAXPY(*this, other, FunctionAXPY::Direction::ADD_SUB);
}
//-----------------------------------------------------------------------------
FunctionAXPY Function::operator-(const FunctionAXPY& axpy) const
{
  return FunctionAXPY(axpy, *this, FunctionAXPY::Direction::SUB_ADD);
}
//-----------------------------------------------------------------------------
FunctionAXPY Function::operator*(double scalar) const
{
  return FunctionAXPY(*this, scalar);
}
//-----------------------------------------------------------------------------
FunctionAXPY Function::operator/(double scalar) const
{
  return FunctionAXPY(*this, 1.0/scalar);
}
//-----------------------------------------------------------------------------
void Function::operator=(const FunctionAXPY& axpy)
{
  if (axpy.pairs().size() == 0)
  {
    dolfin_error("Function.cpp",
                 "assign function",
                 "FunctionAXPY is empty.");
  }

  // Make an initial assign and scale
  *this = *(axpy.pairs()[0].second);
  if (axpy.pairs()[0].first != 1.0)
    *_vector *= axpy.pairs()[0].first;

  // Start from item 2 and axpy
  std::vector<std::pair<double, const Function*>>::const_iterator it;
  for (it = axpy.pairs().begin()+1; it != axpy.pairs().end(); it++)
    _vector->axpy(it->first, *(it->second->vector()));
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> Function::vector()
{
  dolfin_assert(_vector);
  dolfin_assert(_function_space->dofmap());

  // Check that this is not a sub function.
  if (_vector->size() != _function_space->dofmap()->global_dimension())
  {
    dolfin_error("Function.cpp",
                 "access vector of degrees of freedom",
                 "Cannot access a non-const vector from a subfunction");
  }

  return _vector;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const GenericVector> Function::vector() const
{
  dolfin_assert(_vector);
  return _vector;
}
//-----------------------------------------------------------------------------
bool Function::in(const FunctionSpace& V) const
{
  dolfin_assert(_function_space);
  return *_function_space == V;
}
//-----------------------------------------------------------------------------
std::size_t Function::geometric_dimension() const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());
  return _function_space->mesh()->geometry().dim();
}
//-----------------------------------------------------------------------------
void Function::eval(Array<double>& values, const Array<double>& x) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  // Find the cell that contains x
  const double* _x = x.data();
  const Point point(mesh.geometry().dim(), _x);

  // Get index of first cell containing point
  unsigned int id
    = mesh.bounding_box_tree()->compute_first_entity_collision(point);

  // If not found, use the closest cell
  if (id == std::numeric_limits<unsigned int>::max())
  {
    if (_allow_extrapolation)
      id = mesh.bounding_box_tree()->compute_closest_entity(point).first;
    else
    {
      dolfin_error("Function.cpp",
                   "evaluate function at point",
                   "The point is not inside the domain. Consider calling \"Function::set_allow_extrapolation(true)\" on this Function to allow extrapolation");
    }
  }

  // Create cell that contains point
  const Cell cell(mesh, id);
  ufc::cell ufc_cell;
  cell.get_cell_data(ufc_cell);

  // Call evaluate function
  eval(values, x, cell, ufc_cell);
}
//-----------------------------------------------------------------------------
void Function::eval(Array<double>& values, const Array<double>& x,
                    const Cell& dolfin_cell, const ufc::cell& ufc_cell) const
{
  // Developer note: work arrays/vectors are re-created each time this
  //                 function is called for thread-safety

  dolfin_assert(_function_space->element());
  const FiniteElement& element = *_function_space->element();

  // Compute in tensor (one for scalar function, . . .)
  const std::size_t value_size_loc = value_size();

  dolfin_assert(values.size() == value_size_loc);

  // Create work vector for expansion coefficients
  std::vector<double> coefficients(element.space_dimension());

  // Cell coordinates (re-allocated inside function for thread safety)
  std::vector<double> coordinate_dofs;
  dolfin_cell.get_coordinate_dofs(coordinate_dofs);

  // Restrict function to cell
  restrict(coefficients.data(), element, dolfin_cell,
           coordinate_dofs.data(), ufc_cell);

  // Create work vector for basis
  std::vector<double> basis(value_size_loc);

  // Initialise values
  for (std::size_t j = 0; j < value_size_loc; ++j)
    values[j] = 0.0;

  // Compute linear combination
  for (std::size_t i = 0; i < element.space_dimension(); ++i)
  {
    element.evaluate_basis(i, basis.data(), x.data(),
                           coordinate_dofs.data(),
                           ufc_cell.orientation);
    for (std::size_t j = 0; j < value_size_loc; ++j)
      values[j] += coefficients[i]*basis[j];
  }
}
//-----------------------------------------------------------------------------
void Function::interpolate(const GenericFunction& v)
{
  dolfin_assert(_vector);
  dolfin_assert(_function_space);

  // Interpolate
  _function_space->interpolate(*_vector, v);
}
//-----------------------------------------------------------------------------
void Function::extrapolate(const Function& v)
{
  Extrapolation::extrapolate(*this, v);
}
//-----------------------------------------------------------------------------
std::size_t Function::value_rank() const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->element());
  return _function_space->element()->value_rank();
}
//-----------------------------------------------------------------------------
std::size_t Function::value_dimension(std::size_t i) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->element());
  return _function_space->element()->value_dimension(i);
}
//-----------------------------------------------------------------------------
void Function::eval(Array<double>& values,
                    const Array<double>& x,
                    const ufc::cell& ufc_cell) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  // Check if UFC cell comes from mesh, otherwise
  // find the cell which contains the point
  dolfin_assert(ufc_cell.mesh_identifier >= 0);
  if (ufc_cell.mesh_identifier == (int) mesh.id())
  {
    const Cell cell(mesh, ufc_cell.index);
    eval(values, x, cell, ufc_cell);
  }
  else
    eval(values, x);
}
//-----------------------------------------------------------------------------
void Function::restrict(double* w, const FiniteElement& element,
                        const Cell& dolfin_cell,
                        const double* coordinate_dofs,
                        const ufc::cell& ufc_cell) const
{
  dolfin_assert(w);
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->dofmap());

  // Check if we are restricting to an element of this function space
  if (_function_space->has_element(element)
      && _function_space->has_cell(dolfin_cell))
  {
    // Get dofmap for cell
    const GenericDofMap& dofmap = *_function_space->dofmap();
    const ArrayView<const dolfin::la_index> dofs
      = dofmap.cell_dofs(dolfin_cell.index());

    // Note: We should have dofmap.max_element_dofs() == dofs.size() here.
    // Pick values from vector(s)
    _vector->get_local(w, dofs.size(), dofs.data());
  }
  else
  {
    // Restrict as UFC function (by calling eval)
    restrict_as_ufc_function(w, element, dolfin_cell, coordinate_dofs,
                             ufc_cell);
  }
}
//-----------------------------------------------------------------------------
void Function::compute_vertex_values(std::vector<double>& vertex_values,
                                     const Mesh& mesh) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());

  // Check that the mesh matches. Notice that the hash is only
  // compared if the pointers are not matching.
  if (&mesh != _function_space->mesh().get()
      && mesh.hash() != _function_space->mesh()->hash())
  {
    dolfin_error("Function.cpp",
                 "interpolate function values at vertices",
                 "Non-matching mesh");
  }

  // Get finite element
  dolfin_assert(_function_space->element());
  const FiniteElement& element = *_function_space->element();

  // Local data for interpolation on each cell
  const std::size_t num_cell_vertices
    = mesh.type().num_vertices(mesh.topology().dim());

  // Compute in tensor (one for scalar function, . . .)
  const std::size_t value_size_loc = value_size();

  // Resize Array for holding vertex values
  vertex_values.resize(value_size_loc*(mesh.num_vertices()));

  // Create vector to hold cell vertex values
  std::vector<double> cell_vertex_values(value_size_loc*num_cell_vertices);

  // Create vector for expansion coefficients
  std::vector<double> coefficients(element.space_dimension());

  // Interpolate vertex values on each cell (using last computed value
  // if not continuous, e.g. discontinuous Galerkin methods)
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;
  for (CellIterator cell(mesh, "all"); !cell.end(); ++cell)
  {
    // Update to current cell
    cell->get_coordinate_dofs(coordinate_dofs);
    cell->get_cell_data(ufc_cell);

    // Pick values from global vector
    restrict(coefficients.data(), element, *cell, coordinate_dofs.data(),
             ufc_cell);

    // Interpolate values at the vertices
    element.interpolate_vertex_values(cell_vertex_values.data(),
                                      coefficients.data(),
                                      coordinate_dofs.data(),
                                      ufc_cell.orientation,
                                      ufc_cell);

    // Copy values to array of vertex values
    for (VertexIterator vertex(*cell); !vertex.end(); ++vertex)
    {
      for (std::size_t i = 0; i < value_size_loc; ++i)
      {
        const std::size_t local_index  = vertex.pos()*value_size_loc + i;
        const std::size_t global_index = i*mesh.num_vertices()+vertex->index();
        vertex_values[global_index] = cell_vertex_values[local_index];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void Function::compute_vertex_values(std::vector<double>& vertex_values)
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());
  compute_vertex_values(vertex_values, *_function_space->mesh());
}
//-----------------------------------------------------------------------------
void Function::init_vector()
{
  Timer timer("Init dof vector");

  // Get dof map
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->dofmap());
  const GenericDofMap& dofmap = *(_function_space->dofmap());

  // Check that function space is not a subspace (view)
  if (dofmap.is_view())
  {
    dolfin_error("Function.cpp",
                 "initialize vector of degrees of freedom for function",
                 "Cannot be created from subspace. Consider collapsing the "
                 "function space");
  }

  // Get index map
  std::shared_ptr<const IndexMap> index_map = dofmap.index_map();
  dolfin_assert(index_map);

  DefaultFactory factory;

  // Create layout for initialising tensor
  std::shared_ptr<TensorLayout> tensor_layout;
  tensor_layout = factory.create_layout(1);
  dolfin_assert(tensor_layout);
  dolfin_assert(!tensor_layout->sparsity_pattern());
  dolfin_assert(_function_space->mesh());
  tensor_layout->init(_function_space->mesh()->mpi_comm(), {index_map},
                      TensorLayout::Ghosts::GHOSTED);

  // Create vector of dofs
  if (!_vector)
    _vector = factory.create_vector(_function_space->mesh()->mpi_comm());
  dolfin_assert(_vector);
  if (!_vector->empty())
  {
    dolfin_error("Function.cpp",
                 "initialize vector of degrees of freedom for function",
                 "Cannot re-initialize a non-empty vector. Consider creating a new function");

  }
  _vector->init(*tensor_layout);
  _vector->zero();
}
//-----------------------------------------------------------------------------
void
Function::compute_ghost_indices(std::pair<std::size_t, std::size_t> range,
                                std::vector<la_index>& ghost_indices) const
{
  // Clear data
  ghost_indices.clear();

  // Get mesh
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());
  const Mesh& mesh = *_function_space->mesh();

  // Get dof map
  dolfin_assert(_function_space->dofmap());
  const GenericDofMap& dofmap = *(_function_space->dofmap());

  // Get local range
  const std::size_t n0 = range.first;
  const std::size_t n1 = range.second;

  // Iterate over local mesh and check which dofs are needed
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get dofs on cell
    auto dofs = dofmap.cell_dofs(cell->index());
    for (std::size_t d = 0; d < dofs.size(); ++d)
    {
      const std::size_t dof = dofs[d];
      if (dof < n0 || dof >= n1)
      {
        // FIXME: Could we use dolfin::Set here? Or unordered_set?
        if (std::find(ghost_indices.begin(), ghost_indices.end(), dof)
            == ghost_indices.end())
        {
          ghost_indices.push_back(dof);
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
