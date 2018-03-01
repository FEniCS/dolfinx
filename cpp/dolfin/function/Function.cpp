// Copyright (C) 2003-2012 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Function.h"
#include "Expression.h"
#include "FunctionSpace.h"
#include <algorithm>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/utils.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <map>
#include <utility>
#include <vector>

using namespace dolfin;
using namespace dolfin::function;

//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V)
    : _function_space(V), _allow_extrapolation(false)
{
  // Check that we don't have a subspace
  if (!V->component().empty())
  {
    log::dolfin_error(
        "Function.cpp", "create function",
        "Cannot be created from subspace. Consider collapsing the "
        "function space");
  }

  // Initialize vector
  init_vector();
}
//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V,
                   std::shared_ptr<la::PETScVector> x)
    : _function_space(V), _vector(x), _allow_extrapolation(false)
{
  // We do not check for a subspace since this constructor is used for
  // creating subfunctions

  // Assertion uses '<=' to deal with sub-functions
  dolfin_assert(V->dofmap());
  dolfin_assert(V->dofmap()->global_dimension() <= x->size());
}
//-----------------------------------------------------------------------------
Function::Function(const Function& v) : _allow_extrapolation(false)
{
  // Make a copy of all the data, or if v is a sub-function, then we
  // collapse the dof map and copy only the relevant entries from the
  // vector of v.
  dolfin_assert(v._vector);
  if (v._vector->size() == v._function_space->dim())
  {
    // Copy function space pointer
    this->_function_space = v._function_space;

    // Copy vector
    this->_vector = std::make_shared<la::PETScVector>(*v._vector);
  }
  else
  {
    // Create new collapsed FunctionSpace
    std::unordered_map<std::size_t, std::size_t> collapsed_map;
    _function_space = v._function_space->collapse(collapsed_map);

    // Get row indices of original and new vectors
    std::unordered_map<std::size_t, std::size_t>::const_iterator entry;
    std::vector<dolfin::la_index_t> new_rows(collapsed_map.size());
    std::vector<dolfin::la_index_t> old_rows(collapsed_map.size());
    std::size_t i = 0;
    for (entry = collapsed_map.begin(); entry != collapsed_map.end(); ++entry)
    {
      new_rows[i] = entry->first;
      old_rows[i++] = entry->second;
    }

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
    this->_vector->apply();
  }
}
//-----------------------------------------------------------------------------
Function::~Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
/*
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
    std::vector<dolfin::la_index_t> new_rows(collapsed_map.size());
    std::vector<dolfin::la_index_t> old_rows(collapsed_map.size());
    std::size_t i = 0;
    for (entry = collapsed_map.begin(); entry != collapsed_map.end(); ++entry)
    {
      new_rows[i]   = entry->first;
      old_rows[i++] = entry->second;
    }

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

  return *this;
}
*/
//-----------------------------------------------------------------------------
const Function& Function::operator=(const Expression& v)
{
  interpolate(v);
  return *this;
}
//-----------------------------------------------------------------------------
Function Function::sub(std::size_t i) const
{
  // Extract function subspace
  auto sub_space = _function_space->sub({i});

  // Return sub-function
  dolfin_assert(sub_space);
  dolfin_assert(_vector);
  return Function(sub_space, _vector);
}
//-----------------------------------------------------------------------------
void Function::operator=(const function::FunctionAXPY& axpy)
{
  if (axpy.pairs().size() == 0)
  {
    log::dolfin_error("Function.cpp", "assign function",
                      "FunctionAXPY is empty.");
  }

  // Make an initial assign and scale
  dolfin_assert(axpy.pairs()[0].second);
  *this = *(axpy.pairs()[0].second);
  if (axpy.pairs()[0].first != 1.0)
    *_vector *= axpy.pairs()[0].first;

  // Start from item 2 and axpy
  std::vector<
      std::pair<double, std::shared_ptr<const Function>>>::const_iterator it;
  for (it = axpy.pairs().begin() + 1; it != axpy.pairs().end(); it++)
  {
    dolfin_assert(it->second);
    dolfin_assert(it->second->vector());
    _vector->axpy(it->first, *(it->second->vector()));
  }
}
//-----------------------------------------------------------------------------
std::shared_ptr<la::PETScVector> Function::vector()
{
  dolfin_assert(_vector);
  dolfin_assert(_function_space->dofmap());

  // Check that this is not a sub function.
  if (_vector->size() != _function_space->dofmap()->global_dimension())
  {
    log::dolfin_error("Function.cpp", "access vector of degrees of freedom",
                      "Cannot access a non-const vector from a subfunction");
  }

  return _vector;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const la::PETScVector> Function::vector() const
{
  dolfin_assert(_vector);
  return _vector;
}
//-----------------------------------------------------------------------------
void Function::eval(Eigen::Ref<EigenRowMatrixXd> values,
                    Eigen::Ref<const EigenRowMatrixXd> x) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());
  const mesh::Mesh& mesh = *_function_space->mesh();

  // Find the cell that contains x
  for (unsigned int i = 0; i != x.rows(); ++i)
  {
    const double* _x = x.row(i).data();
    const geometry::Point point(mesh.geometry().dim(), _x);

    // Get index of first cell containing point
    unsigned int id
        = mesh.bounding_box_tree()->compute_first_entity_collision(point, mesh);

    // If not found, use the closest cell
    if (id == std::numeric_limits<unsigned int>::max())
    {
      // Check if the closest cell is within DOLFIN_EPS. This we can
      // allow without _allow_extrapolation
      std::pair<unsigned int, double> close
          = mesh.bounding_box_tree()->compute_closest_entity(point, mesh);

      if (_allow_extrapolation or close.second < DOLFIN_EPS)
        id = close.first;
      else
      {
        log::dolfin_error(
            "Function.cpp", "evaluate function at point",
            "The point is not inside the domain. Consider calling "
            "\"Function::set_allow_extrapolation(true)\" on this "
            "Function to allow extrapolation");
      }
    }

    // Create cell that contains point
    const mesh::Cell cell(mesh, id);
    ufc::cell ufc_cell;
    cell.get_cell_data(ufc_cell);

    // Call evaluate function
    eval(values.row(i), x.row(i), cell, ufc_cell);
  }
}
//-----------------------------------------------------------------------------
void Function::eval(Eigen::Ref<EigenRowMatrixXd> values,
                    Eigen::Ref<const EigenRowMatrixXd> x,
                    const mesh::Cell& dolfin_cell,
                    const ufc::cell& ufc_cell) const
{
  // Developer note: work arrays/vectors are re-created each time this
  //                 function is called for thread-safety

  dolfin_assert(_function_space->element());
  const fem::FiniteElement& element = *_function_space->element();

  // Compute in tensor (one for scalar function, . . .)
  const std::size_t value_size_loc = value_size();

  dolfin_assert((std::size_t)values.size() == value_size_loc);

  // Create work vector for expansion coefficients
  std::vector<double> coefficients(element.space_dimension());

  // Cell coordinates (re-allocated inside function for thread safety)
  std::vector<double> coordinate_dofs;
  dolfin_cell.get_coordinate_dofs(coordinate_dofs);

  // Restrict function to cell
  restrict(coefficients.data(), element, dolfin_cell, coordinate_dofs.data(),
           ufc_cell);

  // Create work vector for basis
  std::vector<double> basis(value_size_loc);

  // Initialise values
  values.setZero();
  //  for (std::size_t j = 0; j < value_size_loc; ++j)
  //    values[j] = 0.0;

  // Compute linear combination
  std::size_t k = 1;
  for (std::size_t i = 0; i < element.space_dimension(); ++i)
  {
    element.evaluate_basis(i, basis.data(), x.data(), coordinate_dofs.data(),
                           ufc_cell.orientation);

    for (std::size_t j = 0; j < value_size_loc; ++j)
      values(k, j) += coefficients[i] * basis[j];
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
  dolfin_not_implemented();
  // Was in "adaptivity"
  //  Extrapolation::extrapolate(*this, v);
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
std::vector<std::size_t> Function::value_shape() const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->element());
  std::vector<std::size_t> _shape(this->value_rank(), 1);
  for (std::size_t i = 0; i < _shape.size(); ++i)
    _shape[i] = this->value_dimension(i);
  return _shape;
}
//-----------------------------------------------------------------------------
void Function::eval(Eigen::Ref<EigenRowMatrixXd> values,
                    Eigen::Ref<const EigenRowMatrixXd> x,
                    const ufc::cell& ufc_cell) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());
  const mesh::Mesh& mesh = *_function_space->mesh();

  // Check if UFC cell comes from mesh, otherwise
  // find the cell which contains the point
  dolfin_assert(ufc_cell.mesh_identifier >= 0);
  if (ufc_cell.mesh_identifier == (int)mesh.id())
  {
    const mesh::Cell cell(mesh, ufc_cell.index);
    eval(values, x, cell, ufc_cell);
  }
  else
    eval(values, x);
}
//-----------------------------------------------------------------------------
void Function::restrict(double* w, const fem::FiniteElement& element,
                        const mesh::Cell& dolfin_cell,
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
    const fem::GenericDofMap& dofmap = *_function_space->dofmap();
    auto dofs = dofmap.cell_dofs(dolfin_cell.index());

    // Note: We should have dofmap.max_element_dofs() == dofs.size() here.
    // Pick values from vector(s)
    _vector->get_local(w, dofs.size(), dofs.data());
  }
  else
    dolfin_not_implemented();

  //  {
  //    // Restrict as UFC function (by calling eval)
  //    element.evaluate_dofs(w, *this, coordinate_dofs, ufc_cell.orientation,
  //                          ufc_cell);
  //  }
}
//-----------------------------------------------------------------------------
void Function::compute_vertex_values(std::vector<double>& vertex_values,
                                     const mesh::Mesh& mesh) const
{
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->mesh());

  // Check that the mesh matches. Notice that the hash is only
  // compared if the pointers are not matching.
  if (&mesh != _function_space->mesh().get()
      && mesh.hash() != _function_space->mesh()->hash())
  {
    log::dolfin_error("Function.cpp", "interpolate function values at vertices",
                      "Non-matching mesh");
  }

  // Get finite element
  dolfin_assert(_function_space->element());
  const fem::FiniteElement& element = *_function_space->element();

  // Local data for interpolation on each cell
  const std::size_t num_cell_vertices
      = mesh.type().num_vertices(mesh.topology().dim());

  // Compute in tensor (one for scalar function, . . .)
  const std::size_t value_size_loc = value_size();

  // Resize Array for holding vertex values
  vertex_values.resize(value_size_loc * (mesh.num_vertices()));

  // Create vector to hold cell vertex values
  std::vector<double> cell_vertex_values(value_size_loc * num_cell_vertices);

  // Create vector for expansion coefficients
  std::vector<double> coefficients(element.space_dimension());

  // Interpolate vertex values on each cell (using last computed value
  // if not continuous, e.g. discontinuous Galerkin methods)
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    // Update to current cell
    cell.get_coordinate_dofs(coordinate_dofs);
    cell.get_cell_data(ufc_cell);

    // Pick values from global vector
    restrict(coefficients.data(), element, cell, coordinate_dofs.data(),
             ufc_cell);

    // Interpolate values at the vertices
    element.interpolate_vertex_values(
        cell_vertex_values.data(), coefficients.data(), coordinate_dofs.data(),
        ufc_cell.orientation);

    // Copy values to array of vertex values
    std::size_t local_index = 0;
    for (auto& vertex : mesh::EntityRange<mesh::Vertex>(cell))
    {
      for (std::size_t i = 0; i < value_size_loc; ++i)
      {
        const std::size_t global_index
            = i * mesh.num_vertices() + vertex.index();
        vertex_values[global_index] = cell_vertex_values[local_index];
        ++local_index;
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
  common::Timer timer("Init dof vector");

  // Get dof map
  dolfin_assert(_function_space);
  dolfin_assert(_function_space->dofmap());
  const fem::GenericDofMap& dofmap = *(_function_space->dofmap());

  // Check that function space is not a subspace (view)
  if (dofmap.is_view())
  {
    log::dolfin_error(
        "Function.cpp", "initialize vector of degrees of freedom for function",
        "Cannot be created from subspace. Consider collapsing the "
        "function space");
  }

  // Get index map
  /*
  std::shared_ptr<const common::IndexMap> index_map = dofmap.index_map();
  dolfin_assert(index_map);

  MPI_Comm comm = _function_space->mesh()->mpi_comm();

  // Create layout for initialising tensor
  //std::shared_ptr<TensorLayout> tensor_layout;
  //tensor_layout = factory.create_layout(comm, 1);
  auto tensor_layout = std::make_shared<TensorLayout>(comm, 0,
  TensorLayout::Sparsity::DENSE);

  dolfin_assert(tensor_layout);
  dolfin_assert(!tensor_layout->sparsity_pattern());
  dolfin_assert(_function_space->mesh());
  tensor_layout->init({index_map}, TensorLayout::Ghosts::GHOSTED);

  // Create vector of dofs
  if (!_vector)
    _vector =
  std::make_shared<la::la::PETScVector>(_function_space->mesh()->mpi_comm());
  dolfin_assert(_vector);
  if (!_vector->empty())
  {
    log::dolfin_error("Function.cpp",
                 "initialize vector of degrees of freedom for function",
                 "Cannot re-initialize a non-empty vector. Consider creating a
  new function");

  }
  _vector->init(*tensor_layout);
  _vector->zero();
  */

  // Get index map
  std::shared_ptr<const common::IndexMap> index_map = dofmap.index_map();
  dolfin_assert(index_map);

  // Get block size
  std::size_t bs = index_map->block_size();

  // Build local-to-global map (blocks)
  std::vector<dolfin::la_index_t> local_to_global(
      index_map->size(common::IndexMap::MapSize::ALL));
  for (std::size_t i = 0; i < local_to_global.size(); ++i)
    local_to_global[i] = index_map->local_to_global(i);

  // Build list of ghosts (global block indices)
  const std::size_t nowned = index_map->size(common::IndexMap::MapSize::OWNED);
  dolfin_assert(nowned + index_map->size(common::IndexMap::MapSize::UNOWNED)
                == local_to_global.size());
  std::vector<dolfin::la_index_t> ghosts(local_to_global.begin() + nowned,
                                         local_to_global.end());

  // Create vector of dofs
  if (!_vector)
    _vector = std::make_shared<la::PETScVector>(
        _function_space->mesh()->mpi_comm());
  dolfin_assert(_vector);

  if (!_vector->empty())
  {
    log::dolfin_error(
        "Function.cpp", "initialize vector of degrees of freedom for function",
        "Cannot re-initialize a non-empty vector. Consider creating a "
        "new function");
  }

  _vector->init(index_map->local_range(), local_to_global, ghosts, bs);
  _vector->zero();
}
//-----------------------------------------------------------------------------
