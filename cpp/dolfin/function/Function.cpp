// Copyright (C) 2003-2012 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Function.h"
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
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
#include <vector>

#include <ufc.h>

using namespace dolfin;
using namespace dolfin::function;

//-----------------------------------------------------------------------------
Function::Function(std::shared_ptr<const FunctionSpace> V) : _function_space(V)
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
    : _function_space(V), _vector(x)
{
  // We do not check for a subspace since this constructor is used for
  // creating subfunctions

  // Assertion uses '<=' to deal with sub-functions
  dolfin_assert(V->dofmap());
  dolfin_assert(V->dofmap()->global_dimension() <= x->size());
}
//-----------------------------------------------------------------------------
Function::Function(const Function& v)
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
  std::vector<std::pair<double,
                        std::shared_ptr<const Function>>>::const_iterator it;
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
void Function::eval(Eigen::Ref<EigenRowArrayXXd> values,
                    Eigen::Ref<const EigenRowArrayXXd> x) const
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

      if (close.second < DOLFIN_EPS)
        id = close.first;
      else
      {
        log::dolfin_error("Function.cpp", "evaluate function at point",
                          "The point is not inside the domain.");
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
void Function::eval(Eigen::Ref<EigenRowArrayXXd> values,
                    Eigen::Ref<const EigenRowArrayXXd> x,
                    const mesh::Cell& dolfin_cell,
                    const ufc::cell& ufc_cell) const
{
  dolfin_assert(x.rows() == values.rows());
  dolfin_assert(_function_space->element());
  const fem::FiniteElement& element = *_function_space->element();

  // Create work vector for expansion coefficients
  Eigen::RowVectorXd coefficients(element.space_dimension());

  // Cell coordinates (re-allocated inside function for thread safety)
  std::vector<double> coordinate_dofs;
  dolfin_cell.get_coordinate_dofs(coordinate_dofs);

  // Restrict function to cell
  restrict(coefficients.data(), element, dolfin_cell, coordinate_dofs.data(),
           ufc_cell);

  // Get coordinate mapping
  auto cmap = _function_space->mesh()->geometry().ufc_coord_mapping;
  // assert(cmap);
  // if (!cmap)
  // {
  //   throw std::runtime_error(
  //       "ufc::coordinate_mapping has not been attached to mesh.");
  // }

  if (cmap)
  {
    // New implementation using ufc::coordinate_mappping

    std::size_t num_points = x.rows();
    std::size_t gdim = _function_space->mesh()->geometry().dim();
    std::size_t tdim = _function_space->mesh()->topology().dim();

    auto ufc_element = _function_space->element()->ufc_element();
    std::size_t reference_value_size = ufc_element->reference_value_size();
    std::size_t value_size = ufc_element->value_size();
    std::size_t space_dimension = ufc_element->space_dimension();

    Eigen::Tensor<double, 3, Eigen::RowMajor> J(num_points, gdim, tdim);
    EigenArrayXd detJ(num_points);
    Eigen::Tensor<double, 3, Eigen::RowMajor> K(num_points, tdim, gdim);

    // EigenRowArrayXXd X(x.rows(), tdim) ;
    EigenRowArrayXXd X(x.rows(), tdim);

    // boost::multi_array<double, 3> basis_reference_values(
    //     boost::extents[num_points][space_dimension][reference_value_size]);
    Eigen::Tensor<double, 3, Eigen::RowMajor> basis_reference_values(
        num_points, space_dimension, reference_value_size);

    Eigen::Tensor<double, 3, Eigen::RowMajor> basis_values(
        num_points, space_dimension, value_size);

    // Compute reference coordinates X, and J, detJ and K
    cmap->compute_reference_geometry(X.data(), J.data(), detJ.data(), K.data(),
                                     num_points, x.data(),
                                     coordinate_dofs.data(), 1);

    // std::cout << "Physical x: " << std::endl;
    // std::cout << x << std::endl;
    // std::cout << "Reference X: " << std::endl;
    // std::cout << X << std::endl;

    // // Compute basis on reference element
    element.evaluate_reference_basis(basis_reference_values, X);

    // // Push basis forward to physical element
    element.transform_reference_basis(basis_values, basis_reference_values, X,
                                      J, detJ, K);

    // Compute expansion
    // std::cout << "Num points, space dim, value_size: " << num_points << ", "
    //           << space_dimension << ", " << value_size << std::endl;
    values.setZero();
    for (std::size_t p = 0; p < num_points; ++p)
    {
      for (std::size_t i = 0; i < space_dimension; ++i)
      {
        for (std::size_t j = 0; j < value_size; ++j)
        {
          // std::cout << "Loop: " << p << ", " << i << ", " << j << std::endl;
          // std::cout << "  Coeff, Basis: " << coefficients[i] << ", "
          //           << basis_values(p, i, j) << std::endl;

          // TODO: Find an Eigen shortcut fot this operation
          values.row(p)[j] += coefficients[i] * basis_values(p, i, j);
        }
      }
    }
  }
  else
  {
    // Old implementation

    // Compute in tensor (one for scalar function, . . .)
    const std::size_t value_size_loc = value_size();

    dolfin_assert((std::size_t)values.cols() == value_size_loc);

    // Create work space for basis
    EigenRowArrayXXd basis(element.space_dimension(), value_size_loc);

    // Compute linear combination for each row of x
    for (unsigned int k = 0; k < x.rows(); ++k)
    {
      element.evaluate_basis_all(basis.data(), x.row(k).data(),
                                 coordinate_dofs.data(), ufc_cell.orientation);

      values.row(k).matrix() = coefficients.matrix() * basis.matrix();
    }
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
void Function::eval(Eigen::Ref<EigenRowArrayXXd> values,
                    Eigen::Ref<const EigenRowArrayXXd> x,
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
EigenRowArrayXXd Function::compute_vertex_values(const mesh::Mesh& mesh) const
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

  // Local data for interpolation on each cell
  const std::size_t num_cell_vertices
      = mesh.type().num_vertices(mesh.topology().dim());

  // Compute in tensor (one for scalar function, . . .)
  const std::size_t value_size_loc = value_size();

  // Resize Array for holding vertex values
  EigenRowArrayXXd vertex_values(mesh.num_vertices(), value_size_loc);

  // Interpolate vertex values on each cell (using last computed value
  // if not continuous, e.g. discontinuous Galerkin methods)
  ufc::cell ufc_cell;
  EigenRowArrayXXd x(num_cell_vertices, mesh.geometry().dim());
  EigenRowArrayXXd values(num_cell_vertices, value_size_loc);

  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    // Update to current cell
    cell.get_cell_data(ufc_cell);
    cell.get_coordinate_dofs(x);

    // Call evaluate function
    eval(values, x, cell, ufc_cell);

    // Copy values to array of vertex values
    std::size_t local_index = 0;
    for (auto& vertex : mesh::EntityRange<mesh::Vertex>(cell))
    {
      vertex_values.row(vertex.index()) = values.row(local_index);
      ++local_index;
    }
  }

  return vertex_values;
}
//-----------------------------------------------------------------------------
EigenRowArrayXXd Function::compute_vertex_values() const
{
  assert(_function_space);
  assert(_function_space->mesh());
  return compute_vertex_values(*_function_space->mesh());
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
