// Copyright (C) 2009-2018 Michal Habera, Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Expression.h"
#include <dolfin/fem/CoordinateMapping.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <spdlog/spdlog.h>

using namespace dolfin;
using namespace dolfin::function;

//-----------------------------------------------------------------------------
Expression::Expression(std::vector<std::size_t> value_shape)
    : _value_shape(value_shape)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Expression::Expression(
    std::function<void(PetscScalar* values, const double* x,
                       const int64_t* cell_idx, int num_points, int value_size,
                       int gdim, int num_cells)>
        eval_ptr,
    std::vector<std::size_t> value_shape)
    : _eval_ptr(eval_ptr), _value_shape(value_shape)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t Expression::value_rank() const { return _value_shape.size(); }
//-----------------------------------------------------------------------------
std::size_t Expression::value_dimension(std::size_t i) const
{
  if (i >= _value_shape.size())
  {
    spdlog::error("Expression.cpp", "evaluate expression",
                  "Illegal axis %d for value dimension for value of rank %d", i,
                  _value_shape.size());
    throw std::runtime_error("Value dimension axis");
  }
  return _value_shape[i];
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> Expression::value_shape() const
{
  return _value_shape;
}
//-----------------------------------------------------------------------------
void Expression::restrict(
    PetscScalar* w, const fem::FiniteElement& element, const mesh::Cell& cell,
    const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs) const
{
  // Get evaluation points
  const std::size_t value_size = std::accumulate(
      std::begin(_value_shape), std::end(_value_shape), 1, std::multiplies<>());
  const std::size_t ndofs = element.space_dimension();
  const std::size_t gdim = cell.mesh().geometry().dim();

  // FIXME: for Vector Lagrange elements (and probably Tensor too),
  // this repeats the same evaluation points "gdim" times. Should only
  // do them once, and remove the "mapping" below (which is the identity).

  // Get dof coordinates on reference element
  const EigenRowArrayXXd& X = element.dof_reference_coordinates();

  // Get coordinate mapping
  if (!cell.mesh().geometry().coord_mapping)
  {
    throw std::runtime_error(
        "CoordinateMapping has not been attached to mesh.");
  }
  const fem::CoordinateMapping& cmap = *cell.mesh().geometry().coord_mapping;

  // FIXME: Avoid dynamic memory allocation
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      eval_points(ndofs, gdim);
  cmap.compute_physical_coordinates(eval_points, X, coordinate_dofs);

  // FIXME: Avoid dynamic memory allocation
  // Storage for evaluation values
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      eval_values(ndofs, value_size);

  // Evaluate all points in one call
  eval(eval_values, eval_points, cell);

  // FIXME: *do not* use UFC directly
  // Apply a mapping to the reference element.
  // FIXME: not needed for Lagrange elements, eliminate.
  // See: ffc/uflacs/backends/ufc/evaluatedof.py:_change_variables()
  element.transform_values(w, eval_values, coordinate_dofs);
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Expression::compute_point_values(const mesh::Mesh& mesh) const
{
  // Local data for vertex values
  const std::size_t size = std::accumulate(
      std::begin(_value_shape), std::end(_value_shape), 1, std::multiplies<>());
  Eigen::Matrix<PetscScalar, 1, Eigen::Dynamic> local_vertex_values(size);

  // Resize vertex_values
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      vertex_values(mesh.num_vertices(), size);

  // Iterate over cells, overwriting values when repeatedly visiting vertices
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    // Iterate over cell vertices
    for (auto& vertex : mesh::EntityRange<mesh::Vertex>(cell))
    {
      // Wrap coordinate data
      const Eigen::Ref<const Eigen::VectorXd> x = vertex.x();

      // Evaluate at vertex
      eval(local_vertex_values, x, cell);

      // Copy to array
      vertex_values.row(vertex.index()) = local_vertex_values;
    }
  }

  return vertex_values;
}
//-----------------------------------------------------------------------------
void Expression::eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic,
                                              Eigen::Dynamic, Eigen::RowMajor>>
                          values,
                      const Eigen::Ref<const EigenRowArrayXXd> x,
                      const mesh::Cell& cell) const
{
  const int64_t cell_idx = cell.index();
  assert(_eval_ptr);
  _eval_ptr(values.data(), x.data(), &cell_idx, x.rows(), values.cols(),
            x.cols(), 1);
}
//-----------------------------------------------------------------------------
