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
    std::function<void(PetscScalar* values, int num_points, int value_size,
                       const double* x, int gdim, double t)>
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
    throw std::runtime_error("Illegal axis " + std::to_string(i)
                             + " for value dimension for value of rank "
                             + std::to_string(_value_shape.size()));
  }

  return _value_shape[i];
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> Expression::value_shape() const
{
  return _value_shape;
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
      vertex_values(mesh.num_entities(0), size);

  // Iterate over cells, overwriting values when repeatedly visiting vertices
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    // Iterate over cell vertices
    for (auto& vertex : mesh::EntityRange<mesh::Vertex>(cell))
    {
      // Wrap coordinate data
      const Eigen::Ref<const Eigen::VectorXd> x = vertex.x();

      // Evaluate at vertex
      eval(local_vertex_values, x);

      // Copy to array
      vertex_values.row(vertex.index()) = local_vertex_values;
    }
  }

  return vertex_values;
}
//-----------------------------------------------------------------------------
void Expression::eval(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>>
        values,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>
        x) const
{
  assert(_eval_ptr);
  assert(values.rows() == x.rows());
  _eval_ptr(values.data(), values.rows(), values.cols(), x.data(), x.cols(),
            this->t);
}
//-----------------------------------------------------------------------------
