// Copyright (C) 2009-2018 Michal Habera, Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Expression.h"
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>

using namespace dolfin;
using namespace dolfin::function;

//-----------------------------------------------------------------------------
Expression::Expression(std::vector<int> value_shape) : _value_shape(value_shape)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Expression::set_eval(
    const std::function<void(
        Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>,
        const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                            Eigen::Dynamic, Eigen::RowMajor>>)>&
        eval_ptr)
{
  _eval_eigen_fn = eval_ptr;
}
//-----------------------------------------------------------------------------
void Expression::set_eval_c(
    const std::function<void(PetscScalar*, int, int, const double*, int)>&
        eval_fn)
{
  _eval_ptr = eval_fn;
}
//-----------------------------------------------------------------------------
int Expression::value_rank() const { return _value_shape.size(); }
//-----------------------------------------------------------------------------
int Expression::value_dimension(int i) const
{
  if (i >= (int)_value_shape.size())
  {
    throw std::runtime_error("Illegal axis " + std::to_string(i)
                             + " for value dimension for value of rank "
                             + std::to_string(_value_shape.size()));
  }

  return _value_shape[i];
}
//-----------------------------------------------------------------------------
std::vector<int> Expression::value_shape() const { return _value_shape; }
//-----------------------------------------------------------------------------
Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Expression::compute_point_values(const mesh::Mesh& mesh) const
{
  // Get vertex coordinates
  const int num_vertices_per_cell = mesh.type().num_entities(0);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x
      = mesh.geometry().points().leftCols(num_vertices_per_cell);

  // Prepare data structure for vertex values
  const int size = std::accumulate(
      std::begin(_value_shape), std::end(_value_shape), 1, std::multiplies<>());
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      vertex_values(x.rows(), size);

  // Evaluate Expression at x
  eval(vertex_values, x);

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
  assert(values.rows() == x.rows());
  if (_eval_eigen_fn)
    _eval_eigen_fn(values, x);
  else
  {
    assert(_eval_ptr);
    _eval_ptr(values.data(), values.rows(), values.cols(), x.data(), x.cols());
  }
}
//-----------------------------------------------------------------------------
