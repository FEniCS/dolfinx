// Copyright (C) 2009-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Expression.h"
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/log/log.h>
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
Expression::Expression(const Expression& expression)
    : _value_shape(expression._value_shape)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Expression::~Expression()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Expression::eval(Eigen::Ref<EigenRowMatrixXd> values,
                      Eigen::Ref<const EigenRowMatrixXd> x,
                      const ufc::cell& cell) const
{
  // Redirect to simple eval
  eval(values, x);
}
//-----------------------------------------------------------------------------
void Expression::eval(Eigen::Ref<EigenRowMatrixXd> values,
                      Eigen::Ref<const EigenRowMatrixXd> x) const
{
  log::dolfin_error("Expression.cpp", "evaluate expression",
                    "Missing eval() function (must be overloaded)");
}
//-----------------------------------------------------------------------------
std::size_t Expression::value_rank() const { return _value_shape.size(); }
//-----------------------------------------------------------------------------
std::size_t Expression::value_dimension(std::size_t i) const
{
  if (i >= _value_shape.size())
  {
    log::dolfin_error(
        "Expression.cpp", "evaluate expression",
        "Illegal axis %d for value dimension for value of rank %d", i,
        _value_shape.size());
  }
  return _value_shape[i];
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> Expression::value_shape() const
{
  return _value_shape;
}
//-----------------------------------------------------------------------------
void Expression::set_property(std::string name, double value)
{
  log::dolfin_error("Expression.cpp", "set property",
                    "This method should be overloaded in the derived class");
}
//-----------------------------------------------------------------------------
double Expression::get_property(std::string name) const
{
  log::dolfin_error("Expression.cpp", "get property",
                    "This method should be overloaded in the derived class");
  return 0.0;
}
//-----------------------------------------------------------------------------
void Expression::set_generic_function(std::string name,
                                      std::shared_ptr<GenericFunction>)
{
  log::dolfin_error("Expression.cpp", "set property",
                    "This method should be overloaded in the derived class");
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericFunction>
Expression::get_generic_function(std::string name) const
{
  log::dolfin_error("Expression.cpp", "get property",
                    "This method should be overloaded in the derived class");
  return std::shared_ptr<GenericFunction>();
}
//-----------------------------------------------------------------------------
void Expression::restrict(double* w, const fem::FiniteElement& element,
                          const mesh::Cell& dolfin_cell,
                          const double* coordinate_dofs,
                          const ufc::cell& ufc_cell) const
{
  // Get evaluation points
  const std::size_t vs = value_size();
  const std::size_t ndofs = element.space_dimension();
  const std::size_t gdim = element.geometric_dimension();

  // FIXME: for Vector Lagrange elements (and probably Tensor too),
  // this repeats the same evaluation points "gdim" times. Should only
  // do them once, and remove the "mapping" below (which is the identity).

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      eval_points(ndofs, gdim);
  element.ufc_element()->tabulate_dof_coordinates(eval_points.data(),
                                                  coordinate_dofs);

  // Storage for evaluation values
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      eval_values(ndofs, vs);

  // Evaluate all points in one call
  eval(eval_values, eval_points, ufc_cell);

  // Transpose for vector values
  // FIXME: remove need for this - needs work in ffc
  eval_values.transposeInPlace();

  // Apply a mapping to the reference element.
  // FIXME: not needed for Lagrange elements, eliminate.
  // See: ffc/uflacs/backends/ufc/evaluatedof.py:_change_variables()
  element.ufc_element()->map_dofs(w, eval_values.data(), coordinate_dofs, -1);
}
//-----------------------------------------------------------------------------
EigenRowArrayXXd Expression::compute_vertex_values(const mesh::Mesh& mesh) const
{
  // Local data for vertex values
  const std::size_t size = value_size();
  Eigen::RowVectorXd local_vertex_values(size);

  // Resize vertex_values
  EigenRowArrayXXd vertex_values(mesh.num_vertices(), size);

  // Iterate over cells, overwriting values when repeatedly visiting vertices
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh, mesh::MeshRangeType::ALL))
  {
    // Iterate over cell vertices
    for (auto& vertex : mesh::EntityRange<mesh::Vertex>(cell))
    {
      // Wrap coordinate data
      Eigen::Map<const Eigen::VectorXd> x(vertex.x(), mesh.geometry().dim());

      // Evaluate at vertex
      eval(local_vertex_values, x);

      // Copy to array
      for (std::size_t i = 0; i < size; i++)
        vertex_values(vertex.index(), i) = local_vertex_values[i];
    }
  }

  return vertex_values;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FunctionSpace> Expression::function_space() const
{
  return std::shared_ptr<const FunctionSpace>();
}
//-----------------------------------------------------------------------------
