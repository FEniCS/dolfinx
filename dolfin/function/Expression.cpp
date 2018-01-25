// Copyright (C) 2009-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Expression.h"
#include "Transform.h"
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>

using namespace dolfin;

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
void Expression::eval(Eigen::Ref<Eigen::VectorXd> values,
                      Eigen::Ref<const Eigen::VectorXd> x,
                      const ufc::cell& cell) const
{
  // Redirect to simple eval
  eval(values, x);
}
//-----------------------------------------------------------------------------
void Expression::eval(Eigen::Ref<Eigen::VectorXd> values,
                      Eigen::Ref<const Eigen::VectorXd> x) const
{
  dolfin_error("Expression.cpp", "evaluate expression",
               "Missing eval() function (must be overloaded)");
}
//-----------------------------------------------------------------------------
std::size_t Expression::value_rank() const { return _value_shape.size(); }
//-----------------------------------------------------------------------------
std::size_t Expression::value_dimension(std::size_t i) const
{
  if (i >= _value_shape.size())
  {
    dolfin_error("Expression.cpp", "evaluate expression",
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
  dolfin_error("Expression.cpp", "set property",
               "This method should be overloaded in the derived class");
}
//-----------------------------------------------------------------------------
double Expression::get_property(std::string name) const
{
  dolfin_error("Expression.cpp", "get property",
               "This method should be overloaded in the derived class");
  return 0.0;
}
//-----------------------------------------------------------------------------
void Expression::set_generic_function(std::string name,
                                      std::shared_ptr<GenericFunction>)
{
  dolfin_error("Expression.cpp", "set property",
               "This method should be overloaded in the derived class");
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericFunction>
Expression::get_generic_function(std::string name) const
{
  dolfin_error("Expression.cpp", "get property",
               "This method should be overloaded in the derived class");
  return std::shared_ptr<GenericFunction>();
}
//-----------------------------------------------------------------------------
void Expression::restrict(double* w, const FiniteElement& element,
                          const Cell& dolfin_cell,
                          const double* coordinate_dofs,
                          const ufc::cell& ufc_cell) const
{
  // Not working for Hdiv, Hcurl elements etc.
  const std::string family(element.ufc_element()->family());
  if (family != "Lagrange")
    warning("This will probably crash or give wrong results for non-Lagrange elements.");

  // Get evaluation points
  const std::size_t vs = value_size();
  const std::size_t sd = element.space_dimension();
  const std::size_t gdim = element.geometric_dimension();

  std::cout << family << " " << vs << " " << sd << " " << gdim << "\n";

  std::size_t ndofs = sd;
  if (family == "Lagrange")
    ndofs /= vs;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      eval_points(sd, gdim);
  element.ufc_element()->tabulate_dof_coordinates(eval_points.data(),
                                                  coordinate_dofs);

  // Storage for evaluation values
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      eval_values(ndofs, vs);

  // FIXME: should evaluate all at once (maybe needs RowMajor matrix)
  for (unsigned int i = 0; i != ndofs; ++i)
  {
    eval(eval_values.row(i), eval_points.row(i), ufc_cell);
    for (unsigned int j = 0; j != gdim; ++j)
      std::cout << eval_points(i, j) << ", " ;
    for (unsigned int j = 0; j != vs; ++j)
      std::cout << eval_values(i, j) << " ";
    std::cout <<" \n";
  }

  std::cout << "data = ";
  for (unsigned int i = 0; i != sd; ++i)
    std::cout << eval_values.data()[i] << " ";
  std::cout <<" \n";

  // Transpose for vector values
  eval_values.transposeInPlace();

  // Copy for affine mapping - need to add Piola transform for other elements
  std::copy(eval_values.data(), eval_values.data() + sd, w);

  if (family == "Raviart-Thomas")
  {
    Eigen::Matrix2d J, K;
    double det;
    Eigen::Map<const Eigen::Matrix<double, 3, 2>> _coordinate_dofs(coordinate_dofs);
    compute_JK_triangle_2d(J, K, det, _coordinate_dofs);

    std::cout << "K = \n" << K << "\n";

    std::cout << "K*values = \n" << K*eval_values*det << "\n";


  };


  // FIXME: add transforms here - maybe do in generated code?
}
//-----------------------------------------------------------------------------
void Expression::compute_vertex_values(std::vector<double>& vertex_values,
                                       const Mesh& mesh) const
{
  // Local data for vertex values
  const std::size_t size = value_size();
  Eigen::VectorXd local_vertex_values(size);

  // Resize vertex_values
  vertex_values.resize(size * mesh.num_vertices());

  // Iterate over cells, overwriting values when repeatedly visiting vertices
  for (auto &cell : MeshRange<Cell>(mesh, MeshRangeType::ALL))
  {
    // Iterate over cell vertices
    for (auto &vertex : EntityRange<Vertex>(cell))
    {
      // Wrap coordinate data
      Eigen::Map<const Eigen::VectorXd> x(vertex.x(), mesh.geometry().dim());

      // Evaluate at vertex
      eval(local_vertex_values, x);

      // Copy to array
      for (std::size_t i = 0; i < size; i++)
      {
        const std::size_t global_index
            = i * mesh.num_vertices() + vertex.index();
        vertex_values[global_index] = local_vertex_values[i];
      }
    }
  }
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FunctionSpace> Expression::function_space() const
{
  return std::shared_ptr<const FunctionSpace>();
}
//-----------------------------------------------------------------------------
