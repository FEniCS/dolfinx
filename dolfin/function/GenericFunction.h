// Copyright (C) 2009 Anders Logg
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
// Modified by Garth N. Wells, 2009.
//
// First added:  2009-09-28
// Last changed: 2011-01-19

#ifndef __GENERIC_FUNCTION_H
#define __GENERIC_FUNCTION_H

#include <memory>
#include <ufc.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class Mesh;
  class Cell;
  class Point;
  class FiniteElement;
  class FunctionSpace;

  /// This is a common base class for functions. Functions can be
  /// evaluated at a given point and they can be restricted to a given
  /// cell in a finite element mesh. This functionality is implemented
  /// by sub-classes that implement the eval() and restrict() functions.
  ///
  /// DOLFIN provides two implementations of the GenericFunction
  /// interface in the form of the classes Function and Expression.
  ///
  /// Sub-classes may optionally implement the update() function that
  /// will be called prior to restriction when running in parallel.

  class GenericFunction : public ufc::function, public Variable
  {
  public:

    /// Constructor
    GenericFunction();

    /// Destructor
    virtual ~GenericFunction();

    //--- Functions that must be implemented by sub-classes ---

    /// Return value rank
    virtual std::size_t value_rank() const = 0;

    /// Return value dimension for given axis
    virtual std::size_t value_dimension(std::size_t i) const = 0;

    /// Evaluate at given point in given cell
    virtual void eval(Array<double>& values, const Array<double>& x,
                      const ufc::cell& cell) const;

    /// Evaluate at given point
    virtual void eval(Array<double>& values, const Array<double>& x) const;

    /// Restrict function to local cell (compute expansion coefficients w)
    virtual void restrict(double* w,
                          const FiniteElement& element,
                          const Cell& dolfin_cell,
                          const double* coordinate_dofs,
                          const ufc::cell& ufc_cell) const = 0;

    /// Compute values at all mesh vertices
    virtual void compute_vertex_values(std::vector<double>& vertex_values,
                                       const Mesh& mesh) const = 0;

    //--- Optional functions to be implemented by sub-classes ---

    /// Update off-process ghost coefficients
    virtual void update() const {}

    //--- Convenience functions ---

    /// Evaluation at given point (scalar function)
    double operator() (double x) const;

    /// Evaluation at given point (scalar function)
    double operator() (double x, double y) const;

    /// Evaluation at given point (scalar function)
    double operator() (double x, double y, double z) const;

    /// Evaluation at given point (scalar function)
    double operator() (const Point& p) const;

    /// Evaluation at given point (vector-valued function)
    void operator() (Array<double>& values, double x) const;

    /// Evaluation at given point (vector-valued function)
    void operator() (Array<double>& values, double x, double y) const;

    /// Evaluation at given point (vector-valued function)
    void operator() (Array<double>& values, double x, double y, double z) const;

    /// Evaluation at given point (vector-valued function)
    void operator() (Array<double>& values, const Point& p) const;

    /// Evaluation at given point

    /// Return value size (product of value dimensions)
    std::size_t value_size() const;

    //--- Implementation of ufc::function interface ---

    /// Evaluate function at given point in cell
    virtual void evaluate(double* values,
                          const double* coordinates,
                          const ufc::cell& cell) const;

    // Pointer to FunctionSpace, if appropriate, otherwise NULL
    virtual std::shared_ptr<const FunctionSpace> function_space() const = 0;

  protected:

    // Restrict as UFC function (by calling eval)
    void restrict_as_ufc_function(double* w,
                                  const FiniteElement& element,
                                  const Cell& dolfin_cell,
                                  const double* coordinate_dofs,
                                  const ufc::cell& ufc_cell) const;

  };

}

#endif
