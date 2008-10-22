// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2008.
// Modified by Kristian B. Oelgaard, 2007.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2003-11-28
// Last changed: 2008-10-21

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <ufc.h>
#include <tr1/memory>
#include <dolfin/common/simple_array.h>
#include <dolfin/common/Variable.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  class Cell;
  class FiniteElement;
  class FunctionSpace;
  class GenericVector;
  class Point;
  class SubFunction;

  /// This class represents a function u_h in a finite element
  /// function space V_h, given by
  ///
  ///   u_h = sum_i U_i phi_i
  ///
  /// where {phi_i}_i is a basis for V_h, and U is a vector of
  /// expansion coefficients for u_h.

  class Function : public Variable, public ufc::function
  {
  public:

    /// Create an empty function
    Function() { warning("Creating function without a basis"); };

    /// Create function on given function space
    explicit Function(const FunctionSpace& V);

    /// Create function on given function space (shared data)
    explicit Function(const std::tr1::shared_ptr<FunctionSpace> V);

    /// Create function from file
    explicit Function(const std::string filename);

    /// Create function from sub function
    explicit Function(const SubFunction& v);

    /// Copy constructor
    Function(const Function& v);

    /// Destructor
    virtual ~Function();

    /// Assignment from function
    const Function& operator= (const Function& v);

    /// Extract sub function
    SubFunction operator[] (uint i);

    /// Return the function space
    const FunctionSpace& function_space() const;

    /// Return the finite element
    const FiniteElement& element() const;

    /// Return the vector of expansion coefficients
    GenericVector& vector();

    /// Return the vector of expansion coefficients (const version)
    const GenericVector& vector() const;

    /// Return the current time
    double time() const;

    /// Check if function is a member of the given function space
    bool in(const FunctionSpace& V) const;

    /// Evaluate function at point x (overload for user-defined function)
    virtual void eval(double* values, const double* x) const;

    /// Evaluate function at point x and time t (overload for user-defined function)
    virtual void eval(double* values, const double* x, double t) const;

    /// Evaluate function at point x (overload for scalar user-defined function)
    virtual double eval(const double* x) const;

    /// Evaluate function at point x and time t (overload for scalar user-defined function)
    virtual double eval(const double* x, double t) const;

    /// Evaluate function at given point (used for subclassing through SWIG interface)
    void eval(simple_array<double>& values, const simple_array<double>& x) const;

    /// Evaluate function (needed for ufc::function interface)
    void evaluate(double* values, const double* coordinates, const ufc::cell& cell) const;

    /// Interpolate function to given function space
    void interpolate(GenericVector& coefficients, const FunctionSpace& V) const;

    /// Interpolate function to local function space on cell
    void interpolate(double* coefficients, const ufc::cell& ufc_cell) const;

    /// Interpolate function to local function space on cell (with check on function space)
    void interpolate(double* coefficients, const FunctionSpace& V, const ufc::cell& ufc_cell) const;

    /// Interpolate function to vertices of mesh
    void interpolate(double* vertex_values) const;

  protected:

    /// Access current cell (available during assembly for user-defined function)
    const Cell& cell() const;

    /// Access current facet (available during assembly for user-defined function)
    uint facet() const;

    /// Access current facet normal (available during assembly for user-defined function)
    Point normal() const;

  private:

    // Initialize vector
    void init();

    // The function space
    const std::tr1::shared_ptr<const FunctionSpace> _function_space;

    // The vector of expansion coefficients
    GenericVector* _vector;

    // The current time
    double _time;

    // The current cell (if any, otherwise 0)
    Cell* _cell;

    // The current facet (if any, otherwise -1)
    int _facet;

  };

}

#endif
