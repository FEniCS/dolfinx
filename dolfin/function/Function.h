// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2009.
// Modified by Kristian B. Oelgaard, 2007.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2003-11-28
// Last changed: 2009-01-07

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <ufc.h>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/Variable.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  class FunctionSpace;
  class GenericVector;
  class Data;
  class SubFunction;

  /// This class represents a function u_h in a finite element
  /// function space V_h, given by
  ///
  ///   u_h = sum_i U_i phi_i
  ///
  /// where {phi_i}_i is a basis for V_h, and U is a vector of
  /// expansion coefficients for u_h.

  class Function : public Variable
  {
  public:

    /// Create function (and let DOLFIN figure out the correct function space)
    Function();

    /// Create function on given function space
    explicit Function(const FunctionSpace& V);

    /// Create function on given function space with a given vector
    Function(const FunctionSpace& V, GenericVector& x);

    /// Create function on given function space (shared data)
    explicit Function(boost::shared_ptr<const FunctionSpace> V);

    /// Create function on given function space with a given vector (shared data)
    Function(boost::shared_ptr<const FunctionSpace> V, boost::shared_ptr<GenericVector> x);

    /// Create function from file
    explicit Function(std::string filename);

    /// Create function from sub function
    Function(const SubFunction& v);

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

    /// Return the function space
    boost::shared_ptr<const FunctionSpace> function_space_ptr() const;

    /// Return the vector of expansion coefficients (non-const version)
    GenericVector& vector();

    /// Return the vector of expansion coefficients (const version)
    const GenericVector& vector() const;

    /// Test for the function space
    bool has_function_space() const;

    /// Test for a vector of expansion coefficients
    bool has_vector() const;

    /// Check if function is a member of the given function space
    bool in(const FunctionSpace& V) const;

    /// Return geometric dimension
    uint geometric_dimension() const;
    
    /// Function evaluation (overload for user-defined function, simple version)
    virtual void eval(double* values, const double* x) const;

    /// Function evaluation (overload for user-defined function, alternate version)
    virtual void eval(double* values, const Data& data) const;

    /// Evaluate function v at given point in given cell
    void eval(double* values, const double* x, const ufc::cell& ufc_cell, uint cell_index) const;

    /// Interpolate function to local function space on cell
    void interpolate(double* coefficients, const ufc::cell& ufc_cell, uint cell_index, int local_facet=-1) const;

    /// Interpolate function to local function space on cell with check on function space
    void interpolate(double* coefficients, const FunctionSpace& V, const ufc::cell& ufc_cell, uint cell_index, int local_facet=-1) const;

    /// Interpolate function to given function space
    void interpolate(GenericVector& coefficients, const FunctionSpace& V) const;

    /// Interpolate function to vertices of mesh
    void interpolate(double* vertex_values) const;

    /// Friends
    friend class Coefficient;
    friend class LinearPDE;
    friend class NonlinearPDE;
    friend class VariationalProblem;

  private:

    // Initialize vector
    void init();

    // The function space
    boost::shared_ptr<const FunctionSpace> _function_space;

    // The vector of expansion coefficients
    boost::shared_ptr<GenericVector> _vector;

  };

}

#endif
