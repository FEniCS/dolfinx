// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2008.
// Modified by Kristian B. Oelgaard, 2007.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2003-11-28
// Last changed: 2008-10-09

#ifndef __NEW_FUNCTION_H
#define __NEW_FUNCTION_H

#include <tr1/memory>
#include <dolfin/common/simple_array.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class FunctionSpace;
  class GenericVector;

  /// This class represents a function u_h in a finite element
  /// function space V_h, given by
  ///
  ///   u_h = sum_i U_i phi_i
  ///
  /// where {phi_i}_i is a basis for V_h, and U is a vector of
  /// expansion coefficients for u_h.

  class NewFunction : public Variable
  {
  public:

    /// Create function on given function space
    explicit NewFunction(const FunctionSpace& V);

    /// Create function on given function space (may be shared)
    explicit NewFunction(const std::tr1::shared_ptr<FunctionSpace> V);

    /// Create function from file
    explicit NewFunction(const std::string filename);

    /// Copy constructor
    NewFunction(const NewFunction& v);

    /// Destructor
    virtual ~NewFunction();

    /// Assignment operator
    const NewFunction& operator= (const NewFunction& v);

    /// Return the function space
    const FunctionSpace& function_space() const;

    /// Return the vector of expansion coefficients
    GenericVector& vector();

    /// Return the vector of expansion coefficients (const version)
    const GenericVector& vector() const;

    /// Check if function is a member of the given function space
    bool in(const FunctionSpace& V) const;

    /// Return the current time
    double time() const;

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

    /// Interpolate function to local finite element space
    void interpolate(double* coefficients, Cell& cell);

    /// Interpolate function to given global finite element space
    void interpolate(GenericVector& coefficients, FunctionSpace& V);

    /// Interpolate function to vertices of mesh
    void interpolate(double* vertex_values);

    /* FIXME: Functions below should be added somehow
    
    /// Create discrete function from sub function
    explicit Function(SubFunction sub_function);
    
    /// Return the rank of the value space
    virtual uint rank() const;

    /// Return the dimension of the value space for axis i
    virtual uint dim(uint i) const;

    /// Return the signature of a DiscreteFunction
    std::string signature() const;

    /// Return the number of sub functions (only for discrete functions)
    uint numSubFunctions() const;
    
    /// Extract sub function/slice (only for discrete function)
    SubFunction operator[] (uint i);

    /// Assign sub function/slice
    const Function& operator= (SubFunction f);
    
    /// Make current cell and facet available to user-defined function
    void update(Cell& cell, int facet=-1);

    */

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

    // The current cell (if any, otherwise 0)
    Cell* _cell;

    // The current facet (if any, otherwise -1)
    int _facet;

    // The current time
    double _time;

  };

}

#endif
