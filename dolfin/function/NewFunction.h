// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2008.
// Modified by Kristian B. Oelgaard, 2007.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2003-11-28
// Last changed: 2008-09-25

#ifndef __NEW_FUNCTION_H
#define __NEW_FUNCTION_H

#include <tr1/memory>
#include <dolfin/common/simple_array.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class FunctionSpace;
  class GenericVector;

  /// This class represents a function v in a finite element
  /// function space V,
  ///
  ///   v = sum_i x_i phi_i
  ///
  /// where {phi_i}_i is a basis for V, and x is a vector of
  /// degrees of freedom (dofs) for v.

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

    /// Return the vector of degrees of freedom (non-const version)
    GenericVector& vector();

    /// Return the vector of degrees of freedom (const version)
    const GenericVector& vector() const;

    /// Evaluate function at given point p (overload for user-defined function)
    virtual void eval(double* values, const double* p) const;

    /// Evaluate function at given point p (overload for scalar user-defined function)
    virtual double eval(const double* p) const;

    /// Evaluate function at given point (used for subclassing through SWIG interface)
    void eval(simple_array<double>& values, const simple_array<double>& p) const;

  private:

    // Initialize vector
    void init();

    // The function space
    const std::tr1::shared_ptr<const FunctionSpace> V;

    // The vector of degrees of freedom
    GenericVector* x;

  };

}

#endif
