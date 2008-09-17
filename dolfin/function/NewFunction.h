// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2008.
// Modified by Kristian B. Oelgaard, 2007.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2003-11-28
// Last changed: 2008-09-11

#ifndef __FUNCTION_H
#define __FUNCTION_H

#include <tr1/memory>
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
  /// degrees of freedom for v.

  class NewFunction : public Variable
  {
  public:

    /// Create function on given function space
    explicit NewFunction(FunctionSpace& V);

    /// Create function on given function space (may be shared)
    explicit NewFunction(std::tr1::shared_ptr<FunctionSpace> V);

    /// Create function from file
    explicit NewFunction(const std::string filename);
    
    /// Copy constructor
    NewFunction(const NewFunction& v);

    /// Destructor
    virtual ~NewFunction();

    /// Return the function space
    FunctionSpace& function_space();

    /// Return the function space (const version)
    const FunctionSpace& function_space() const;

    /// Return the vector of degrees of freedom
    GenericVector& vector();

    /// Return the vector of degrees of freedom
    const GenericVector& vector() const;

    /// Assignment operator
    const NewFunction& operator= (const NewFunction& v);

  private:

    // Initialize vector
    void init();

    // The function space
    std::tr1::shared_ptr<FunctionSpace> V;

    // The vector of degrees of freedom
    std::tr1::shared_ptr<GenericVector> x;

  };

}

#endif
