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

  /// This class represents a function u_h on a finite element
  /// function space V,
  ///
  ///   u_h = sum_i U_i phi_i
  ///
  /// where {phi_i}_i is a basis for V, and U is a vector of
  /// degrees of freedom for u_h.

  class NewFunction : public Variable
  {
  public:

    /// Create function on given function space
    NewFunction(FunctionSpace& V);

    /// Create function on given function space (may be shared)
    NewFunction(std::tr1::shared_ptr<FunctionSpace> V);

    /// Create function from file
    explicit NewFunction(const std::string filename);
    
    /// Copy constructor
    NewFunction(const NewFunction& f);

    /// Destructor
    virtual ~NewFunction();

    /// Return the function space
    FunctionSpace& V();

    /// Return the function space (const version)
    const FunctionSpace& V() const;

    /// Return the vector of degrees of freedom
    GenericVector& U();

    /// Return the vector of degrees of freedom
    const GenericVector& U() const;

    /// Assignment operator
    const NewFunction& operator= (const NewFunction& v);
    
  private:
    
    // Initialize vector
    void init();

    // The function space
    std::tr1::shared_ptr<FunctionSpace> _V;

    // The vector of degrees of freedom
    std::tr1::shared_ptr<GenericVector> _U;

  };

}

#endif
