// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-28
// Last changed: 2007-04-12

#ifndef __GENERIC_FUNCTION_H
#define __GENERIC_FUNCTION_H

#include <ufc.h>
#include <dolfin/constants.h>

namespace dolfin
{

  class Mesh;
  
  /// This class serves as a base class/interface for implementations
  /// of specific function representations.

  class GenericFunction
  {
  public:

    /// Constructor
    GenericFunction(Mesh& mesh) : mesh(mesh) {};

    /// Destructor
    virtual ~GenericFunction() {};

    /// Return the rank of the value space
    virtual uint rank() const = 0;

    /// Return the dimension of the value space for axis i
    virtual uint dim(uint i) const = 0;

    /// Interpolate function to vertices of mesh
    virtual void interpolate(real* values) = 0;

    /// Interpolate function to finite element space on cell
    virtual void interpolate(real* coefficients,
                             const ufc::cell& cell,
                             const ufc::finite_element& finite_element) = 0;

    /// The mesh
    Mesh& mesh;

  };

}

#endif
