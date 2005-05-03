// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __PDE_H
#define __PDE_H

namespace dolfin
{

  class BilinearForm;
  class LinearForm;

  /// A PDE represents a (linearized) partial differential equation,
  /// given by a variation problem of the form: Find u in V such that
  ///
  ///     a(u,v) = L(v) for all v in V,
  ///
  /// where a(.,.) is a given bilinear form and L(.) is a given linear form.

  class PDE
  {
  public:

    /// Constructor
    PDE();

    /// Constructor
    PDE(BilinearForm& a, LinearForm& L);

    /// Destructor
    ~PDE();

    /// Return the bilinear form a(.,.)
    BilinearForm& a();

    /// Return the linear form L(.,.)
    LinearForm& L();

  protected:

    BilinearForm* bilinear;
    LinearForm* linear;

  };

}

#endif
