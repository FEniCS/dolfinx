// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_PDE_H
#define __NEW_PDE_H

namespace dolfin
{

  class BilinearForm;
  class LinearForm;

  /// A NewPDE represents a (linearized) partial differential equation,
  /// given by a variation problem of the form: Find u in V such that
  ///
  ///     a(u,v) = L(v) for all v in V,
  ///
  /// where a(.,.) is a given bilinear form and L(.) is a given linear form.

  class NewPDE
  {
  public:

    /// Constructor
    NewPDE();

    /// Constructor
    NewPDE(BilinearForm& a, LinearForm& L);

    /// Destructor
    ~NewPDE();

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
