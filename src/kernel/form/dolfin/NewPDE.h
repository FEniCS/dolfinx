// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_PDE_H
#define __NEW_PDE_H

namespace dolfin
{

  class BilinearForm;
  class LinearForm;

  class NewPDE
  {
  public:

    /// Constructor
    NewPDE(BilinearForm& a, LinearForm& L);

    /// Destructor
    ~NewPDE();

    /// The bilinear form
    BilinearForm& a;

    /// The linear form
    LinearForm& L;

  };

}

#endif
