// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_PDE_H
#define __NEW_PDE_H

#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>

namespace dolfin
{

  /// EXPERIMENTAL: Redesign of the evaluation of variational forms

  class NewPDE
  {
  public:

    /// Constructor
    NewPDE();

    /// Destructor
    virtual ~NewPDE();

    /// Evaluation of left-hand side
    virtual real lhs(Form::TrialFunction u, Form::TestFunction v);

    /// Evaluation of right-hand side
    virtual real rhs(Form::TestFunction v);

  private:

    BilinearForm a;
    LinearForm l;

  };

}

#endif
