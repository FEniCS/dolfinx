// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class TestProblem9 : public ODE
{
public:
  
  TestProblem9() : ODE(3)
  {
    dolfin_info("A mixed stiff/nonstiff test problem.");

    // Parameters
    lambda = 1000.0;

    // Final time
    T = 30.0;

    // Compute sparsity
    sparse();
  }

  real u0(unsigned int i)
  {
    if ( i == 0 )
      return 0.0;

    if ( i == 1 )
      return 1.0;

    return 1.0;
  }

  real f(const Vector& u, real t, unsigned int i)
  {
    if ( i == 0 )
      return u(1);

    if ( i == 1 )
      return -(1.0-u(2))*u(0);

    return -lambda * (u(1)*u(1) + u(2)*u(2)) * u(2);
  }

  Element::Type method(unsigned int i)
  {
    if ( i == 2 )
      return Element::dg;
    
    return Element::cg;
  }

  unsigned int order(unsigned int i)
  {
    if ( i == 2 )
      return 0;
    
    return 1;
  }

private:

  real lambda;

};
