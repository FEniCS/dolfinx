// Copyright (C) 2003 Johan Jansson.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include <dolfin/Partition.h>
#include <dolfin/TimeSlab.h>

using namespace dolfin;

class Minimal : public ODE {
public:

  Minimal() : ODE(1)
  {
    // Parameters
    lambda1 = 1.0;
    lambda2 = 2.0;

    // Final time
    T = 10.0;

    // Initial value
    u0(0) = 1.0;
    //u0(1) = 1.0;

    // Sparsity (not really necessary here)
    sparse();
  }

  real f(const Vector& u, real t, int i)
  {
    dolfin_debug("foo");
    dolfin::cout << "u: " << dolfin::endl;
    u.show();

    if(i == 0)
    {
      return -lambda1 * u(0);
    }
    else if(i == 1)
    {
      return -lambda2 * u(1);
    }
    return 0;
  }

private:
  real lambda1, lambda2;
};

int main()
{
  dolfin_set("output", "plain text");

  Minimal minimal;

  real foo;
  Vector u(1);
  u(0) = 1;


  foo = minimal.f(u, 0, 0);

  dolfin::cout << foo << dolfin::endl;

  dolfin::cout << (int)sizeof(Element) << dolfin::endl;


  Partition p(minimal.size(), 0.1);
  TimeSlabData data(minimal);
  RHS f(minimal, data);
  TimeSlab(0, 0.1, f, data, p, 0);


  //minimal.solve();
  
  return 0;
}
