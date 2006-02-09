// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.
//
// First added:  2005
// Last changed: 2005-12-20

#include <string>

#include <stdio.h>
#include <dolfin.h>

using namespace dolfin;

class Harmonic : public ODE
{
public:
  
  Harmonic() : ODE(2, 0.4)
  {
    // Compute sparsity
//     sparse();
  }

  real u0(unsigned int i)
  {
    if ( i == 0 )
      return 0.0;

    return 0.0;
  }

  real f(const real u[], real t, unsigned int i)
  {
    if ( i == 0 )
      return pow(t, 3);

    return 0.0;
  }

  /// Specify time step, mono-adaptive version (used for testing)
  real timestep(real t, real k0) const
  {
    return 5e-2;
  }

  /// Specify time step, mono-adaptive version (used for testing)
  real timestep(real t, unsigned int i, real k0) const
  {
    if(i == 0)
    {
      return 4e-2;
    }
    else if(i == 1)
    {
      return 1e-1;
    }
    else
    {
      return 1e-1;
    }
  }

  bool update(const real u[], real t, bool end)
  {
    return true;
  }
  
};

int main(int argc, char* argv[])
{
  std::string method = argv[1];

  std::string filename = std::string("primal") + method +
    std::string(".py");

  cout << "filename: " << filename << endl;

  set("method", method.c_str());
  set("solver", "fixed-point");
  set("order", 1);

  real tol = 1.0e-4;

  if(method == "cg")
    set("tolerance", tol);
  else
    set("tolerance", tol);
  //   set("discrete tolerance factor", 1.0e-3);

//   set("fixed time step", true);
//   set("partitioning threshold", 1e-7);
//   set("partitioning threshold", 1.0 - 1e-7);
  set("partitioning threshold", 0.999);
//   set("interval threshold", 0.9);
  set("interval threshold", 0.9);

  set("initial time step", 5.0e-1);
  set("maximum time step", 5.0e-1);

  set("ode solution file name", filename.c_str());
  set("save solution", true);
  set("number of samples", 100);

  Harmonic ode;
  ode.solve();
  

  return 0;
}
