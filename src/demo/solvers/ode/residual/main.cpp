// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2005-11-03

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

  dolfin_set("method", method.c_str());
  dolfin_set("solver", "fixed-point");
  dolfin_set("order", 1);

  real tol = 1.0e-4;

  if(method == "cg")
    dolfin_set("tolerance", tol);
  else
    dolfin_set("tolerance", tol);
  //   dolfin_set("discrete tolerance factor", 1.0e-3);

//   dolfin_set("fixed time step", true);
//   dolfin_set("partitioning threshold", 1e-7);
//   dolfin_set("partitioning threshold", 1.0 - 1e-7);
  dolfin_set("partitioning threshold", 0.999);
//   dolfin_set("interval threshold", 0.9);
  dolfin_set("interval threshold", 0.9);

  dolfin_set("initial time step", 5.0e-1);
  dolfin_set("maximum time step", 5.0e-1);

  dolfin_set("file name", filename.c_str());
  dolfin_set("save solution", true);
  dolfin_set("number of samples", 100);

  Harmonic ode;
  ode.solve();
  

  return 0;
}
