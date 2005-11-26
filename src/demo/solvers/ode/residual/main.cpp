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
  
  Harmonic() : ODE(3, 40.0)
  {
    // Compute sparsity
//     sparse();
  }

  real u0(unsigned int i)
  {
    if ( i == 0 )
      return 0.0;
    if ( i == 1 )
      return 1.0;

    return 0.0;
  }

  real f(const real u[], real t, unsigned int i)
  {
//     if ( i == 0 )
//       return u[1];

//     return -u[0];

    if ( i == 0 )
      return u[1];
    if ( i == 1 )
      return -u[0];

    return 0.0;
  }

  /// Specify time step, mono-adaptive version (used for testing)
  real timestep(real t, real k0) const
  {
//     real k_0 = 5e-5 + 5e-4 * fabs(pow(cos(t), 1));
//     real k_1 = 5e-5 + 5e-4 * fabs(pow(sin(t), 1));

//     return std::min(k_0, k_1);
    return 1e-3;
  }

  /// Specify time step, mono-adaptive version (used for testing)
  real timestep(real t, unsigned int i, real k0) const
  {
//     real k_0 = 5e-5 + 5e-4 * fabs(pow(cos(t + i), 1));
//     real k_1 = 5e-5 + 5e-4 * fabs(pow(sin(t + i), 1));

//     if(i == 0)
//     {
//       return std::min(1.5e-3, k_0);
//     }
//     else
//     {
//       return std::min(1.5e-3, k_1);
//     }

    if(i == 0)
    {
      return 1e-3;
    }
    else if(i == 1)
    {
      return 1e-3;
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

  real tol = 1.0e-8;

  if(method == "cg")
    dolfin_set("tolerance", tol);
  else
    dolfin_set("tolerance", tol);
  //   dolfin_set("discrete tolerance factor", 1.0e-3);

  dolfin_set("fixed time step", true);
//   dolfin_set("partitioning threshold", 1e-7);
  dolfin_set("partitioning threshold", 1.0 - 1e-7);
//   dolfin_set("partitioning threshold", 0.4);

  dolfin_set("initial time step", 1.0e-5);
  dolfin_set("maximum time step", 1.0e-1);

  dolfin_set("file name", filename.c_str());
  dolfin_set("save solution", true);

  Harmonic ode;
  ode.solve();
  

  return 0;
}
