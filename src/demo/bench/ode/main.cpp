// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdlib.h>
#include <dolfin.h>

using namespace dolfin;

class WaveEquation : public ODE
{
public:

  WaveEquation(unsigned int n) : ODE((n+1)*(n+1)*(n+1))
  {


  }

  ~WaveEquation()
  {



  }

  real u0(unsigned int i)
  {
    return 0.0;
  }

private:

  
};

int main(int argc, const char* argv[])
{
  // Parse command line arguments
  if ( argc != 2 )
  {
    dolfin_info("Usage: dolfin-bench-ode n");
    return 1;
  }
  unsigned int n = static_cast<unsigned int>(atoi(argv[1]));

  // Solve the wave equation
  WaveEquation wave(n);
  wave.solve();
}
