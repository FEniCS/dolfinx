// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdlib.h>
#include <dolfin.h>

using namespace dolfin;

class WaveEquation : public ODE
{
public:

  WaveEquation(unsigned int n) : ODE(2*(n+1)*(n+1)*(n+1)), n(n), mesh(n, n, n)
  {
    T = 1.0;
    c = 1.0;

    h = 1.0 / static_cast<real>(n + 1);
    a = c*c / (h*h);
    offset = N/2;
  }

  ~WaveEquation()
  {
    


  }

  // Initial data
  real u0(unsigned int i)
  {
    if ( i < offset )
      if ( mesh.node(i).dist(0.5, 0.5 , 0.5) < h )
	return 1.0;
    
    return 0.0;
  }

  // Right-hand side, multi-adaptive version
  real f(const real u[], real t, unsigned int i)
  {
    // First half of system
    if ( i < offset )
      return u[i + offset];

    return 0.0;
  }

  // Right-hand side, mono-adaptive version
  void f(const real u[], real t, real y[])
  {
    // First half of system
    for (unsigned int i = 0; i < offset; i++)
      y[i] = y[i + offset];
  }

  // Save solution  
  void save(Sample& sample)
  {




  }

private:

  real c; // Speed of light
  real h; // Mesh size
  real a; // Product (c/h)^2

  unsigned int n;      // Number of cells in each direction
  unsigned int offset; // Offset for second half of system
  UnitCube mesh;       // The mesh

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

  // Set parameters
  dolfin_set("use new ode solver", true);

  // Solve the wave equation
  WaveEquation wave(n);
  wave.solve();
}
