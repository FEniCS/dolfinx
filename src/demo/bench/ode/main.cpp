// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdlib.h>
#include <dolfin.h>

using namespace dolfin;

class WaveEquation : public ODE
{
public:

  WaveEquation(unsigned int n) : ODE(2*(n+1)*(n+1)*(n+1)), 
				 n(n), offset(N/2), mesh(n, n, n),
				 ufile("solution.dx"), kfile("timesteps.dx")
  {
    T = 1.0;
    c = 1.0;

    h = 1.0 / static_cast<real>(n + 1);
    a = c*c / (h*h);
    offset = N/2;

    // FIXME: Need to compute sparsity here
  }

  ~WaveEquation() {}

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
    
    // Second half of system
    const unsigned int j = i - offset;
    const unsigned int m = n + 1;
    const unsigned int jx = j % m;
    const unsigned int jy = (j / m) % m;
    const unsigned int jz = j / (m*m);

    real sum = -6.0*u[j];
    if ( jx > 0 ) sum += u[j - 1];
    if ( jy > 0 ) sum += u[j - m];
    if ( jz > 0 ) sum += u[j - m*m];
    if ( jx < n ) sum += u[j + 1];
    if ( jy < n ) sum += u[j + m];
    if ( jz < n ) sum += u[j + m*m];

    return a*sum;
  }

  // Right-hand side, mono-adaptive version
  void f(const real u[], real t, real y[])
  {
    // First half of system
    for (unsigned int i = 0; i < offset; i++)
      y[i] = y[i + offset];

    // Second half of system
    for (unsigned int i = offset; i < N; i++)
    {
      const unsigned int j = i - offset;
      const unsigned int m = n + 1;
      const unsigned int jx = j % m;
      const unsigned int jy = (j / m) % m;
      const unsigned int jz = j / (m*m);

      real sum = -6.0*u[j];
      if ( jx > 0 ) sum += u[j - 1];
      if ( jy > 0 ) sum += u[j - m];
      if ( jz > 0 ) sum += u[j - m*m];
      if ( jx < n ) sum += u[j + 1];
      if ( jy < n ) sum += u[j + m];
      if ( jz < n ) sum += u[j + m*m];
      
      y[i] = sum;
    }
  }

  // Save solution  
  void save(NewSample& sample)
  {
    // FIXME: Don't save solution when running benchmark

    // Create vectors
    NewVector ux(N/2);
    NewVector kx(N/2);
    NewFunction u(mesh, ux);
    NewFunction k(mesh, kx);
    u.rename("u", "Solution of the wave equation");
    k.rename("k", "Time steps for the wave equation");

    // Get the degrees of freedom and set current time
    u.set(sample.t());
    k.set(sample.t());
    for (unsigned int i = 0; i < N/2; i++)
    {
      ux(i) = sample.u(i);
      kx(i) = sample.k(i);
    }

    // Save solution to file
    ufile << u;
    kfile << k;
  }

private:

  real c; // Speed of light
  real h; // Mesh size
  real a; // Product (c/h)^2

  unsigned int n;      // Number of cells in each direction
  unsigned int offset; // Offset for second half of system
  UnitCube mesh;       // The mesh
  File ufile, kfile;   // Files for saving solution

};

int main(int argc, const char* argv[])
{
  // Parse command line arguments
  if ( argc != 2 )
  {
    dolfin_info("Usage: dolfin-bench-ode n");
    // FIXME: Should be dolfin-bench-ode method n with method = cg, dg, mcg or mdg
    return 1;
  }
  unsigned int n = static_cast<unsigned int>(atoi(argv[1]));

  // FIXME: Parse method here

  // Set parameters
  dolfin_set("solve dual problem", false);
  dolfin_set("use new ode solver", true);
  //dolfin_set("method", "mcg");

  // Solve the wave equation
  WaveEquation wave(n);
  wave.solve();
}
