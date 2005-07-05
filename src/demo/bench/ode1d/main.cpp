// Copyright (C) 2005 Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2005

#include <stdio.h>
#include <stdlib.h>
#include <dolfin.h>

#define DEBUG_BENCHMARK 1
//#define COMPUTE_REFERENCE 1

using namespace dolfin;

class WaveEquation : public ODE
{
public:

  WaveEquation(unsigned int n) : ODE(2*(n+1)), 
				 n(n), offset(N/2),
				 num_f(0), num_fi(0)
  {
    T = 1.0;
    c = 0.5;

    h = 1.0e-3;
    a = 1.0 / (h * h);
    w = 20 * h;

    nu = 1.0 * 3e-5;

    setSparsity();
  }

  ~WaveEquation()
  {
    cout << "Number of mono-adaptive evaluations of f:  " << num_f << endl;
    cout << "Number of multi-adaptive evaluations of f: " << num_fi/N << endl;

  }

  void setSparsity()
  {
    // Dependencies for first half of system
    for (unsigned int i = 0; i < offset; i++)
    {
      //dependencies.clear(i);
      dependencies.setsize(i, 1);
      dependencies.set(i, i + offset);
    }

    // Dependencies for second half of system
    for (unsigned int i = offset; i < N; i++)
    {
      const unsigned int j = i - offset;

      unsigned int size = 2;
      if ( j > 0 ) size += 2;
      if ( j < n ) size += 2;
      dependencies.setsize(i, size);

      dependencies.set(i, j);
      if ( j > 0 ) dependencies.set(i, j - 1);
      if ( j < n ) dependencies.set(i, j + 1);
      if ( j > 0 ) dependencies.set(i, i - 1);
      if ( j < n ) dependencies.set(i, i + 1);
    }
  }

  // Initial data
  real u0(unsigned int i)
  {
    unsigned int j = i;
    if ( i >= offset )
      j -= offset;
    const real p = h * j;
    const real center = 0.5 * h * n;
    const real dist = fabs(p - center);

    if ( dist >= w / 2 )
      return 0.0;
    
    if ( i < offset )
      return 1.0 * 0.5 * (cos(2.0 * M_PI * dist / w) + 1);
    else
      return 1.0 * c * M_PI / w * (sin(2.0 * M_PI * dist / w));
  }

  // Right-hand side, mono-adaptive version
  void f(const real u[], real t, real y[])
  {
    num_f++;

    // First half of system
    for (unsigned int i = 0; i < offset; i++)
      y[i] = u[i + offset];

    // Second half of system
    for (unsigned int i = offset; i < N; i++)
    {
      const unsigned int j = i - offset;

      real sum0 = -2.0*u[j];
      if ( j > 0 ) sum0 += u[j - 1];
      if ( j < n ) sum0 += u[j + 1];

      real sum1 = -2.0*u[i];
      if ( j > 0 ) sum1 += u[i - 1];
      if ( j > 0 ) sum1 += u[i + 1];
      
      y[i] = a*(c*c*sum0 + nu*sum1);
    }
  }

  // Right-hand side, multi-adaptive version
  real f(const real u[], real t, unsigned int i)
  {
    num_fi++;

    // First half of system
    if ( i < offset )
      return u[i + offset];
    
    // Second half of system
    const unsigned int j = i - offset;

    real sum0 = -2.0*u[j];
    if ( j > 0 ) sum0 += u[j - 1];
    if ( j < n ) sum0 += u[j + 1];
    
    real sum1 = -2.0*u[i];
    if ( j > 0 ) sum1 += u[i - 1];
    if ( j < n ) sum1 += u[i + 1];
    
    return a*(c*c*sum0 + nu*sum1);
  }

#ifdef DEBUG_BENCHMARK
  // Save solution  
  void save(Sample& sample)
  {
  }  
#endif

#ifdef COMPUTE_REFERENCE
  virtual bool update(const real u[], real t, bool end)
  {
    if ( !end )
      return true;

    cout << "Saving reference solution" << endl;

    // Save solution at end-time
    FILE* fp = fopen("solution.data", "w");
    for (unsigned int i = 0; i < N; i++)
    {
      fprintf(fp, "%.15e ", u[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);

    return true;
  }
#endif

private:

  real c; // Speed of light
  real h; // Mesh size
  real a; // Product (c/h)^2
  real w; // Width of initial data
  real nu;// Viscosity

  unsigned int n;      // Number of cells in each direction
  unsigned int offset; // Offset for second half of system

  unsigned int num_f;  // Number of evaluations of mono-adaptive f
  unsigned int num_fi; // Number of evaluations of multi-adaptive f
};

int main(int argc, const char* argv[])
{
  // Parse command line arguments
  if ( argc != 6 )
  {
    dolfin_info("Usage: dolfin-bench-ode method order n k TOL");
    dolfin_info("");
    dolfin_info("method - 'cg', 'dg', 'mcg' or 'mdg'");
    dolfin_info("order  - q, as in cG(q) or dG(q)");
    dolfin_info("n      - number of cells in each dimension");
    dolfin_info("k      - initial time step");
    dolfin_info("TOL    - tolerance");
    return 1;
  }
  const char* method = argv[1];
  const unsigned int q = static_cast<unsigned int>(atoi(argv[2]));
  if ( q < 0 )
    dolfin_error("The order must be positive.");
  const unsigned int n = static_cast<unsigned int>(atoi(argv[3]));
  if ( n < 1 )
    dolfin_error("Number of cells n must be positive.");
  const real k = static_cast<real>(atof(argv[4]));
  if ( k <= 0.0 )
    dolfin_error("Time step must be positive.");
  const real TOL = static_cast<real>(atof(argv[5]));
  if ( TOL <= 0.0 )
    dolfin_error("Tolerance must be positive.");

  // Common parameters
  dolfin_set("method", method);
  dolfin_set("order", q);
  dolfin_set("initial time step", k);
  dolfin_set("solve dual problem", false);
  dolfin_set("save solution", false);
  dolfin_set("monitor convergence", false);

#ifdef COMPUTE_REFERENCE
  // Parameters for reference solution
  dolfin_set("discrete tolerance", 1e-12);
  dolfin_set("fixed time step", true);
  dolfin_set("maximum iterations", 200);
  dolfin_set("progress step", 0.01);
#else
  // Parameters for benchmarks
  dolfin_set("tolerance", TOL);
  dolfin_set("maximum time step", 1.0e-3);
  dolfin_set("partitioning threshold", 0.25);
  
  //dolfin_set("fixed time step", true);
  //dolfin_set("time step conservation", 5.0);
  //dolfin_set("partitioning threshold", 0.5);
#endif
  
#ifdef DEBUG_BENCHMARK
  // Parameters for debug (save solution)
  dolfin_set("number of samples", 10);
  dolfin_set("save solution", true);
  dolfin_set("progress step", 0.01);
  dolfin_set("monitor convergence", true);
#endif

  // Solve the wave equation
  WaveEquation wave(n);
  wave.solve();
}
