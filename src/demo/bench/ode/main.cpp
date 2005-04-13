// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson, 2005.
// Modified by Anders Logg, 2005.

#include <stdlib.h>
#include <dolfin.h>

//#define DEBUG_BENCHMARK 1

using namespace dolfin;

class WaveEquation : public ODE
{
public:

  WaveEquation(unsigned int n) : ODE(2*(n+1)*(n+1)), 
				 n(n), offset(N/2),
				 num_f(0), num_fi(0)
  {
    T = 1.0;
    c = 0.5;

    h = 1.0 / static_cast<real>(n);
    a = c*c / (h*h);
    w = 20 * h;

    setSparsity();

#ifdef DEBUG_BENCHMARK
    mesh = new UnitSquare(n, n);
    ufile = new File("solutionu.m");
    vfile = new File("solutionv.m");
    kfile = new File("timesteps.m");
#endif
  }

  ~WaveEquation()
  {
    cout << "Number of mono-adaptive evaluations of f:  " << num_f << endl;
    cout << "Number of multi-adaptive evaluations of f: " << num_fi/N << endl;

#ifdef DEBUG_BENCHMARK
    delete mesh;
    delete ufile;
    delete vfile;
    delete kfile;
#endif
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
      const unsigned int m = n + 1;
      const unsigned int jx = j % m;
      const unsigned int jy = j / m;

      unsigned int size = 1;
      if ( jx > 0 ) size++;
      if ( jy > 0 ) size++;
      if ( jx < n ) size++;
      if ( jy < n ) size++;
      dependencies.setsize(i, size);

      dependencies.set(i, j);
      if ( jx > 0 ) dependencies.set(i, j - 1);
      if ( jy > 0 ) dependencies.set(i, j - m);
      if ( jx < n ) dependencies.set(i, j + 1);
      if ( jy < n ) dependencies.set(i, j + m);
    }
  }

  // Initial data
  real u0(unsigned int i)
  {
    unsigned int j = i;
    if ( i >= offset )
      j -= offset;
    const unsigned int m = n + 1;
    const unsigned int jx = j % m;
    const unsigned int jy = j / m;
    const Point p(h * static_cast<real>(jx), h * static_cast<real>(jy));
    const Point center(0.5, 0.5);

    const real dist = p.dist(center);

    if ( dist >= w / 2 )
      return 0.0;

    if ( i < offset )
    {
      return 0.5 * (cos(2.0 * M_PI * dist / w) + 1);
    }
    else
    {
      return c * M_PI / w * (sin(2.0 * M_PI * dist / w));
    }
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
      const unsigned int m = n + 1;
      const unsigned int jx = j % m;
      const unsigned int jy = j / m;

      real sum = -4.0*u[j];
      if ( jx > 0 ) sum += u[j - 1];
      if ( jy > 0 ) sum += u[j - m];
      if ( jx < n ) sum += u[j + 1];
      if ( jy < n ) sum += u[j + m];
      
      y[i] = a*sum;
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
    const unsigned int m = n + 1;
    const unsigned int jx = j % m;
    const unsigned int jy = j / m;

    real sum = -4.0*u[j];
    if ( jx > 0 ) sum += u[j - 1];
    if ( jy > 0 ) sum += u[j - m];
    if ( jx < n ) sum += u[j + 1];
    if ( jy < n ) sum += u[j + m];

    return a*sum;
  }

#ifdef DEBUG_BENCHMARK
  // Save solution  
  void save(NewSample& sample)
  {
    cout << "Saving data at t = " << sample.t() << endl;

    // Create vectors
    static Vector ux(N/2);
    static Vector vx(N/2);
    static Vector kx(N/2);
    static NewFunction u(*mesh, ux);
    static NewFunction v(*mesh, vx);
    static NewFunction k(*mesh, kx);
    u.rename("u", "Solution of the wave equation");
    v.rename("v", "Speed of the wave equation");
    k.rename("k", "Time steps for the wave equation");

    // Get the degrees of freedom and set current time
    u.set(sample.t());
    v.set(sample.t());
    k.set(sample.t());
    for (unsigned int i = 0; i < N/2; i++)
    {
      ux(i) = sample.u(i);
      vx(i) = sample.u(i + offset);
      kx(i) = sample.k(i);
    }

    // Save solution to file
    *ufile << u;
    *vfile << v;
    *kfile << k;
  }
#endif

private:

  real c; // Speed of light
  real h; // Mesh size
  real a; // Product (c/h)^2
  real w; // Width of initial data

  unsigned int n;      // Number of cells in each direction
  unsigned int offset; // Offset for second half of system

  unsigned int num_f;  // Number of evaluations of mono-adaptive f
  unsigned int num_fi; // Number of evaluations of multi-adaptive f

#ifdef DEBUG_BENCHMARK
  UnitSquare* mesh;            // The mesh
  File *ufile, *vfile, *kfile; // Files for saving solution
#endif

};

int main(int argc, const char* argv[])
{
  // Parse command line arguments
  if ( argc != 3 )
  {
    dolfin_info("Usage: dolfin-bench-ode method n");
    dolfin_info("");
    dolfin_info("method - 'cg', 'dg', 'mcg' or 'mdg'");
    dolfin_info("n      - number of cells in each dimension");
    return 1;
  }
  const char* method = argv[1];
  unsigned int n = static_cast<unsigned int>(atoi(argv[2]));
  if ( n < 1 )
    dolfin_error("Number of cells n must be positive.");

  // Set parameters
  dolfin_set("solve dual problem", false);
  dolfin_set("use new ode solver", true);
  //dolfin_set("solver", "newton");
  dolfin_set("method", method);
  dolfin_set("fixed time step", true);
  dolfin_set("save solution", false);
  dolfin_set("initial time step", 1e-2);
  dolfin_set("number of samples", 20);
  dolfin_set("discrete tolerance", 1e-7);
  dolfin_set("tolerance", 1e-5);

#ifdef DEBUG_BENCHMARK
  dolfin_set("save solution", true);
#endif

  // Solve the wave equation
  WaveEquation wave(n);
  wave.solve();
}
