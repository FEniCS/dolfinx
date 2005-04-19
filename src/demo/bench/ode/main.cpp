// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson, 2005.
// Modified by Anders Logg, 2005.

#include <stdio.h>
#include <stdlib.h>
#include <dolfin.h>

//#define DEBUG_BENCHMARK 1
//#define COMPUTE_REFERENCE 1

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
    a = 1.0 / (h*h);
    w = 10 * h;

    nu = 0.0005;

    setSparsity();

#ifdef DEBUG_BENCHMARK
    mesh = new UnitSquare(n, n);
    ufile = new File("solutionu.m");
    vfile = new File("solutionv.m");
    kfile = new File("timesteps.m");
    rfile = new File("residual.m");
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
    delete rfile;
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

      unsigned int size = 2;
      if ( jx > 0 ) size += 2;
      if ( jy > 0 ) size += 2;
      if ( jx < n ) size += 2;
      if ( jy < n ) size += 2;
      dependencies.setsize(i, size);

      dependencies.set(i, j);
      if ( jx > 0 ) dependencies.set(i, j - 1);
      if ( jy > 0 ) dependencies.set(i, j - m);
      if ( jx < n ) dependencies.set(i, j + 1);
      if ( jy < n ) dependencies.set(i, j + m);
      if ( jx > 0 ) dependencies.set(i, i - 1);
      if ( jy > 0 ) dependencies.set(i, i - m);
      if ( jx < n ) dependencies.set(i, i + 1);
      if ( jy < n ) dependencies.set(i, i + m);
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
      const unsigned int m = n + 1;
      const unsigned int jx = j % m;
      const unsigned int jy = j / m;

      real sum0 = -4.0*u[j];
      if ( jx > 0 ) sum0 += u[j - 1];
      if ( jy > 0 ) sum0 += u[j - m];
      if ( jx < n ) sum0 += u[j + 1];
      if ( jy < n ) sum0 += u[j + m];

      real sum1 = -4.0*u[i];
      if ( jx > 0 ) sum1 += u[i - 1];
      if ( jy > 0 ) sum1 += u[i - m];
      if ( jx < n ) sum1 += u[i + 1];
      if ( jy < n ) sum1 += u[i + m];
      
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
    const unsigned int m = n + 1;
    const unsigned int jx = j % m;
    const unsigned int jy = j / m;

    real sum0 = -4.0*u[j];
    if ( jx > 0 ) sum0 += u[j - 1];
    if ( jy > 0 ) sum0 += u[j - m];
    if ( jx < n ) sum0 += u[j + 1];
    if ( jy < n ) sum0 += u[j + m];
    
    real sum1 = -4.0*u[i];
    if ( jx > 0 ) sum1 += u[i - 1];
    if ( jy > 0 ) sum1 += u[i - m];
    if ( jx < n ) sum1 += u[i + 1];
    if ( jy < n ) sum1 += u[i + m];
    
    return a*(c*c*sum0 + nu*sum1);
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
    static Vector rx(N/2);
    static NewFunction u(ux, *mesh);
    static NewFunction v(vx, *mesh);
    static NewFunction k(kx, *mesh);
    static NewFunction r(rx, *mesh);
    u.rename("u", "Solution of the wave equation");
    v.rename("v", "Speed of the wave equation");
    k.rename("k", "Time steps for the wave equation");
    r.rename("r", "Time residual for the wave equation");

    // Get the degrees of freedom and set current time
    u.set(sample.t());
    v.set(sample.t());
    k.set(sample.t());
    r.set(sample.t());
    for (unsigned int i = 0; i < N/2; i++)
    {
      ux(i) = sample.u(i);
      vx(i) = sample.u(i + offset);
      kx(i) = sample.k(i + offset);
      rx(i) = sample.r(i + offset);
    }

    // Save solution to file
    *ufile << u;
    *vfile << v;
    *kfile << k;
    *rfile << r;
  }  
#endif

  //#ifdef COMPUTE_REFERENCE
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
  //#endif

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

#ifdef DEBUG_BENCHMARK
  UnitSquare* mesh;                    // The mesh
  File *ufile, *vfile, *kfile, *rfile; // Files for saving solution
#endif

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
  dolfin_set("discrete tolerance", 1e-10);
  dolfin_set("maximum time step", 1.0 / static_cast<real>(n));
  
  dolfin_set("save solution", true);
  dolfin_set("adaptive samples", true);
  //dolfin_set("monitor convergence", true);
  //dolfin_set("fixed time step", true);
  dolfin_set("partitioning threshold", 0.5);
  //dolfin_set("time step conservation", 5.0);
#endif
  
#ifdef DEBUG_BENCHMARK
  // Parameters for debug (save solution)
  dolfin_set("number of samples", 10);
  dolfin_set("save solution", true);
#endif

  // FIXME: Remove when working
  //dolfin_set("save solution", true);
  //dolfin_set("progress step", 0.1);
  //dolfin_set("fixed time step", false);
  //dolfin_set("tolerance", k);

  // Solve the wave equation
  WaveEquation wave(n);
  wave.solve();
}
