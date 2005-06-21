// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

/// This test problem solves the wave equation in 2D, using large time
/// steps given by the CFL condition. The problem can be run either in
/// mono-adaptive mode (using the same time step in the entire domain)
/// or in multi-adaptive mode (using small time steps only where the
/// mesh size is small).

class WaveEquation : public ODE
{
public:

  WaveEquation(Mesh& mesh) : ODE(2*mesh.noNodes()), mesh(mesh),
			     A(mesh), offset(N/2), h(N/2)
  {
    T = 1.0;  // Final time
    w = 0.25; // Width of initial wave

    // Lump mass matrix
    MassMatrix M(mesh);
    FEM::lump(M, m);

    // Set dependencies
    for (unsigned int i = 0; i < offset; i++)
    {
      // Dependencies for first half of system
      dependencies.setsize(i, 1);
      dependencies.set(i, i + offset);

      // Dependencies for second half of system
      int ncols = 0;
      const int* cols = 0;
      const real* vals = 0;
      MatGetRow(A.mat(), static_cast<int>(i), &ncols, &cols, &vals);
      dependencies.setsize(i + offset, ncols);
      real rowsum = 0.0;
      for (unsigned int j = 0; j < ncols; j++)
	dependencies.set(i + offset, cols[j]);
      MatRestoreRow(A.mat(), static_cast<int>(i), &ncols, &cols, &vals);
    }

    // Get local mesh size (check smallest neighboring triangle)
    hmin = 1.0;
    for (NodeIterator n(mesh); !n.end(); ++n)
    {
      real dmin = 1.0;
      for (CellIterator c(n); !c.end(); ++c)
      {
	const real d = c->diameter();
	if ( d < dmin )
	  dmin = d;
      }
      h[n->id()] = dmin;
      if ( dmin < hmin )
	hmin = dmin;
    }
    dolfin::cout << "Minimum mesh size: h = " << hmin << dolfin::endl;
    dolfin::cout << "Maximum time step: k = " << 0.25*hmin << dolfin::endl;
  }

  // Initial condition: a wave coming in from the right
  real u0(unsigned int i)
  {
    const real x0 = 1.0 - 0.5*w;
    
    if ( i < offset )
    {
      const Point& p = mesh.node(i).coord();
      if ( fabs(p.x - x0) < 0.5*w )
	return 0.5*(cos(2.0*DOLFIN_PI*(p.x - x0)/w) + 1.0);
    }
    else
    {
      const Point& p = mesh.node(i - offset).coord();
      if ( fabs(p.x - x0) < 0.5*w )
      	return -(DOLFIN_PI/w)*sin(2.0*DOLFIN_PI*(p.x - x0)/w);
    }

    return 0.0;
  }

  // Global time step
  real timestep()
  {
    return 0.1*hmin;
  }
  
  // Local time step
  real timestep(unsigned int i)
  {
    return 0.1*h[i % offset];
  }

  // Right-hand side, mono-adaptive version
  void f(const real u[], real t, real y[])
  {
    // First half of system
    for (unsigned int i = 0; i < offset; i++)
    {
      y[i] = u[i + offset];
    }
    
    // Second half of system
    for (unsigned int i = offset; i < N; i++)
    {
      const unsigned int j = i - offset;
      y[i] = -A.mult(u, j) / m(j);
    }
  }

  // Right-hand side, multi-adaptive version
  real f(const real u[], real t, unsigned int i)
  {
    // First half of system
    if ( i < offset )
      return u[i + offset];
    
    // Second half of system
    const unsigned int j = i - offset;

    return -A.mult(u, j) / m(j);
  }
  
  void save(Sample& sample)
  {
    cout << "Saving data at t = " << sample.t() << endl;

    // Create functions
    static Vector ux(N/2);
    static Vector vx(N/2);
    static Vector kx(N/2);
    static Vector rx(N/2);
    static P1Tri element;
    static Function u(ux, mesh, element);
    static Function v(vx, mesh, element);
    static Function k(kx, mesh, element);
    static Function r(rx, mesh, element);
    static File ufile("solutionu.m");
    static File vfile("solutionv.m");
    static File kfile("timesteps.m");
    static File rfile("residual.m");
    
    u.rename("u", "Solution of the wave equation");
    v.rename("v", "Speed of the wave equation");
    k.rename("k", "Time steps for the wave equation");
    r.rename("r", "Time residual for the wave equation");

    // Get the degrees of freedom and set current time
    u.set(sample.t());
    v.set(sample.t());
    k.set(sample.t());
    r.set(sample.t());
    for (unsigned int i = 0; i < offset; i++)
    {
      ux(i) = sample.u(i);
      vx(i) = sample.u(i + offset);
      kx(i) = sample.k(i + offset);
      rx(i) = sample.r(i + offset);
    }

    // Save solution to file
    ufile << u;
    vfile << v;
    kfile << k;
    rfile << r;
  }

private:

  Mesh& mesh;          // The mesh
  StiffnessMatrix A;   // Stiffness matrix
  Vector m;            // Lumped mass matrix
  Array<real> h;       // Local mesh size
  real hmin;           // Minimum mesh size
  real w;              // Width of initial wave
  unsigned int offset; // N/2, number of nodes

};

int main()
{
  dolfin_set("method", "mcg");
  dolfin_set("fixed time step", true);
  dolfin_set("discrete tolerance", 0.1);

  // Comment this out to save solution
  dolfin_set("save solution", false);

  Mesh mesh("slit.xml.gz");
  WaveEquation ode(mesh);
  
  ode.solve();

  return 0;
}
