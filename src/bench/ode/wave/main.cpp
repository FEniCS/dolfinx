// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005
//
// First added:  2005
// Last changed: 2006-10-26

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

  WaveEquation(Mesh& mesh) : ODE(2*mesh.numVertices(), 1.0), mesh(mesh),
			     A(mesh), offset(N/2), h(N/2)
  {
    // Width of initial wave
    w = 0.25;

    // Lump mass matrix
    uBlasMassMatrix M(mesh);
    FEM::lump(M, m);

    // Set dependencies
    for (unsigned int i = 0; i < offset; i++)
    {
      // Dependencies for first half of system
      dependencies.setsize(i, 1);
      dependencies.set(i, i + offset);

      // Dependencies for second half of system
      int ncols = 0;
      Array<int> columns;
      Array<real> values;
      A.getRow(i, ncols, columns, values);
      dependencies.setsize(i + offset, ncols);
      for (int j = 0; j < ncols; j++)
	dependencies.set(i + offset, columns[j]);
    }

    // Get local mesh size (check smallest neighboring triangle)
    hmin = 1.0;
    for (VertexIterator v(mesh); !v.end(); ++v)
    {
      real dmin = 1.0;
      for (CellIterator c(v); !c.end(); ++c)
      {
	const real d = c->diameter();
	if ( d < dmin )
	  dmin = d;
      }
      h[v->index()] = dmin;
      if ( dmin < hmin )
	hmin = dmin;
    }
    dolfin::cout << "Minimum mesh size: h = " << hmin << dolfin::endl;
    dolfin::cout << "Maximum time step: k = " << 0.25*hmin << dolfin::endl;
  }

  // Initial condition: a wave coming in from the right
  void u0(uBlasVector& u)
  {
    for (unsigned int i = 0; i < N; i++)
    {
      const real x0 = 1.0 - 0.5*w;
      u(i) = 0.0;

      if ( i < offset )
      {
	const Point p = mesh.geometry().point(i);
	if ( std::abs(p.x() - x0) < 0.5*w )
	  u(i) = 0.5*(cos(2.0*DOLFIN_PI*(p.x() - x0)/w) + 1.0);
      }
      else
      {
	const Point p = mesh.geometry().point(i);
	if ( std::abs(p.x() - x0) < 0.5*w )
	  u(i) = -(DOLFIN_PI/w)*sin(2.0*DOLFIN_PI*(p.x() - x0)/w);
      }
    }
  }

  // Global time step
  real timestep(real t, real k0) const
  {
    return 0.1*hmin;
  }
  
  // Local time step
  real timestep(real t, unsigned int i, real k0) const
  {
    return 0.1*h[i % offset];
  }

  // Right-hand side, mono-adaptive version
  void f(const uBlasVector& u, real t, uBlasVector& y)
  {
    // First half of system
    for (unsigned int i = 0; i < offset; i++)
    {
      y(i) = u(i + offset);
    }
    
    // Second half of system
    for (unsigned int i = offset; i < N; i++)
    {
      const unsigned int j = i - offset;
      y(i) = - inner_prod(row(A, j), u) / m(j);
    }
  }

  // Right-hand side, multi-adaptive version
  real f(const uBlasVector& u, real t, unsigned int i)
  {
    // First half of system
    if ( i < offset )
      return u(i + offset);
    
    // Second half of system
    const unsigned int j = i - offset;

    return - inner_prod(row(A, j), u) / m(j);
  }
  
  void save(Sample& sample)
  {
    cout << "Saving data at t = " << sample.t() << endl;

    // Create functions
    static Vector ux(N/2);
    static Vector vx(N/2);
    static Vector kx(N/2);
    static Vector rx(N/2);
    static P1tri element;
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
    u.sync(sample.t());
    v.sync(sample.t());
    k.sync(sample.t());
    r.sync(sample.t());
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

  Mesh& mesh;             // The mesh
  uBlasStiffnessMatrix A; // Stiffness matrix
  uBlasVector m;          // Lumped mass matrix
  unsigned int offset;    // N/2, number of vertices
  Array<real> h;          // Local mesh size
  real hmin;              // Minimum mesh size
  real w;                 // Width of initial wave

};

int main(int argc, char* argv[])
{
  // Parse command line arguments
  if ( argc != 2 )
  {
    dolfin_info("Usage: dolfin-ode-reaction method");
    dolfin_info("");
    dolfin_info("method - 'cg' or 'mcg'");
    return 1;
  }
  const char* method = argv[1];

  // Load solver parameters from file
  File file("parameters-bench.xml");
  file >> ParameterSystem::parameters;

  // Set solution method
  set("ODE method", method);

  // Solve system of ODEs
  Mesh mesh("slit.xml.gz");
  WaveEquation ode(mesh);
  ode.solve();

  return 0;
}
