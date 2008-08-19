// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005
//
// First added:  2005
// Last changed: 2007-05-24

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
    uBLASMassMatrix M(mesh);
    M.lump(m);

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
      A.getrow(i, ncols, columns, values);
      dependencies.setsize(i + offset, ncols);
      for (int j = 0; j < ncols; j++)
	dependencies.set(i + offset, columns[j]);
    }

    // Get local mesh size (check smallest neighboring triangle)
    hmin = 1.0;
    mesh.init(0, mesh.topology().dim());
    for (VertexIterator v(mesh); !v.end(); ++v)
    {
      real dmin = 1.0;
      for (CellIterator c(*v); !c.end(); ++c)
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
  void u0(uBLASVector& u)
  {
    for (unsigned int i = 0; i < N; i++)
    {
      const real x0 = 1.0 - 0.5*w;
      u(i) = 0.0;

      if ( i < offset )
      {
        const real x = mesh.geometry().x(i, 0);
	if ( std::abs(x - x0) < 0.5*w )
	  u(i) = 0.5*(cos(2.0*DOLFIN_PI*(x - x0)/w) + 1.0);
      }
      else
      {
        const real x = mesh.geometry().x(i - offset, 0);
	if ( std::abs(x - x0) < 0.5*w )
	  u(i) = -(DOLFIN_PI/w)*sin(2.0*DOLFIN_PI*(x - x0)/w);
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
  void f(const uBLASVector& u, real t, uBLASVector& y)
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
  real f(const uBLASVector& u, real t, unsigned int i)
  {
    // First half of system
    if ( i < offset )
      return u(i + offset);
    
    // Second half of system
    const unsigned int j = i - offset;

    return - inner_prod(row(A, j), u) / m(j);
  }
 
private:

  Mesh& mesh;             // The mesh
  uBLASStiffnessMatrix A; // Stiffness matrix
  uBLASVector m;          // Lumped mass matrix
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
    message("Usage: dolfin-ode-reaction method");
    message("");
    message("method - 'cg' or 'mcg'");
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
