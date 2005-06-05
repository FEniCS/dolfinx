// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

/// This test problem solves the wave equation in 2D, using the
/// largest possible stable time step given by the CFL condition.  The
/// problem can be run either in mono-adaptive mode (using the same
/// time step in the entire domain) or in multi-adaptive mode (using
/// small time steps only where the mesh size is small.

class WaveEquation : public ODE
{
public:

  WaveEquation(Mesh& mesh) : ODE(2*mesh.noNodes()), mesh(mesh),
			     A(mesh), M(mesh), offset(N/2)
  {
    // Parameters
    T = 1.0;

    setSparsity();
  }

  // Initial condition: a wave coming in from the right
  real u0(unsigned int i)
  {
    if ( i < offset )
    {
      const Point& p = mesh.node(i).coord();
      if ( p.x > 0.9 )
	return 1.0;
    }

    return 0.0;
  }

  // Right-hand side, mono-adaptive version
  real f(const real u[], real t, unsigned int i)
  {
    // First half of system
    if ( i < offset )
      return u[i + offset];
    
    // Second half of system
    //const unsigned int j = i - offset;

    return 0.0;
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
      /*
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
      */
    }
  }

  Mesh& mesh;

  StiffnessMatrix A;
  MassMatrix M;

  unsigned int offset;

};

int main()
{
  UnitSquare mesh(8, 8);
  WaveEquation ode(mesh);

  ode.solve();

  return 0;
}

/*

class RadiationDiffusion : public ODE
{
public:

  RadiationDiffusion(Mesh& mesh) : ODE(2*mesh.noNodes()), mesh(mesh), 
				   A(N, N), B(N, N), Dx(N, N), Dy(N, N), bnd(N/2),
				   ufile("solution.m"), kfile("timesteps.m"),
				   u1x(N/2), u2x(N/2), k1x(N/2), k2x(N/2),
				   u1(mesh, u1x), u2(mesh, u2x),
				   k1(mesh, k1x), k2(mesh, k2x)
  {
    // Parameters
    T  = 3.0;
    h  = 1.0 / 6.0;
    Z0 = 10.0;
    kappa = 0.005;

    // Create data for space discretization of system
    createStiffnessMatrix(mesh);
    createBoundaryConditions(mesh);
    createDerivatives(mesh);

    // Set names for samples
    u1.rename("u1", "Radiation energy (E)");
    u2.rename("u2", "Material temperature (T)");
    k1.rename("k1", "Time steps for the radiation energy (E)");
    k2.rename("k2", "Time steps for the material temperature (T)");
    
    // Specify sparsity pattern
    setSparsity();
  }
  
  /// Initial condition
  real u0(unsigned int i)
  {
    real E0 = 1.0e-5;
    real T0 = pow(E0, 0.25);
    
    if ( i < N/2 )
      return E0;
    else
      return T0;
  }
  
  /// The right-hand side
  real f(const Vector& u, real t, unsigned int i)
  {
    if ( i < N/2 )
      return radiation1(u, i) + diffusion1(u, i);
    else
      return radiation2(u, i) + diffusion2(u, i);
  }

private:

  /// Radiation for first component
  real radiation1(const Vector& u, unsigned int i)
  {
    real u1 = u(i);
    real u2 = u(i + N/2);
    const Point& p = mesh.node(i).coord();

    return sigma(p.x, p.y, u2) * (pow(u2, 4.0) - u1);
  }

  /// Radiation for second component
  real radiation2(const Vector& u, unsigned int i)
  {
    real u1 = u(i - N/2);
    real u2 = u(i);
    const Point& p = mesh.node(i - N/2).coord();

    return - sigma(p.x, p.y, u2) * (pow(u2, 4.0) - u1);    
  }

  /// Diffusion for first component
  real diffusion1(const Vector& u, unsigned int i)
  {
    real u1 = u(i);
    real u2 = u(i + N/2);
    const Point& p = mesh.node(i).coord();
    real s = sigma(p.x, p.y, u2);
    real epsilon = 1.0 / (3.0*s + grad(u, i) / u1);
    real bc = 0.0;

    if ( fabs(p.x - 0.0) < DOLFIN_EPS )
      bc = 6.0*s*bnd(i);
    
    return epsilon * (A.mult(u, i) + 1.5*s*B.mult(u, i) + bc);
  }

  /// Diffusion for second component
  real diffusion2(const Vector& u, unsigned int i)
  {
    real u2 = u(i);
    real epsilon = kappa * pow(u2, 2.5);
    
    return epsilon * A.mult(u, i);
  }
  
  /// Atomic number
  real z(real x, real y)
  {
    return (fabs(x - 0.5) <= h && fabs(y - 0.5) <= h ? Z0 : 1.0);
  }

  /// The factor sigma
  real sigma(real x, real y, real u2)
  {
    return pow(z(x, y) / u2, 3.0);
  }

  /// Compute norm of gradient, used only for first component
  real grad(const Vector& u, unsigned int i)
  {
    // We only need to compute the gradient of u1
    dolfin_assert(i < N/2);

    real dx = Dx.mult(u, i);
    real dy = Dy.mult(u, i);

    return sqrt(dx*dx + dy*dy);
  }

  /// Create system stiffness matrix from simple stiffness matrix,
  /// used for both components
  void createStiffnessMatrix(Mesh& mesh)
  {
    StiffnessMatrix A0(mesh);
    LoadVector b(mesh); // Same as lumped mass matrix

    for (unsigned int i = 0; i < N/2; i++)
    {
      for (unsigned int pos = 0; !A0.endrow(i, pos); pos++)
      {
        unsigned int j = 0;
        real element = A0(i, j, pos);
        A(i, j) = -element / b(i);
        A(i + N/2, j + N/2) = -element / b(i);
      }
    }
  }

  /// Create matrix for boundary conditions, used only for first component
  void createBoundaryConditions(Mesh& mesh)
  {
    Boundary boundary(mesh);
    LoadVector b(mesh); // Same as lumped mass matrix

    for (EdgeIterator edge(boundary); !edge.end(); ++edge)
    {
      Point& p0 = edge->node(0).coord();
      Point& p1 = edge->node(1).coord();
      unsigned int i = edge->node(0).id();
      unsigned int j = edge->node(1).id();

      if ( (fabs(p0.x - 0.0) < DOLFIN_EPS && fabs(p1.x - 0.0) < DOLFIN_EPS) ||
	   (fabs(p0.x - 1.0) < DOLFIN_EPS && fabs(p1.x - 1.0) < DOLFIN_EPS) )
      {
	real h = p0.dist(p1);

	B(i, i) -= (h/3.0) / b(i);
	B(i, j) -= (h/6.0) / b(i);
	B(j, i) -= (h/6.0) / b(j);
	B(j, j) -= (h/3.0) / b(j);

	bnd(i) += (h/2.0) / b(i);
	bnd(j) += (h/2.0) / b(j);
      }
    }
  }

  /// Create matrices for derivative, used only for first component
  void createDerivatives(Mesh& mesh)
  {
    DxMatrix Dx0(mesh);
    DyMatrix Dy0(mesh);
   
    for (unsigned int i = 0; i < N/2; i++)
    {
      for (unsigned int pos = 0; !Dx0.endrow(i, pos); pos++)
      {
	unsigned int j = 0;
	real element = Dx0(i, j, pos);
	Dx(i, j) = element;
      }

      for (unsigned int pos = 0; !Dy0.endrow(i, pos); pos++)
      {
	unsigned int j = 0;
	real element = Dy0(i, j, pos);
	Dy(i, j) = element;
      }
     }
  }
  
  void setSparsity()
  {
    cout << "Setting sparsity pattern." << endl;

    sparsity.clear();

    // Allocate number of dependencies
    for (NodeIterator n(mesh); !n.end(); ++n)
    {
      sparsity.setsize(n->id(), n->noNodeNeighbors() + 1);
      sparsity.setsize(n->id() + N/2, n->noNodeNeighbors() + 1);
    }

    // Set dependencies
    for (unsigned int i = 0; i < N/2; i++)
    {
      // First component
      unsigned int j = 0;
      for (unsigned int pos = 0; !A.endrow(i, pos); pos++)
      {
	real element = A(i, j, pos);
	if ( fabs(element) > DOLFIN_EPS )
	  sparsity.set(i, j, true);
      }
      for (unsigned int pos = 0; !Dx.endrow(i, pos); pos++)
      {
	real element = Dx(i, j, pos);
	if ( fabs(element) > DOLFIN_EPS )
	  sparsity.set(i, j, true);
      }
      for (unsigned int pos = 0; !Dx.endrow(i, pos); pos++)
      {
	real element = Dx(i, j, pos);
	if ( fabs(element) > DOLFIN_EPS )
	  sparsity.set(i, j, true);
      }
      sparsity.set(i, i + N/2, true);

      // Second component
      for (unsigned int pos = 0; !A.endrow(i + N/2, pos); pos++)
      {
	real element = A(i + N/2, j, pos);
	if ( fabs(element) > DOLFIN_EPS )
	  sparsity.set(i + N/2, j, true);
      }
      sparsity.set(i + N/2, i, true);
    }

    cout << "Done." << endl;
  }

  void save(Sample& sample)
  {
    // Set current time
    u1.update(sample.t());
    u2.update(sample.t());
    k1.update(sample.t());
    k2.update(sample.t());
    
    // Get the degrees of freedom
    for (unsigned int i = 0; i < N/2; i++)
    {
      u1x(i) = sample.u(i);
      u2x(i) = sample.u(i + N/2);
      k1x(i) = sample.k(i);
      k2x(i) = sample.k(i + N/2);
    }

    // Save solution and time steps to file
    dolfin_log(false);
    ufile << u1;
    ufile << u2;
    kfile << k1;
    kfile << k2;
    dolfin_log(true);
  }

  void test()
  {
    cout << "Checking gradient" << endl;

    Vector x(N);
    
    for (NodeIterator n(mesh); !n.end(); ++n)
    {
      const Point& p = n->coord();
      x(n->id()) = p.x*p.y;
      cout << "exact gradient = " << sqrt(p.x*p.x + p.y*p.y) << endl;
    }

    for (unsigned int i = 0; i < x.size()/2; i++)
      cout << "approximate gradient = " << grad(x, i) << endl;
  }
  
private:
  
  Mesh& mesh; // The mesh
  Matrix A;   // Stiffness matrix
  Matrix B;   // Matrix for boundary conditions
  Matrix Dx;  // Derivative in x-direction
  Matrix Dy;  // Derivative in y-direction
  Vector bnd; // Load vector for boundary conditions
  File ufile; // File for storing the solution
  File kfile; // File for storing the time steps
  
  real h;     // Half size of box with higher density
  real Z0;    // Large density
  real kappa; // Diffusion parameter 

  Vector u1x, u2x, k1x, k2x; // Sample of solution (data)
  Function u1, u2, k1, k2;   // Sample of solution (functions)  
};

*/

/*

int main()
{
  // FIXME: BROKEN
  dolfin_error("Broken, needs to be updated.");
  
  // Settings
  //dolfin_set("output", "plain text");
  dolfin_set("fixed time step", true);
  dolfin_set("initial time step", 1e-3);
  dolfin_set("solve dual problem", false);
  dolfin_set("maximum time step", 1.0);
  dolfin_set("tolerance", 1e-10);
  dolfin_set("number of samples", 20);
  dolfin_set("progress step", 0.01);

  dolfin_info("This code does not yet produce correct results.");
  delay(1.0);

  // Number of refinements
  unsigned int refinements = 6;
  
  // Read and refine mesh
  Mesh mesh("mesh.xml.gz");
  for (unsigned int i = 0; i < refinements; i++)
    mesh.refineUniformly();

  // Save mesh to file
  File file("mesh.m");
  file << mesh;

  // Solve equation
  RadiationDiffusion ode(mesh);
  ode.solve();

  return 0;
}

*/
