// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

/// This test problem is taken from the paper "An Implicit-Explicit
/// Runge-Kutta-Chebyshev Scheme for Diffusion-Reaction Equations" by
/// Verwer and Sommeijer, published in SIAM J. Scientific Computing
/// (2004).

class RadiationDiffusion : public ODE
{
public:

  RadiationDiffusion(Mesh& mesh) : ODE(2*mesh.noNodes()), mesh(mesh), 
				   A(2*mesh.noNodes(), 2*mesh.noNodes()),
				   Dx(2*mesh.noNodes(), 2*mesh.noNodes()),
				   Dy(2*mesh.noNodes(), 2*mesh.noNodes()),
				   b(2*mesh.noNodes()),
				   ufile("solution.m"), kfile("timesteps.m")
  {
    // Parameters
    T  = 3.0;
    h  = 1.0 / 6.0;
    Z0 = 10.0;
    kappa = 0.005;
    
    // Create data for space discretization of system
    createStiffnessMatrix(mesh);
    createLoadVector(mesh);
    createDerivatives(mesh);
  }
  
  /// Initial condition
  real u0(unsigned int i)
  {
    // Using constant initial conditions throughout the domain
    
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
      return (radiation1(u, i) + diffusion1(u, i)) / b(i);
    else
      return (radiation2(u, i) + diffusion2(u, i)) / b(i);
  }

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
    real epsilon = 1.0 / (3.0*sigma(p.x, p.y, u2) + grad(u, i) / u1);

    return epsilon * A.mult(u, i);
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

  /// Compute norm of gradient of the first component
  real grad(const Vector& u, unsigned int i)
  {
    // We only need to compute the gradient of u1
    dolfin_assert(i < N/2);

    return 1.0;
  }

  /// Create system stiffness matrix from simple stiffness matrix
  void createStiffnessMatrix(Mesh& mesh)
  {
    StiffnessMatrix A0(mesh);
   
    for (unsigned int i = 0; i < N/2; i++)
    {
      for (unsigned int pos = 0; !A0.endrow(i, pos); pos++)
      {
        unsigned int j = 0;
        real element = A0(i, j, pos);
        A(i, j) = element;
        A(i + N/2, j + N/2) = element;
      }
    }
  }

  /// Create system load vector from simple load vector
  void createLoadVector(Mesh& mesh)
  {
    LoadVector b0(mesh);

    for (unsigned int i = 0; i < N/2; i++)
    {
      b(i) = b0(i);
      b(i + N/2) = b0(i);
    }
  }

  /// Create matrices for derivative of first component
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

      for (unsigned int pos = 0; !Dx0.endrow(i, pos); pos++)
      {
	unsigned int j = 0;
	real element = Dy0(i, j, pos);
	Dy(i, j) = element;
      }
     }
   }

  void save(Sample& sample)
  {
    // Create  mesh-dependent functions from the sample
    Vector u1x(N/2), u2x(N/2), k1x(N/2), k2x(N/2);
    Function u1(mesh, u1x), u2(mesh, u2x), k1(mesh, k1x), k2(mesh, k2x);
    u1.rename("u1", "Radiation energy (E)");
    u2.rename("u2", "Material temperature (T)");
    k1.rename("k1", "Time steps for the radiation energy (E)");
    k2.rename("k2", "Time steps for the material temperature (T)");

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
    ufile << u1;
    ufile << u2;
    kfile << k1;
    kfile << k2;
  }
  
private:

  Mesh& mesh; // The mesh
  Matrix A;   // Stiffness matrix
  Matrix Dx;  // Derivative in x-direction
  Matrix Dy;  // Derivative in y-direction
  Vector b;   // Load vector
  File ufile; // File for storing the solution
  File kfile; // File for storing the time steps

  real h;     // Half size of box with higher density
  real Z0;    // Large density
  real kappa; // Diffusion parameter 
};

int main()
{
  // Settings
  dolfin_set("output", "plain text");
  dolfin_set("solve dual problem", false);
  dolfin_set("maximum time step", 1.0);
  dolfin_set("tolerance", 0.001);
  dolfin_set("number of samples", 100);
  dolfin_set("progress step", 0.01);

  // Number of refinements
  unsigned int refinements = 2;
  
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
