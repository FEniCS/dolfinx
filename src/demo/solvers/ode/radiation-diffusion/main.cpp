// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

/// This test problem is taken from the paper "An Implicit-Explicit Runge-Kutta-Chebyshev
/// Scheme for Diffusion-Reaction Equations" by Verwer and Sommeijer, published in
/// SIAM J. Scientific Computing (2004).

class RadiationDiffusion : public ODE
{
public:

  RadiationDiffusion(Mesh& mesh) : ODE(2*mesh.noNodes()), mesh(mesh), 
				   A(mesh), b(mesh),
				   ufile("solution.m"), kfile("timesteps.m")
  {
    // Parameters
    T = 3.0;
    
    // Create lumped mass matrix
    MassMatrix M(mesh);
    M.lump(m);

    // Compute sparsity
    sparse(A);
  }

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
  
  real f(const Vector& u, real t, unsigned int i)
  {
    return 1.0;
    //return (-A.mult(u, i) + b(i)*u(i)*(1.0 - u(i)*u(i))) / m(i);
  }

  void save(Sample& sample)
  {
    // Create  mesh-dependent functions from the sample
    Vector Ex(N), Tx(N), kEx(N), kTx(N);
    Function E(mesh, Ex), T(mesh, Tx), kE(mesh, kEx), kT(mesh, kTx);
    E.rename("E", "Radiation energy");
    T.rename("T", "Material temperature");
    kE.rename("kE", "Time steps for the radiation energy");
    kT.rename("kT", "Time steps for the material temperature");

    // Set current time
    E.update(sample.t());
    T.update(sample.t());
    kE.update(sample.t());
    kT.update(sample.t());
    
    // Get the degrees of freedom
    for (unsigned int i = 0; i < N; i++)
    {
      Ex(i)  = sample.u(i);
      Tx(i)  = sample.u(i + N/2);
      kEx(i) = sample.k(i);
      kTx(i) = sample.k(i + N/2);
    }

    // Save solution and time steps to file
    ufile << E;
    ufile << T;
    kfile << kE;
    kfile << kT;
  }
  
private:

  Mesh& mesh;        // The mesh
  StiffnessMatrix A; // The unmodified stiffness matrix
  LoadVector b;      // The load vector
  Vector m;          // Lumped mass matrix

  File ufile; // File for storing the solution
  File kfile; // File for storing the time steps

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
