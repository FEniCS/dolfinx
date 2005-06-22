// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <dolfin.h>

using namespace dolfin;

class Bistable : public ODE
{
public:

  Bistable(Mesh& mesh) : ODE(mesh.noNodes()), mesh(mesh), A(mesh, 0.001),
			 ufile("solution.dx"), kfile("timesteps.dx")
  {
    // Parameters
    T = 20.0;
    
    // Create lumped mass matrix
    MassMatrix M(mesh);
    FEM::lump(M, m);

    // Compute sparsity
    sparse(A);
  }
  
  real u0(unsigned int i)
  {
    return 2.0*(dolfin::rand() - 0.5);
  }
  
  real f(const real u[], real t, unsigned int i)
  {
    return (-A.mult(u, i) + m(i)*u[i]*(1.0 - u[i]*u[i])) / m(i);
  }

  void save(Sample& sample)
  {
    // Create a mesh-dependent function from the sample
    static Vector ux(N);
    static Vector kx(N);
    static P1Tri element;
    static Function u(ux, mesh, element);
    static Function k(kx, mesh, element);
    u.rename("u", "Solution of the bistable equation");
    k.rename("k", "Time steps for the bistable equation");

    // Get the degrees of freedom and set current time
    u.set(sample.t());
    k.set(sample.t());
    for (unsigned int i = 0; i < N; i++)
    {
      ux(i) = sample.u(i);
      kx(i) = sample.k(i);
    }

    // Save solution to file
    ufile << u;
    kfile << k;
  }
  
private:

  Mesh& mesh;        // The mesh
  StiffnessMatrix A; // The stiffness matrix
  Vector m;          // Lumped mass matrix

  File ufile; // OpenDX file for the solution
  File kfile; // OpenDX file for the time steps

};

int main()
{
  // Settings
  dolfin_set("solve dual problem", false);
  dolfin_set("maximum time step", 1.0);
  dolfin_set("tolerance", 0.001);
  dolfin_set("number of samples", 100);
  dolfin_set("progress step", 0.01);
  dolfin_set("solver", "newton");
  
  // Number of refinements
  unsigned int refinements = 3;
  //unsigned int refinements = 5;
  
  // Read and refine mesh
  Mesh mesh("mesh.xml.gz");
  for (unsigned int i = 0; i < refinements; i++)
    mesh.refineUniformly();

  // Save mesh to file
  File file("mesh.dx");
  file << mesh;

  // Solve equation
  Bistable bistable(mesh);
  bistable.solve();

  return 0;
}
