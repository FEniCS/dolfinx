// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class Bistable : public ODE {
public:

  Bistable(Mesh& mesh) : ODE(mesh.noNodes()), mesh(mesh), 
			 A(mesh, 0.001), b(mesh),
			 ufile("solution.dx"), kfile("timesteps.dx")
  {
    // Parameters
    T = 20.0;
    
    // Create lumped mass matrix
    MassMatrix M(mesh);
    M.lump(m);

    // Compute sparsity
    sparse(A);
  }
  
  real u0(unsigned int i)
  {
    return 2.0*(dolfin::rand() - 0.5);
  }
  
  real f(const Vector& u, real t, unsigned int i)
  {
    return (-A.mult(u, i) + b(i)*u(i)*(1.0 - u(i)*u(i))) / m(i);
  }

  void save(Sample& sample)
  {
    // Create a mesh-dependent function from the sample
    Vector ux(N);
    Vector kx(N);
    Function u(mesh, ux);
    Function k(mesh, kx);
    u.rename("u", "Solution of the bistable equation");
    u.rename("k", "Time steps for the bistable equation");

    // Get the degrees of freedom and set current time
    u.update(sample.t());
    k.update(sample.t());
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
  LoadVector b;      // The load vector
  Vector m;          // Lumped mass matrix

  File ufile; // OpenDX file for the solution
  File kfile; // OpenDX file for the time steps

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
  unsigned int refinements = 5;
  
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
