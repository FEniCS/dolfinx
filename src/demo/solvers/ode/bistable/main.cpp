// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class Stiffness : public PDE {
public:
  
  Stiffness(real epsilon) : PDE(2), epsilon(epsilon) {}
  
  real lhs(const ShapeFunction& u, const ShapeFunction& v)
  {
    return epsilon*(grad(u),grad(v)) * dK;
  }
  
  real rhs(const ShapeFunction& v)
  {
    return 1.0*v * dK;
  }

private:

  real epsilon;

};

class Mass : public PDE {
public:
  
  Mass() : PDE(2) {}

  real lhs(const ShapeFunction& u, const ShapeFunction& v)
  {
    return u*v * dK;
  }

};

class Bistable : public ODE {
public:

  Bistable(Mesh& mesh) : ODE(mesh.noNodes()), mesh(mesh)
  {
    // Parameters
    T = 50.0;
    real epsilon = 0.001;

    Galerkin fem;
    
    // Assemble stiffness matrix
    Stiffness stiffness(epsilon);
    fem.assemble(stiffness, mesh, A, b);
    
    // Assemble lumped mass matrix
    Matrix M;
    Mass mass;
    fem.assemble(mass, mesh, M);
    M.lump(m);

    // Compute sparsity
    sparse(A);
  }
  
  real u0(unsigned int i)
  {
    // Initial data specified in Computer Session E4
    //Point p = mesh.node(i).coord();
    //return cos(2.0*DOLFIN_PI*p.x*p.x)*cos(2.0*DOLFIN_PI*p.y*p.y);

    // Random initial data
    return 2.0*(dolfin::rand() - 0.5);
  }
  
  real f(const Vector& u, real t, unsigned int i)
  {
    return (-A.mult(u, i) + b(i)*u(i)*(1.0 - u(i)*u(i))) / m(i);
  }
  
private:

  Mesh& mesh; // The mesh
  Matrix A;   // Stiffness matrix
  Vector m;   // Lumped mass matrix
  Vector b;   // Weights for right-hand side

};

int main()
{
  // Settings
  dolfin_set("output", "plain text");
  dolfin_set("solve dual problem", false);
  dolfin_set("maximum time step", 1.0);
  dolfin_set("tolerance", 0.001);
  dolfin_set("number of samples", 50);
  dolfin_set("progress step", 0.01);

  // Number of refinements
  unsigned int refinements = 5;
  
  // Read and refine mesh
  Mesh mesh("mesh.xml.gz");
  for (unsigned int i = 0; i < refinements; i++)
    mesh.refineUniformly();

  // Save mesh to file
  File file("mesh.m");
  file << mesh;

  // Solve equation
  Bistable bistable(mesh);
  bistable.solve();

  return 0;
}
