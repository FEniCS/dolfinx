#include <dolfin.h>

void assemble(dolfin::LinearForm& L, dolfin::Vector& b, dolfin::Mesh& mesh)
{
  dolfin::FEM::assemble(L, b, mesh);
}

void assemble(dolfin::BilinearForm& a, dolfin::Matrix& A, dolfin::Mesh& mesh)
{
  dolfin::FEM::assemble(a, A, mesh);
}

void assemble(dolfin::BilinearForm& a, dolfin::LinearForm& L,
	      dolfin::Matrix& A, dolfin::Vector& b,
	      dolfin::Mesh& mesh, dolfin::BoundaryCondition& bc)
{
  dolfin::FEM::assemble(a, L, A, b, mesh, bc);
}

void applyBC(dolfin::Vector& b, dolfin::Mesh& mesh,
	     dolfin::FiniteElement& element, dolfin::BoundaryCondition& bc)
{
  dolfin::FEM::applyBC(b, mesh, element, bc);
}

void applyBC(dolfin::Matrix& A, dolfin::Mesh& mesh,
	     dolfin::FiniteElement& element, dolfin::BoundaryCondition& bc)
{
  dolfin::FEM::applyBC(A, mesh, element, bc);
}
