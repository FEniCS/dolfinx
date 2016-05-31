#ifndef MM_FEM_TEST_HH
#define MM_FEM_TEST_HH

#include <dolfin.h>

namespace fem
{
  using namespace dolfin;

  template<class TMultiMeshFunctionSpace,
	   class TMultiMeshBilinearForm,
	   class TMultiMeshLinearForm>
  inline
  std::shared_ptr<const MultiMeshFunction>
  solve(std::shared_ptr<const MultiMesh> multimesh,
	std::shared_ptr<const SubDomain> dirichletboundary,
	std::shared_ptr<const Expression> source)
  {
    // Create function space
    auto V = std::make_shared<TMultiMeshFunctionSpace>(multimesh);

    // Create forms
    auto a = std::make_shared<TMultiMeshBilinearForm>(V,V);
    auto L = std::make_shared<TMultiMeshLinearForm>(V);

    // Attach coefficients
    L->f = source;

    // Assemble linear system
    auto A = std::make_shared<Matrix>();
    auto b = std::make_shared<Vector>();
    assemble_multimesh(*A, *a);
    assemble_multimesh(*b, *L);

    // Apply boundary condition
    auto zero = std::make_shared<Constant>(0);
    auto bc = std::make_shared<MultiMeshDirichletBC>(V, zero, dirichletboundary);
    bc->apply(*A, *b);

    // Compute solution
    auto u = std::make_shared<MultiMeshFunction>(V);
    solve(*A, *u->vector(), *b);

    return u;
  }

}

#endif
