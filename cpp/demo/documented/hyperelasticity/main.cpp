#include "HyperElasticity.h"
#include <dolfin.h>

using namespace dolfin;

// Sub domain for clamp at left end
class Left : public mesh::SubDomain
{
  EigenArrayXb inside(Eigen::Ref<const EigenRowArrayXXd> x,
                      bool on_boundary) const
  {
    EigenArrayXb flags(x.rows());
    for (int i = 0; i < x.rows(); ++i)
      flags[i] = (std::abs(x(i, 0)) < DOLFIN_EPS) and on_boundary;

    return flags;
  }
};

// Sub domain for rotation at right end
class Right : public mesh::SubDomain
{
  EigenArrayXb inside(Eigen::Ref<const EigenRowArrayXXd> x,
                      bool on_boundary) const
  {
    EigenArrayXb flags(x.rows());
    for (int i = 0; i < x.rows(); ++i)
      flags[i] = (std::abs(x(i, 0) - 1.0) < DOLFIN_EPS) and on_boundary;

    return flags;
  }
};

// Dirichlet boundary condition for clamp at left end
class Clamp : public function::Expression
{
public:
  Clamp() : function::Expression({3}) {}

  void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                values,
            Eigen::Ref<const EigenRowArrayXXd> x,
            const dolfin::mesh::Cell& cell) const
  {
    for (int i = 0; i < x.rows(); ++i)
    {
      values(i, 0) = 0.0;
      values(i, 1) = 0.0;
      values(i, 2) = 0.0;
    }
  }
};

// Dirichlet boundary condition for rotation at right end
class Rotation : public function::Expression
{
public:
  Rotation() : function::Expression({3}) {}

  void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                values,
            Eigen::Ref<const EigenRowArrayXXd> x,
            const dolfin::mesh::Cell& cell) const
  {
    const double scale = 0.5;

    // Center of rotation
    const double y0 = 0.5;
    const double z0 = 0.5;

    // Large angle of rotation (60 degrees)
    double theta = 1.04719755;

    for (int i = 0; i < x.rows(); ++i)
    {
      // New coordinates
      double y = y0 + (x(1, 1) - y0) * cos(theta) - (x(i, 2) - z0) * sin(theta);
      double z = z0 + (x(i, 1) - y0) * sin(theta) + (x(i, 2) - z0) * cos(theta);

      // Rotate at right end
      values(i, 0) = 0.0;
      values(i, 1) = scale * (y - x(i, 1));
      values(i, 2) = scale * (z - x(i, 2));
    }
  }
};

// Next:
//
// .. code-block:: cpp

class HyperElasticProblem : public nls::NonlinearProblem
{
public:
  HyperElasticProblem(std::shared_ptr<function::Function> u,
                      std::shared_ptr<fem::Form> L,
                      std::shared_ptr<fem::Form> J,
                      std::vector<std::shared_ptr<const fem::DirichletBC>> bcs)
      : _u(u), _l(L), _j(J), _bcs(bcs)
  {
    // Do nothing
  }

  /// Destructor
  virtual ~HyperElasticProblem() = default;

  /// Compute F at current point x
  virtual la::PETScVector* F(const la::PETScVector& x)
  {
    // TODO
    return b.get();
  }

  /// Compute J = F' at current point x
  virtual la::PETScMatrix* J(const la::PETScVector& x)
  {
    // TODO
    return A.get();
  }

private:
  std::shared_ptr<function::Function> _u;
  std::shared_ptr<fem::Form> _j;
  std::shared_ptr<fem::Form> _l;
  std::vector<std::shared_ptr<const fem::DirichletBC>> _bcs;

  std::unique_ptr<la::PETScVector> b;
  std::unique_ptr<la::PETScMatrix> A;

  // A, b
  // _u, _l, _j, _bcs
};

int main()
{

  // Inside the ``main`` function, we begin by defining a tetrahedral mesh
  // of the domain and the function space on this mesh. Here, we choose to
  // create a unit cube mesh with 25 ( = 24 + 1) verices in one direction
  // and 17 ( = 16 + 1) vertices in the other two directions. With this
  // mesh, we initialize the (finite element) function space defined by the
  // generated code.
  //
  // .. code-block:: cpp

  // Create mesh and define function space
  std::array<geometry::Point, 2> pt
      = {geometry::Point(0.0, 0.0, 0.0), geometry::Point(1.0, 1.0, 1.0)};
  auto mesh = std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
      MPI_COMM_WORLD, pt, {{8, 8, 8}}, mesh::CellType::Type::tetrahedron,
      mesh::GhostMode::none));

  auto space
      = std::unique_ptr<dolfin_function_space>(HyperElasticityFunctionSpace());
  auto V = std::make_shared<function::FunctionSpace>(
      mesh,
      std::make_shared<fem::FiniteElement>(
          std::shared_ptr<ufc_finite_element>(space->element())),
      std::make_shared<fem::DofMap>(
          std::shared_ptr<ufc_dofmap>(space->dofmap()), *mesh));

  // Define Dirichlet boundaries
  auto left = std::make_shared<Left>();
  auto right = std::make_shared<Right>();

  // Define Dirichlet boundary functions
  auto c = std::make_shared<Clamp>();
  auto r = std::make_shared<Rotation>();

  // Create Dirichlet boundary conditions
  auto u0 = std::make_shared<function::Function>(V);
  std::vector<std::shared_ptr<const fem::DirichletBC>> bc
      = {std::make_shared<fem::DirichletBC>(V, u0, left),
         std::make_shared<fem::DirichletBC>(V, u0, right)};

  // Define solution function
  auto u = std::make_shared<function::Function>(V);

  // fem::DirichletBC bcl(V, c, left);
  // fem::DirichletBC bcr(V, r, right);
  // std::vector<const DirichletBC*> bcs = {{&bcl, &bcr}};

  // // The two boundary conditions are collected in the container ``bcs``.
  // //
  // // We use two instances of the class :cpp:class:`Constant` to define the
  // // source ``B`` and the traction ``T``.
  // //
  // // .. code-block:: cpp

  //   // Define source and boundary traction functions
  //   auto B = std::make_shared<Constant>(0.0, -0.5, 0.0);
  //   auto T = std::make_shared<Constant>(0.1,  0.0, 0.0);

  // // The solution for the displacement will be an instance of the class
  // // :cpp:class:`Function`, living in the function space ``V``; we define
  // // it here:
  // //
  // // .. code-block:: cpp

  //   // Define solution function
  //   auto u = std::make_shared<Function>(V);

  // // Next, we set the material parameters
  // //
  // // .. code-block:: cpp

  //   // Set material parameters
  //   const double E  = 10.0;
  //   const double nu = 0.3;
  //   auto mu = std::make_shared<Constant>(E/(2*(1 + nu)));
  //   auto lambda = std::make_shared<Constant>(E*nu/((1 + nu)*(1 - 2*nu)));

  // // Now, we can initialize the bilinear and linear forms (``a``, ``L``)
  // // using the previously defined :cpp:class:`FunctionSpace` ``V``. We
  // // attach the material parameters and previously initialized functions to
  // // the forms.
  // //
  // // .. code-block:: cpp

  //   // Create (linear) form defining (nonlinear) variational problem
  //   HyperElasticity::ResidualForm F(V);
  //   F.mu = mu; F.lmbda = lambda; F.u = u;
  //   F.B = B; F.T = T;

  //   // Create Jacobian dF = F' (for use in nonlinear solver).
  //   HyperElasticity::JacobianForm J(V, V);
  //   J.mu = mu; J.lmbda = lambda; J.u = u;

  // // Now, we have specified the variational forms and can consider the
  // // solution of the variational problem.
  // //
  // // .. code-block:: cpp

  //   // Solve nonlinear variational problem F(u; v) = 0
  //   solve(F == 0, *u, bcs, J);

  // // Finally, the solution ``u`` is saved to a file named
  // // ``displacement.pvd`` in VTK format.
  // //
  // // .. code-block:: cpp

  //   // Save solution in VTK format
  //   File file("displacement.pvd");
  //   file << *u;

  return 0;
}
