// Poisson equation (C++)
// ======================
//
// This demo illustrates how to:
//
// * Solve a linear partial differential equation
// * Create and apply Dirichlet boundary conditions
// * Define Expressions
// * Define a FunctionSpace
// * Create a SubDomain
//
// The solution for :math:`u` in this demo will look as follows:
//
// .. image:: ../poisson_u.png
//     :scale: 75 %
//
//
// Equation and problem definition
// -------------------------------
//
// The Poisson equation is the canonical elliptic partial differential
// equation.  For a domain :math:`\Omega \subset \mathbb{R}^n` with
// boundary :math:`\partial \Omega = \Gamma_{D} \cup \Gamma_{N}`, the
// Poisson equation with particular boundary conditions reads:
//
// .. math::
//    - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
//      u &= 0 \quad {\rm on} \ \Gamma_{D}, \\
//      \nabla u \cdot n &= g \quad {\rm on} \ \Gamma_{N}. \\
//
// Here, :math:`f` and :math:`g` are input data and :math:`n` denotes the
// outward directed boundary normal. The most standard variational form
// of Poisson equation reads: find :math:`u \in V` such that
//
// .. math::
//    a(u, v) = L(v) \quad \forall \ v \in V,
//
// where :math:`V` is a suitable function space and
//
// .. math::
//    a(u, v) &= \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x, \\
//    L(v)    &= \int_{\Omega} f v \, {\rm d} x
//    + \int_{\Gamma_{N}} g v \, {\rm d} s.
//
// The expression :math:`a(u, v)` is the bilinear form and :math:`L(v)`
// is the linear form. It is assumed that all functions in :math:`V`
// satisfy the Dirichlet boundary conditions (:math:`u = 0 \ {\rm on} \
// \Gamma_{D}`).
//
// In this demo, we shall consider the following definitions of the input
// functions, the domain, and the boundaries:
//
// * :math:`\Omega = [0,1] \times [0,1]` (a unit square)
// * :math:`\Gamma_{D} = \{(0, y) \cup (1, y) \subset \partial \Omega\}`
// (Dirichlet boundary)
// * :math:`\Gamma_{N} = \{(x, 0) \cup (x, 1) \subset \partial \Omega\}`
// (Neumann boundary)
// * :math:`g = \sin(5x)` (normal derivative)
// * :math:`f = 10\exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)` (source term)
//
//
// Implementation
// --------------
//
// The implementation is split in two files: a form file containing the
// definition of the variational forms expressed in UFL and a C++ file
// containing the actual solver.
//
// Running this demo requires the files: :download:`main.cpp`,
// :download:`Poisson.ufl` and :download:`CMakeLists.txt`.
//
//
// UFL form file
// ^^^^^^^^^^^^^
//
// The UFL file is implemented in :download:`Poisson.ufl`, and the
// explanation of the UFL file can be found at :doc:`here <Poisson.ufl>`.
//
//
// C++ program
// ^^^^^^^^^^^
//
// The main solver is implemented in the :download:`main.cpp` file.
//
// At the top we include the DOLFIN header file and the generated header
// file "Poisson.h" containing the variational forms for the Poisson
// equation.  For convenience we also include the DOLFIN namespace.
//
// .. code-block:: cpp

#include "poisson.h"
#include <Eigen/Dense>
#include <cfloat>
#include <dolfin.h>
#include <dolfin/mesh/Ordering.h>

using namespace dolfin;

// Then follows the definition of the coefficient functions (for
// :math:`f` and :math:`g`), which are derived from the
// :cpp:class:`Expression` class in DOLFIN
//
// .. code-block:: cpp

class P1Element
{
public:
  static int evaluate_basis_derivs(double* ref_vals, int order, int npoints,
                                   const double* X)
  {
    if (order == 0)
    {
      Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
          _X(X, npoints, 2);

      Eigen::Map<
          Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          _ref_vals(ref_vals, npoints, 3);

      _ref_vals.col(0) = 1.0 - _X.col(0) - _X.col(1);
      _ref_vals.col(1) = _X.col(0);
      _ref_vals.col(2) = _X.col(1);

      return 0;
    }
    else if (order == 1)
    {
      // NB - not used by this demo - needs checking
      Eigen::Map<
          Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          _ref_vals(ref_vals, npoints, 6);

      Eigen::Array<double, 1, 6> derivs;
      derivs << -1, 1, 0, -1, 0, 1;

      _ref_vals = derivs.replicate(npoints, 1);

      return 0;
    }
    return -1;
  }

  static int transform_basis_derivs(double* values, int order, int num_points,
                                    const double* reference_values,
                                    const double* X, const double* J,
                                    const double* detJ, const double* K,
                                    int cell_orientation)
  {
    if (order == 0)
    {
      Eigen::Map<
          Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          _values(values, num_points, 3);
      Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
          _reference_values(reference_values, num_points, 3);
      _values = _reference_values;
      return 0;
    }
    else if (order == 1)
    {
      // NB - not used in this demo - needs checking
      Eigen::Map<
          Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          _values(values, num_points, 6);
      Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
          _reference_values(reference_values, num_points, 6);

      for (int ip = 0; ip < num_points; ++ip)
      {
        double transform[2][2];

        transform[0][0] = K[4 * ip];
        transform[0][1] = K[4 * ip + 2];
        transform[1][0] = K[4 * ip + 1];
        transform[1][1] = K[4 * ip + 3];

        for (int d = 0; d < 3; ++d)
        {
          // Using affine transform to map values back to the physical
          // element.
          // Mapping derivatives back to the physical element
          _values(ip, 2 * d)
              = transform[0][0] * _reference_values(ip, 2 * d)
                + transform[0][1] * _reference_values(ip, 2 * d + 1);

          _values(ip, 2 * d + 1)
              = transform[1][0] * _reference_values(ip, 2 * d)
                + transform[1][1] * _reference_values(ip, 2 * d + 1);
        }
      }
      return 0;
    }

    return -1;
  }

  static int transform_values(ufc_scalar_t* reference_values,
                              const ufc_scalar_t* physical_values,
                              const double* coordinate_dofs,
                              int cell_orientation,
                              const ufc_coordinate_mapping* cm)
  {
    reference_values[0] = physical_values[0];
    reference_values[1] = physical_values[1];
    reference_values[2] = physical_values[2];
    return 0;
  }
};

class LinearTriangleCoordinateMap
{
public:
  static void compute_reference_geometry(double* X, double* J, double* detJ,
                                         double* K, int num_points,
                                         const double* x,
                                         const double* coordinate_dofs, int q)
  {

    J[0] = -coordinate_dofs[0] + coordinate_dofs[2];
    J[1] = -coordinate_dofs[0] + coordinate_dofs[4];
    J[2] = -coordinate_dofs[1] + coordinate_dofs[3];
    J[3] = -coordinate_dofs[1] + coordinate_dofs[5];

    detJ[0] = J[0] * J[3] - J[1] * J[2];

    K[0] = J[3] / detJ[0];
    K[1] = -J[1] / detJ[0];
    K[2] = -J[2] / detJ[0];
    K[3] = J[0] / detJ[0];

    Eigen::Map<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        _X(X, num_points, 2);
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        _x(x, num_points, 2);

    _X.col(0) = K[0] * (_x.col(0) - coordinate_dofs[0])
                + K[1] * (_x.col(1) - coordinate_dofs[1]);
    _X.col(1) = K[2] * (_x.col(0) - coordinate_dofs[0])
                + K[3] * (_x.col(1) - coordinate_dofs[1]);
  }

  static void compute_physical_coordinates(double* x, int nrows,
                                           const double* X,
                                           const double* coordinate_dofs)
  {
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        _X(X, nrows, 2);
    Eigen::Map<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        _x(x, nrows, 2);

    _x.col(0) = coordinate_dofs[0]
                + (coordinate_dofs[2] - coordinate_dofs[0]) * _X.col(0)
                + (coordinate_dofs[4] - coordinate_dofs[0]) * _X.col(1);
    _x.col(1) = coordinate_dofs[1]
                + (coordinate_dofs[3] - coordinate_dofs[1]) * _X.col(0)
                + (coordinate_dofs[4] - coordinate_dofs[1]) * _X.col(1);
  }
};

void tabulate_tensor_bilinear(ufc_scalar_t* A, const ufc_scalar_t* w,
                              const double* coordinate_dofs,
                              int cell_orientation)
{
  const double J_c0 = -coordinate_dofs[0] + coordinate_dofs[2];
  const double J_c3 = -coordinate_dofs[1] + coordinate_dofs[5];
  const double J_c1 = -coordinate_dofs[0] + coordinate_dofs[4];
  const double J_c2 = -coordinate_dofs[1] + coordinate_dofs[3];
  const double v_cell = 0.5 / std::fabs(J_c0 * J_c3 - J_c1 * J_c2);

  // Local tensor
  A[8] = (J_c0 * J_c0 + J_c2 * J_c2) * v_cell;
  A[5] = -(J_c0 * J_c1 + J_c2 * J_c3) * v_cell;
  A[4] = (J_c1 * J_c1 + J_c3 * J_c3) * v_cell;

  A[1] = -A[5] - A[4];
  A[2] = -A[5] - A[8];
  A[0] = -A[1] - A[2];

  A[3] = A[1];
  A[6] = A[2];
  A[7] = A[5];
}

void tabulate_tensor_linear(ufc_scalar_t* A, const ufc_scalar_t* w,
                            const double* coordinate_dofs, int cell_orientation)
{
  // Quadrature rules
  alignas(32) static const ufc_scalar_t weights3[3]
      = {0.1666666666666667, 0.1666666666666667, 0.1666666666666667};
  // Precomputed values of basis functions and precomputations
  // FE* dimensions: [entities][points][dofs]
  // PI* dimensions: [entities][dofs][dofs] or [entities][dofs]
  // PM* dimensions: [entities][dofs][dofs]
  alignas(32) static const ufc_scalar_t FE3_C0_Q3[1][3][3]
      = {{{0.6666666666666669, 0.1666666666666666, 0.1666666666666667},
          {0.1666666666666667, 0.1666666666666666, 0.6666666666666665},
          {0.1666666666666667, 0.6666666666666666, 0.1666666666666666}}};
  alignas(32) static const ufc_scalar_t FE4_C0_D01_Q3[1][1][2]
      = {{{-1.0, 1.0}}};
  // Unstructured piecewise computations
  const double J_c0 = -coordinate_dofs[0] + coordinate_dofs[2];
  const double J_c3 = -coordinate_dofs[1] + coordinate_dofs[5];
  const double J_c1 = -coordinate_dofs[0] + coordinate_dofs[4];
  const double J_c2 = -coordinate_dofs[1] + coordinate_dofs[3];

  alignas(32) ufc_scalar_t sp[4];
  sp[0] = J_c0 * J_c3;
  sp[1] = J_c1 * J_c2;
  sp[2] = sp[0] - sp[1];
  sp[3] = std::fabs(sp[2]);

  // UFLACS block mode: full
  alignas(32) ufc_scalar_t BF0[3] = {0};
  for (int iq = 0; iq < 3; ++iq)
  {
    // Quadrature loop body setup (num_points=3)
    // Unstructured varying computations for num_points=3
    ufc_scalar_t w0 = 0.0;
    for (int ic = 0; ic < 3; ++ic)
      w0 += w[ic] * FE3_C0_Q3[0][iq][ic];
    alignas(32) ufc_scalar_t sv3[1];
    sv3[0] = sp[3] * w0;
    // UFLACS block mode: full
    const ufc_scalar_t fw0 = sv3[0] * weights3[iq];
    for (int i = 0; i < 3; ++i)
      BF0[i] += fw0 * FE3_C0_Q3[0][iq][i];
  }
  // UFLACS block mode: preintegrated
  for (int k = 0; k < 3; ++k)
    A[k] = 0.0;
  for (int i = 0; i < 3; ++i)
    A[i] += BF0[i];
}

void tabulate_tensor_linear_exterior_facet(ufc_scalar_t* A,
                                           const ufc_scalar_t* w,
                                           const double* coordinate_dofs,
                                           int facet, int cell_orientation)
{
  // Quadrature rules
  alignas(32) static const ufc_scalar_t weights2[2] = {0.5, 0.5};
  // Precomputed values of basis functions and precomputations
  // FE* dimensions: [entities][points][dofs]
  // PI* dimensions: [entities][dofs][dofs] or [entities][dofs]
  // PM* dimensions: [entities][dofs][dofs]
  alignas(32) static const ufc_scalar_t FE3_C0_F_Q2[3][2][3]
      = {{{0.0, 0.7886751345948129, 0.2113248654051871},
          {0.0, 0.2113248654051872, 0.7886751345948129}},
         {{0.7886751345948129, 0.0, 0.2113248654051871},
          {0.2113248654051872, 0.0, 0.7886751345948129}},
         {{0.7886751345948129, 0.2113248654051871, 0.0},
          {0.2113248654051871, 0.7886751345948129, 0.0}}};
  alignas(32) static const ufc_scalar_t FE4_C0_D01_F_Q2[1][1][2]
      = {{{-1.0, 1.0}}};
  // Unstructured piecewise computations
  const double J_c0 = coordinate_dofs[0] * FE4_C0_D01_F_Q2[0][0][0]
                      + coordinate_dofs[2] * FE4_C0_D01_F_Q2[0][0][1];
  const double J_c1 = coordinate_dofs[0] * FE4_C0_D01_F_Q2[0][0][0]
                      + coordinate_dofs[4] * FE4_C0_D01_F_Q2[0][0][1];
  const double J_c2 = coordinate_dofs[1] * FE4_C0_D01_F_Q2[0][0][0]
                      + coordinate_dofs[3] * FE4_C0_D01_F_Q2[0][0][1];
  const double J_c3 = coordinate_dofs[1] * FE4_C0_D01_F_Q2[0][0][0]
                      + coordinate_dofs[5] * FE4_C0_D01_F_Q2[0][0][1];
  alignas(32) ufc_scalar_t sp[10];
  sp[0] = J_c0 * triangle_reference_facet_jacobian[facet][0][0];
  sp[1] = J_c1 * triangle_reference_facet_jacobian[facet][1][0];
  sp[2] = sp[0] + sp[1];
  sp[3] = sp[2] * sp[2];
  sp[4] = triangle_reference_facet_jacobian[facet][0][0] * J_c2;
  sp[5] = triangle_reference_facet_jacobian[facet][1][0] * J_c3;
  sp[6] = sp[4] + sp[5];
  sp[7] = sp[6] * sp[6];
  sp[8] = sp[3] + sp[7];
  sp[9] = sqrt(sp[8]);
  // UFLACS block mode: full
  alignas(32) ufc_scalar_t BF0[3] = {0};
  for (int iq = 0; iq < 2; ++iq)
  {
    // Quadrature loop body setup (num_points=2)
    // Unstructured varying computations for num_points=2
    ufc_scalar_t w1 = 0.0;
    for (int ic = 0; ic < 3; ++ic)
      w1 += w[3 + ic] * FE3_C0_F_Q2[facet][iq][ic];
    alignas(32) ufc_scalar_t sv2[1];
    sv2[0] = sp[9] * w1;
    // UFLACS block mode: full
    const ufc_scalar_t fw0 = sv2[0] * weights2[iq];
    for (int i = 0; i < 3; ++i)
      BF0[i] += fw0 * FE3_C0_F_Q2[facet][iq][i];
  }
  // UFLACS block mode: preintegrated
  for (int k = 0; k < 3; ++k)
    A[k] = 0.0;
  for (int i = 0; i < 3; ++i)
    A[i] += BF0[i];
}

// Source term (right-hand side)
class Source : public function::Expression
{
public:
  Source() : function::Expression({}) {}

  void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                values,
            Eigen::Ref<const EigenRowArrayXXd> x) const
  {
    for (unsigned int i = 0; i != x.rows(); ++i)
    {
      double dx = x(i, 0) - 0.5;
      double dy = x(i, 1) - 0.5;
      values(i, 0) = 10 * exp(-(dx * dx + dy * dy) / 0.02);
    }
  }
};

// Normal derivative (Neumann boundary condition)
class dUdN : public function::Expression
{
public:
  dUdN() : function::Expression({}) {}

  void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                values,
            Eigen::Ref<const EigenRowArrayXXd> x) const
  {
    for (unsigned int i = 0; i != x.rows(); ++i)
      values(i, 0) = sin(5 * x(i, 0));
  }
};

// The ``DirichletBoundary`` is derived from the :cpp:class:`SubDomain`
// class and defines the part of the boundary to which the Dirichlet
// boundary condition should be applied.
//
// .. code-block:: cpp

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public mesh::SubDomain
{
  EigenArrayXb inside(Eigen::Ref<const EigenRowArrayXXd> x,
                      bool on_boundary) const
  {
    EigenArrayXb result(x.rows());
    for (unsigned int i = 0; i != x.rows(); ++i)
      result[i] = (x(i, 0) < DBL_EPSILON or x(i, 0) > 1.0 - DBL_EPSILON);
    return result;
  }
};

// Inside the ``main`` function, we begin by defining a mesh of the
// domain. As the unit square is a very standard domain, we can use a
// built-in mesh provided by the :cpp:class:`UnitSquareMesh` factory. In
// order to create a mesh consisting of 32 x 32 squares with each square
// divided into two triangles, and the finite element space (specified in
// the form file) defined relative to this mesh, we do as follows
//
// .. code-block:: cpp

int main(int argc, char* argv[])
{
  common::SubSystemsManager::init_logging(argc, argv);
  common::SubSystemsManager::init_petsc(argc, argv);

  // Create mesh and function space
  std::array<geometry::Point, 2> pt
      = {geometry::Point(0., 0.), geometry::Point(1., 1.)};
  auto mesh = std::make_shared<mesh::Mesh>(generation::RectangleMesh::create(
      MPI_COMM_WORLD, pt, {{320, 320}}, mesh::CellType::Type::triangle,
      mesh::GhostMode::none));

  mesh::Ordering::order_simplex(*mesh);

  // Custom DofMap
  auto layout
      = std::shared_ptr<fem::ElementDofLayout>(new fem::ElementDofLayout(
          1, {{{0}, {1}, {2}}, {{}, {}, {}}, {{}}}, {}, {}, mesh->type()));
  auto dm = std::make_shared<fem::DofMap>(layout, *mesh);

  // Custom FiniteElement
  Eigen::Array<double, 3, 2, Eigen::RowMajor> refX;
  refX << 0, 0, 1, 0, 0, 1;
  auto fe = std::shared_ptr<fem::FiniteElement>(new fem::FiniteElement(
      "FiniteElement('Lagrange', triangle, 1)", "Lagrange", 2, 3, {}, refX, 1,
      1, 1, P1Element::evaluate_basis_derivs, P1Element::transform_basis_derivs,
      P1Element::transform_values));

  // Create FunctionSpace from Mesh, FiniteElement and DofMap
  auto V = std::make_shared<function::FunctionSpace>(mesh, fe, dm);

  // Now, the Dirichlet boundary condition (:math:`u = 0`) can be created
  // using the class :cpp:class:`DirichletBC`. A :cpp:class:`DirichletBC`
  // takes three arguments: the function space the boundary condition
  // applies to, the value of the boundary condition, and the part of the
  // boundary on which the condition applies. In our example, the function
  // space is ``V``, the value of the boundary condition (0.0) can
  // represented using a :cpp:class:`Function`, and the Dirichlet boundary
  // is defined by the class :cpp:class:`DirichletBoundary` listed
  // above. The definition of the Dirichlet boundary condition then looks
  // as follows:
  //
  // .. code-block:: cpp

  // FIXME: zero function and make sure ghosts are updated
  // Define boundary condition
  auto u0 = std::make_shared<function::Function>(V);
  DirichletBoundary boundary;

  std::vector<std::shared_ptr<const fem::DirichletBC>> bc
      = {std::make_shared<fem::DirichletBC>(V, u0, boundary)};

  // Next, we define the variational formulation by initializing the
  // bilinear and linear forms (:math:`a`, :math:`L`) using the previously
  // defined :cpp:class:`FunctionSpace` ``V``.  Then we can create the
  // source and boundary flux term (:math:`f`, :math:`g`) and attach these
  // to the linear form.
  //
  // .. code-block:: cpp

  // Define variational forms
  auto a = std::shared_ptr<fem::Form>(new fem::Form({V, V}));
  a->register_tabulate_tensor_cell(-1, tabulate_tensor_bilinear);

  auto L = std::shared_ptr<fem::Form>(new fem::Form({V}));
  L->register_tabulate_tensor_cell(-1, tabulate_tensor_linear);
  L->register_tabulate_tensor_exterior_facet(
      -1, tabulate_tensor_linear_exterior_facet);

  auto f_expr = Source();
  auto g_expr = dUdN();

  auto f = std::make_shared<function::Function>(V);
  auto g = std::make_shared<function::Function>(V);

  // Attach 'coordinate mapping' to mesh
  auto cmap = std::make_shared<dolfin::fem::CoordinateMapping>(
      dolfin::CellType::triangle, 2, 2, "Linear Triangle Coordinate Map",
      LinearTriangleCoordinateMap::compute_physical_coordinates,
      LinearTriangleCoordinateMap::compute_reference_geometry);

  mesh->geometry().coord_mapping = cmap;

  f->interpolate(f_expr);
  g->interpolate(g_expr);
  L->set_coefficients({{0, f}, {1, g}});

  // Now, we have specified the variational forms and can consider the
  // solution of the variational problem. First, we need to define a
  // :cpp:class:`Function` ``u`` to store the solution. (Upon
  // initialization, it is simply set to the zero function.) Next, we can
  // call the ``solve`` function with the arguments ``a == L``, ``u`` and
  // ``bc`` as follows:
  //
  // .. code-block:: cpp

  // Compute solution
  function::Function u(V);
  la::PETScMatrix A = fem::create_matrix(*a);
  la::PETScVector b(*L->function_space(0)->dofmap()->index_map());

  MatZeroEntries(A.mat());
  dolfin::fem::assemble_matrix(A.mat(), *a, bc);
  MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

  VecSet(b.vec(), 0.0);
  VecGhostUpdateBegin(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
  dolfin::fem::assemble_vector(b.vec(), *L);
  dolfin::fem::apply_lifting(b.vec(), {a}, {{bc}}, {}, 1.0);
  VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  dolfin::fem::set_bc(b.vec(), bc, nullptr);

  la::PETScKrylovSolver lu(MPI_COMM_WORLD);
  la::PETScOptions::set("ksp_type", "preonly");
  la::PETScOptions::set("pc_type", "lu");
  lu.set_from_options();

  lu.set_operator(A.mat());
  lu.solve(u.vector().vec(), b.vec());

  // The function ``u`` will be modified during the call to solve. A
  // :cpp:class:`Function` can be saved to a file. Here, we output the
  // solution to a ``VTK`` file (specified using the suffix ``.pvd``) for
  // visualisation in an external program such as Paraview.
  //
  // .. code-block:: cpp

  // Save solution in VTK format
  io::VTKFile file("poisson.pvd");
  file.write(u);

  return 0;
}
