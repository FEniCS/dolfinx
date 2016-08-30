// Biharmonic equation (C++)
// =========================
// 
// This demo illustrates how to:
// 
// * Solve a linear partial differential equation
// * Use a discontinuous Galerkin method
// * Solve a fourth-order differential equation
// 
// The solution for :math:`u` in this demo will look as follows:
// 
// .. image:: ../biharmonic_u.png
//     :scale: 75 %
// 
// Equation and problem definition
// -------------------------------
// 
// The biharmonic equation is a fourth-order elliptic equation. On the
// domain :math:`\Omega \subset \mathbb{R}^{d}`, :math:`1 \le d \le 3`,
// it reads
// 
// .. math::
//    \nabla^{4} u = f \quad {\rm in} \ \Omega,
// 
// where :math:`\nabla^{4} \equiv \nabla^{2} \nabla^{2}` is the
// biharmonic operator and :math:`f` is a prescribed source term. To
// formulate a complete boundary value problem, the biharmonic equation
// must be complemented by suitable boundary conditions.
// 
// Multiplying the biharmonic equation by a test function and integrating
// by parts twice leads to a problem second-order derivatives, which
// would requires :math:`H^{2}` conforming (roughly :math:`C^{1}`
// continuous) basis functions.  To solve the biharmonic equation using
// Lagrange finite element basis functions, the biharmonic equation can
// be split into two second-order equations (see the Mixed Poisson demo
// for a mixed method for the Poisson equation), or a variational
// formulation can be constructed that imposes weak continuity of normal
// derivatives between finite element cells.  The demo uses a
// discontinuous Galerkin approach to impose continuity of the normal
// derivative weakly.
// 
// Consider a triangulation :math:`\mathcal{T}` of the domain
// :math:`\Omega`, where the set of interior facets is denoted by
// :math:`\mathcal{E}_h^{\rm int}`.  Functions evaluated on opposite
// sides of a facet are indicated by the subscripts ':math:`+`' and
// ':math:`-`'.  Using the standard continuous Lagrange finite element
// space
// 
// .. math::
//     V = \left\{v \in H^{1}_{0}(\Omega)\,:\, v \in P_{k}(K) \ \forall \ K \in \mathcal{T} \right\}
// 
// and considering the boundary conditions
// 
// .. math::
//    u            &= 0 \quad {\rm on} \ \partial\Omega \\
//    \nabla^{2} u &= 0 \quad {\rm on} \ \partial\Omega
// 
// a weak formulation of the biharmonic problem reads: find :math:`u \in
// V` such that
// 
// .. math::
//   a(u,v)=L(v) \quad \forall \ v \in V,
// 
// where the bilinear form is
// 
// .. math::
//    a(u, v) = \sum_{K \in \mathcal{T}} \int_{K} \nabla^{2} u \nabla^{2} v \, {\rm d}x \
//   +\sum_{E \in \mathcal{E}_h^{\rm int}}\left(\int_{E} \frac{\alpha}{h_E} [\!\![ \nabla u ]\!\!] [\!\![ \nabla v ]\!\!] \, {\rm d}s
//   - \int_{E} \left<\nabla^{2} u \right>[\!\![ \nabla v ]\!\!]  \, {\rm d}s
//   - \int_{E} [\!\![ \nabla u ]\!\!]  \left<\nabla^{2} v \right>  \, {\rm d}s\right)
// 
// and the linear form is
// 
// .. math::
//   L(v) = \int_{\Omega} fv \, {\rm d}x
// 
// Furthermore, :math:`\left< u \right> = \frac{1}{2} (u_{+} + u_{-})`, :math:`[\!\![
// w ]\!\!]  = w_{+} \cdot n_{+} + w_{-} \cdot n_{-}`, :math:`\alpha \ge
// 0` is a penalty parameter and :math:`h_E` is a measure of the cell size.
// 
// The input parameters for this demo are defined as follows:
// 
// * :math:`\Omega = [0,1] \times [0,1]` (a unit square)
// * :math:`\alpha = 8.0` (penalty parameter)
// * :math:`f = 4.0 \pi^4\sin(\pi x)\sin(\pi y)` (source term)
// 
// 
// Implementation
// --------------
// 
// The implementation is split in two files, a form file containing the
// definition of the variational forms expressed in UFL and the solver
// which is implemented in a C++ file.
// 
// Running this demo requires the files: :download:`main.cpp`,
// :download:`Biharmonic.ufl` and :download:`CMakeLists.txt`.
// 
// UFL form file
// ^^^^^^^^^^^^^
// 
// The UFL file is implemented in :download:`Biharmonic.ufl`, and the
// explanation of the UFL file can be found at :doc:`here
// <Biharmonic.ufl>`.
// 
// 
// C++ program
// ^^^^^^^^^^^
// 
// The DOLFIN interface and the code generated from the UFL input is
// included, and the DOLFIN namespace is used:
// 
// .. code-block:: c++
// 
//   #include <dolfin.h>
//   #include "Biharmonic.h"
// 
//   using namespace dolfin;
// 
// A class ``Source`` is defined for the function :math:`f`, with the
// function ``Expression::eval`` overloaded:
// 
// .. code-block:: c++
// 
//   // Source term
//   class Source : public Expression
//   {
//   public:
// 
//     void eval(Array<double>& values, const Array<double>& x) const
//     {
//       values[0] = 4.0*std::pow(DOLFIN_PI, 4)*
//         std::sin(DOLFIN_PI*x[0])*std::sin(DOLFIN_PI*x[1]);
//     }
// 
//   };
// 
// A boundary subdomain is defined, which in this case is the entire
// boundary:
// 
// .. code-block:: c++
// 
//   // Sub domain for Dirichlet boundary condition
//   class DirichletBoundary : public SubDomain
//   {
//     bool inside(const Array<double>& x, bool on_boundary) const
//     { return on_boundary; }
//   };
// 
// The main part of the program is begun, and a mesh is created with 32
// vertices in each direction:
// 
// .. code-block:: c++
// 
//   int main()
//   {
//     // Make mesh ghosted for evaluation of DG terms
//     parameters["ghost_mode"] = "shared_facet";
// 
//     // Create mesh
//     auto mesh = std::make_shared<UnitSquareMesh>(32, 32);
// 
// The source function, a function for the cell size and the penalty term
// are declared:
// 
// .. code-block:: c++
// 
//     // Create functions
//     auto f = std::make_shared<Source>();
//     auto alpha = std::make_shared<Constant>(8.0);
// 
// A function space object, which is defined in the generated code, is
// created:
// 
// .. code-block:: c++
// 
//     // Create function space
//     auto V = std::make_shared<Biharmonic::FunctionSpace>(mesh);
// 
// The Dirichlet boundary condition on :math:`u` is constructed by
// defining a :cpp:class:`Constant` which is equal to zero, defining the
// boundary (``DirichletBoundary``), and using these, together with
// ``V``, to create ``bc``:
// 
// .. code-block:: c++
// 
//     // Define boundary condition
//     auto u0 = std::make_shared<Constant>(0.0);
//     auto boundary = std::make_shared<DirichletBoundary>();
//     DirichletBC bc(V, u0, boundary);
// 
// Using the function space ``V``, the bilinear and linear forms are
// created, and function are attached:
// 
// .. code-block:: c++
// 
//     // Define variational problem
//     Biharmonic::BilinearForm a(V, V);
//     Biharmonic::LinearForm L(V);
//     a.alpha = alpha; L.f = f;
// 
// A :cpp:class:`Function` is created to hold the solution and the
// problem is solved:
// 
// .. code-block:: c++
// 
//     // Compute solution
//     Function u(V);
//     solve(a == L, u, bc);
// 
// The solution is then written to a file in VTK format and plotted to
// the screen:
// 
// .. code-block:: c++
// 
//     // Save solution in VTK format
//     File file("biharmonic.pvd");
//     file << u;
// 
//     // Plot solution
//     plot(u);
//     interactive();
// 
//     return 0;
//   }
