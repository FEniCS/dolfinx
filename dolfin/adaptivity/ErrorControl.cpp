// Copyright (C) 2010 Marie E. Rognes
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2011.
//
// First added:  2010-09-16
// Last changed: 2011-03-23

#include <armadillo>
#include <boost/scoped_ptr.hpp>

#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/Hierarchical.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/fem/BoundaryCondition.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/UFC.h>
#include <dolfin/fem/LinearVariationalProblem.h>
#include <dolfin/fem/LinearVariationalSolver.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/SubSpace.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/SpecialFacetFunction.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/solve.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>

#include "LocalAssembler.h"
#include "ErrorControl.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ErrorControl::ErrorControl(boost::shared_ptr<Form> a_star,
                           boost::shared_ptr<Form> L_star,
                           boost::shared_ptr<Form> residual,
                           boost::shared_ptr<Form> a_R_T,
                           boost::shared_ptr<Form> L_R_T,
                           boost::shared_ptr<Form> a_R_dT,
                           boost::shared_ptr<Form> L_R_dT,
                           boost::shared_ptr<Form> eta_T,
                           bool is_linear)
  : Hierarchical<ErrorControl>(*this)
{
  // Assign input
  _a_star = a_star;
  _L_star = L_star;
  _residual = residual;
  _a_R_T = a_R_T;
  _L_R_T = L_R_T;
  _a_R_dT = a_R_dT;
  _L_R_dT = L_R_dT;
  _eta_T = eta_T;
  _is_linear = is_linear;

  // Extract and store additional function spaces
  const uint improved_dual = _residual->num_coefficients() - 1;
  const Function& e_tmp = dynamic_cast<const Function&>(*_residual->coefficient(improved_dual));
  _E = e_tmp.function_space();

  const Function& bubble = dynamic_cast<const Function&>(*_a_R_T->coefficient(0));
  _B = bubble.function_space();
  _cell_bubble.reset(new Function(_B));
  dolfin_assert(_cell_bubble->vector());
  *(_cell_bubble->vector()) = 1.0;

  const Function& cone = dynamic_cast<const Function&>(*_a_R_dT->coefficient(0));
  _C = cone.function_space();
  _cell_cone.reset(new Function(_C));

  // Set parameters
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
double ErrorControl::estimate_error(const Function& u,
  const std::vector<boost::shared_ptr<const BoundaryCondition> > bcs)
{
  // Compute discrete dual approximation
  dolfin_assert(_a_star);
  Function z_h(_a_star->function_space(1));
  compute_dual(z_h, bcs);

  // Compute extrapolation of discrete dual
  compute_extrapolation(z_h, bcs);

  // Extract number of coefficients in residual
  dolfin_assert(_residual);
  const uint num_coeffs = _residual->num_coefficients();

  // Attach improved dual approximation to residual
  _residual->set_coefficient(num_coeffs - 1, _Ez_h);

  // Attach primal approximation if linear problem (already attached
  // otherwise).
  if (_is_linear)
  {
    boost::shared_ptr<const GenericFunction> _u(&u, NoDeleter());
    _residual->set_coefficient(num_coeffs - 2, _u);
  }

  // Assemble error estimate
  info("Assembling error estimate.");
  const double error_estimate = assemble(*_residual);

  // Return estimate
  return error_estimate;
}
//-----------------------------------------------------------------------------
void ErrorControl::compute_dual(Function& z,
   const std::vector<boost::shared_ptr<const BoundaryCondition> > bcs)
{
  info("Solving dual problem.");

  // Create dual boundary conditions by homogenizing
  std::vector<boost::shared_ptr<const BoundaryCondition> > dual_bcs;
  for (std::size_t i = 0; i < bcs.size(); i++)
  {
    // Only handle DirichletBCs
    const DirichletBC* bc_ptr = dynamic_cast<const DirichletBC*>(bcs[i].get());
    if (!bc_ptr)
      dolfin_error("ErrorControl.cpp",
                   "compute dual solution",
                   "Only DirichletBCs can be homogenized (for now)");

    // Create shared_ptr to boundary condition
    boost::shared_ptr<DirichletBC> dual_bc_ptr(new DirichletBC(*bc_ptr));

    // Run homogenize
    dual_bc_ptr->homogenize();

    // Plug pointer into into vector
    dual_bcs.push_back(dual_bc_ptr);
  }

  // Create shared_ptr to dual solution (FIXME: missing interface ...)
  boost::shared_ptr<Function> dual(reference_to_no_delete_pointer(z));

  // Solve dual problem
  LinearVariationalProblem dual_problem(_a_star, _L_star, dual, dual_bcs);
  LinearVariationalSolver solver(dual_problem);
  solver.parameters.update(parameters("dual_variational_solver"));
  solver.solve();
}
//-----------------------------------------------------------------------------
void ErrorControl::compute_extrapolation(const Function& z,
   const std::vector<boost::shared_ptr<const BoundaryCondition> > bcs)
{
  info("Extrapolating dual solution.");

  // Extrapolate
  dolfin_assert(_E);
  _Ez_h.reset(new Function(_E));
  _Ez_h->extrapolate(z);

  // Apply appropriate boundary conditions to extrapolation
  apply_bcs_to_extrapolation(bcs);
}
//-----------------------------------------------------------------------------
void ErrorControl::compute_indicators(MeshFunction<double>& indicators,
                                      const Function& u)
{
  // Create Function for the strong cell residual (R_T)
  dolfin_assert(_a_R_T);
  _R_T.reset(new Function(_a_R_T->function_space(1)));

  // Create SpecialFacetFunction for the strong facet residual (R_dT)
  dolfin_assert(_a_R_dT);
  std::vector<Function> f_e;
  for (uint i = 0; i <= _R_T->geometric_dimension(); i++)
    f_e.push_back(Function(_a_R_dT->function_space(1)));

  if (f_e[0].value_rank() == 0)
    _R_dT.reset(new SpecialFacetFunction(f_e));
  else if (f_e[0].value_rank() == 1)
    _R_dT.reset(new SpecialFacetFunction(f_e, f_e[0].value_dimension(0)));
  else
  {
    _R_dT.reset(new SpecialFacetFunction(f_e, f_e[0].value_dimension(0)));
    dolfin_error("ErrorControl.cpp",
                 "compute error indicators",
                 "Not implemented for tensor-valued functions");
  }

  // Compute residual representation
  residual_representation(*_R_T, *_R_dT, u);

  // Interpolate dual extrapolation into primal test (dual trial space)
  dolfin_assert(_a_star);
  dolfin_assert(_Ez_h);
  _Pi_E_z_h.reset(new Function(_a_star->function_space(1)));
  _Pi_E_z_h->interpolate(*_Ez_h);

  // Attach coefficients to error indicator form
  dolfin_assert(_eta_T);
  _eta_T->set_coefficient(0, _Ez_h);
  _eta_T->set_coefficient(1, _R_T);
  _eta_T->set_coefficient(2, _R_dT);
  _eta_T->set_coefficient(3, _Pi_E_z_h);

  // Assemble error indicator form
  Vector x(indicators.mesh().num_cells());
  assemble(x, *_eta_T);

  // Take absolute value of indicators
  x.abs();

  // Convert vector x to indicator mesh function
  dolfin_assert(_eta_T->function_space(0));
  dolfin_assert(_eta_T->function_space(0)->dofmap());
  const GenericDofMap& dofmap(*_eta_T->function_space(0)->dofmap());
  const Mesh& mesh(indicators.mesh());

  // Convert DG_0 vector to mesh function over cells
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::vector<std::size_t>& dofs = dofmap.cell_dofs(cell->index());
    dolfin_assert(dofs.size() == 1);
    indicators[cell->index()] = x[dofs[0]];
  }
}

//-----------------------------------------------------------------------------
void ErrorControl::residual_representation(Function& R_T,
                                           SpecialFacetFunction& R_dT,
                                           const Function& u)
{
  begin("Computing residual representation.");

  // Compute cell residual
  Timer timer("Computation of residual representation");
  compute_cell_residual(R_T, u);

  // Compute facet residual
  compute_facet_residual(R_dT, u, R_T);
  timer.stop();

  end();
}
//-----------------------------------------------------------------------------
void ErrorControl::compute_cell_residual(Function& R_T, const Function& u)
{
  begin("Computing cell residual representation.");

  dolfin_assert(_a_R_T);
  dolfin_assert(_L_R_T);
  dolfin_assert(_cell_bubble);

  // Attach cell bubble to _a_R_T and _L_R_T
  const uint num_coeffs = _L_R_T->num_coefficients();
  _a_R_T->set_coefficient(0, _cell_bubble);
  _L_R_T->set_coefficient(num_coeffs - 1, _cell_bubble);

  // Attach primal approximation to left-hand side form (residual) if
  // necessary.
  if (_is_linear)
  {
    boost::shared_ptr<const GenericFunction> _u(&u, NoDeleter());
    _L_R_T->set_coefficient(num_coeffs - 2, _u);
  }

  // Create data structures for local assembly data
  UFC ufc_lhs(*_a_R_T);
  UFC ufc_rhs(*_L_R_T);

  // Extract common space, mesh and dofmap
  const FunctionSpace& V = *R_T.function_space();
  dolfin_assert(V.mesh());
  const Mesh& mesh(*V.mesh());
  dolfin_assert(V.dofmap());
  const GenericDofMap& dofmap = *V.dofmap();

  // Define matrices for cell-residual problems
  dolfin_assert(V.element());
  const std::size_t N = V.element()->space_dimension();
  arma::mat A(N, N);
  arma::mat b(N, 1);
  arma::vec x(N);

  // Extract cell_domains etc from right-hand side form
  const MeshFunction<std::size_t>*
    cell_domains = _L_R_T->cell_domains_shared_ptr().get();
  const MeshFunction<std::size_t>*
    exterior_facet_domains = _L_R_T->exterior_facet_domains_shared_ptr().get();
  const MeshFunction<std::size_t>*
    interior_facet_domains = _L_R_T->interior_facet_domains_shared_ptr().get();

  // Assemble and solve local linear systems
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Assemble local linear system
    LocalAssembler::assemble(A, ufc_lhs, *cell, cell_domains,
                             exterior_facet_domains, interior_facet_domains);
    LocalAssembler::assemble(b, ufc_rhs, *cell, cell_domains,
                             exterior_facet_domains, interior_facet_domains);

    // Solve linear system and convert result
    x = arma::solve(A, b);

    // Get local-to-global dof map for cell
    const std::vector<std::size_t>& dofs = dofmap.cell_dofs(cell->index());

    // Plug local solution into global vector
    dolfin_assert(R_T.vector());
    R_T.vector()->set(x.memptr(), N, &dofs[0]);
  }
  end();
}
//-----------------------------------------------------------------------------
void ErrorControl::compute_facet_residual(SpecialFacetFunction& R_dT,
                                          const Function& u,
                                          const Function& R_T)
{
  begin("Computing facet residual representation.");

  // Extract function space for facet residual approximation
  dolfin_assert(R_dT[0].function_space());
  const FunctionSpace& V = *R_dT[0].function_space();
  dolfin_assert(V.element());
  const std::size_t N = V.element()->space_dimension();

  // Extract mesh
  dolfin_assert(V.mesh());
  const Mesh& mesh = *V.mesh();
  const int dim = mesh.topology().dim();

  // Extract dimension of cell cone space (DG_{dim})
  dolfin_assert(_C->element());
  const int local_cone_dim = _C->element()->space_dimension();

  // Extract number of coefficients on right-hand side (for use with
  // attaching coefficients)
  dolfin_assert(_L_R_dT);
  const uint L_R_dT_num_coefficients = _L_R_dT->num_coefficients();

  // Attach primal approximation if linear (already attached
  // otherwise).
  if (_is_linear)
  {
    boost::shared_ptr<const GenericFunction> _u(&u, NoDeleter());
    _L_R_dT->set_coefficient(L_R_dT_num_coefficients - 3, _u);
  }

  // Attach cell residual to residual form
  dolfin_assert(_R_T);
  _L_R_dT->set_coefficient(L_R_dT_num_coefficients - 2, _R_T);

  // Extract (common) dof map
  dolfin_assert(V.dofmap());
  const GenericDofMap& dofmap = *V.dofmap();

  // Define matrices for facet-residual problems
  arma::mat A(N, N);
  arma::mat b(N, 1);
  arma::vec x(N);

  // Variables to be used for the construction of the cone function
  const std::size_t num_cells = mesh.num_cells();
  const std::vector<double> ones(num_cells, 1.0);
  std::vector<std::size_t> facet_dofs(num_cells);

  // Extract cell_domains etc from right-hand side form
  dolfin_assert(_L_R_T);
  const MeshFunction<std::size_t>*
    cell_domains = _L_R_T->cell_domains_shared_ptr().get();
  const MeshFunction<std::size_t>*
    exterior_facet_domains = _L_R_T->exterior_facet_domains_shared_ptr().get();
  const MeshFunction<std::size_t>*
    interior_facet_domains = _L_R_T->interior_facet_domains_shared_ptr().get();

  dolfin_assert(_a_R_dT);
  // Compute the facet residual for each local facet number
  for (int local_facet = 0; local_facet <= dim; local_facet++)
  {
    // Construct "cone" function on this facet. NB: makes assumption
    // on _local_ dof numbering for DG_dim.
    dolfin_assert(_cell_cone->vector());
    *(_cell_cone->vector()) = 0.0;
    facet_dofs.clear();
    const std::size_t local_facet_dof = local_cone_dim - (dim + 1) + local_facet;
    dolfin_assert(_cell_cone->function_space());
    dolfin_assert(_cell_cone->function_space()->dofmap());
    const GenericDofMap& cone_dofmap(*(_cell_cone->function_space()->dofmap()));
    for (std::size_t k = 0; k < num_cells; k++)
      facet_dofs.push_back(cone_dofmap.cell_dofs(k)[local_facet_dof]);
    _cell_cone->vector()->set(&ones[0], num_cells, &facet_dofs[0]);

    // Attach cell cone to _a_R_dT and _L_R_dT
    _a_R_dT->set_coefficient(0, _cell_cone);
    _L_R_dT->set_coefficient(L_R_dT_num_coefficients - 1, _cell_cone);

    // Create data structures for local assembly data
    UFC ufc_lhs(*_a_R_dT);
    UFC ufc_rhs(*_L_R_dT);

    // Assemble and solve local linear systems
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Assemble linear system
      LocalAssembler::assemble(A, ufc_lhs, *cell, cell_domains,
                               exterior_facet_domains, interior_facet_domains);
      LocalAssembler::assemble(b, ufc_rhs, *cell, cell_domains,
                               exterior_facet_domains, interior_facet_domains);

      // Non-singularize local matrix
      for (std::size_t i = 0; i < N; ++i)
      {
        if (std::abs(A(i, i)) < 1.0e-10)
        {
          A(i, i) = 1.0;
          b(i) = 0.0;
        }
      }

      // Solve linear system and convert result
      x = arma::solve(A, b);

      // Get local-to-global dof map for cell
      const std::vector<std::size_t>& dofs = dofmap.cell_dofs(cell->index());

      // Plug local solution into global vector
      dolfin_assert(R_dT[local_facet].vector());
      R_dT[local_facet].vector()->set(x.memptr(), N, &dofs[0]);
    }
  }
  end();
}
//-----------------------------------------------------------------------------
void ErrorControl::apply_bcs_to_extrapolation(
const std::vector<boost::shared_ptr<const BoundaryCondition> > bcs)
{
  // Create boundary conditions for extrapolated dual, and apply
  // these.
  for (uint i = 0; i < bcs.size(); i++)
  {
    // Only handle DirichletBCs
    const DirichletBC* bc = dynamic_cast<const DirichletBC*>(bcs[i].get());
    dolfin_assert(bc);

    // Extract SubSpace component
    dolfin_assert(bc->function_space());
    const std::vector<uint> component = bc->function_space()->component();

    // Extract sub-domain
    boost::shared_ptr<const SubDomain> sub_domain = bc->user_sub_domain();

    // Create zero-valued boundary condition on extrapolation space.
    // (Sub-spaces need special handling, and boundary conditions can
    // be defined and handled in many different ways -- hence the
    // level of logic.)
    boost::scoped_ptr<DirichletBC> e_bc;
    if (component.empty())
    {
      if (sub_domain)
        e_bc.reset(new DirichletBC(_E, bc->value(), sub_domain, bc->method()));
      else
        e_bc.reset(new DirichletBC(_E, bc->value(), bc->markers(), bc->method()));
    }
    else
    {
      boost::shared_ptr<SubSpace> S(new SubSpace(*_E, component));
      if (sub_domain)
        e_bc.reset(new DirichletBC(S, bc->value(), sub_domain, bc->method()));
      else
        e_bc.reset(new DirichletBC(S, bc->value(), bc->markers(), bc->method()));
    }
    e_bc->homogenize();

    // Apply boundary condition
    e_bc->apply(*_Ez_h->vector());
  }
}
//-----------------------------------------------------------------------------
