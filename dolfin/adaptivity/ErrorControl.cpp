// Copyright (C) 2010 Marie E. Rognes
// Licensed under the GNU LGPL Version 3.0 or any later version
//
// First added:  2010-09-16
// Last changed: 2011-01-03

#include <boost/scoped_array.hpp>

#include <dolfin/common/Timer.h>
#include <dolfin/fem/UFC.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/la/solve.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/SubSpace.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/VariationalProblem.h>
#include <dolfin/fem/BoundaryCondition.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/DofMap.h>

#include "ErrorControl.h"
#include "SpecialFacetFunction.h"

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
  const Function& e_tmp(dynamic_cast<const Function&>(_residual->coefficient(improved_dual)));
  _E = e_tmp.function_space_ptr();

  const Function& b_tmp(dynamic_cast<const Function&>(_a_R_dT->coefficient(0)));
  _C = b_tmp.function_space_ptr();
}
//-----------------------------------------------------------------------------
double ErrorControl::estimate_error(const Function& u,
                                    std::vector<const BoundaryCondition*> bcs)
{
  // Compute dual
  Function z_h(_a_star->function_space(1));
  compute_dual(z_h, bcs);

  // Compute extrapolation
  compute_extrapolation(z_h,  bcs);

  // Extract number of coefficients in residual
  const uint num_coeffs = _residual->num_coefficients();

  // Attach improved dual approximation to residual
  _residual->set_coefficient(num_coeffs - 1, _Ez_h);

  // Attach primal approximation if linear (already attached
  // otherwise).
  if (_is_linear)
    _residual->set_coefficient(num_coeffs - 2, u);

  // Assemble error estimate
  _residual->set_mesh(u.function_space().mesh());
  const double error_estimate = assemble(*_residual);

  // Return estimate
  return error_estimate;
}
//-----------------------------------------------------------------------------
void ErrorControl::compute_dual(Function& z,
                                std::vector<const BoundaryCondition*> bcs)
{
  // FIXME: Create dual (homogenized) boundary conditions
  std::vector<const BoundaryCondition*> dual_bcs;
  for (uint i = 0; i < bcs.size(); i++)
    dual_bcs.push_back(bcs[i]);   // push_back(bcs[i].homogenize());

  // Create and solve dual variational problem
  VariationalProblem dual(*_a_star, *_L_star, dual_bcs);
  dual.solve(z);
}
//-----------------------------------------------------------------------------
void ErrorControl::compute_extrapolation(const Function& z,
                                         std::vector<const BoundaryCondition*> bcs)
{
  // Extrapolate
  _Ez_h.reset(new Function(_E));
  _Ez_h->extrapolate(z);

  // Apply homogeneous boundary conditions to extrapolated dual
  Function u0(_E);

  for (uint i = 0; i < bcs.size(); i++)
  {
    // FIXME: Suboptimal cast.
    DirichletBC bc(*dynamic_cast<const DirichletBC*>(bcs[i]));

    // Extract SubSpace component
    const FunctionSpace& V(bc.function_space());
    const Array<uint>& component(V.component());

    // If bcs[i].function_space is non subspace:
    if (component.size() == 0)
    {
      // Define constant 0.0 on this space
      Function u0(V);

      // Create corresponding boundary condition for extrapolation
      DirichletBC e_bc(V, u0, bc.markers());

      // Apply boundary condition to extrapolation
      e_bc.apply(_Ez_h->vector());
      continue;
    }

    // Create Subspace of _Ez_h
    SubSpace S(*_E, component[0]); // FIXME: Only one level allowed so far...

    // Define constant 0.0 on this space
    Function u0(S);

    // Create corresponding boundary condition for extrapolation
    DirichletBC e_bc(S, u0, bc.markers());

    // Apply boundary condition to extrapolation
    e_bc.apply(_Ez_h->vector());
  }
}
//-----------------------------------------------------------------------------
void ErrorControl::compute_indicators(Vector& indicators, const Function& u)
{
  // Create Function for the strong cell residual (R_T)
  Function R_T(_a_R_T->function_space(1));

  // Create SpecialFacetFunction for the strong facet residual (R_dT)
  std::vector<Function *> f_e;
  for (uint i=0; i <= R_T.geometric_dimension(); i++)
    f_e.push_back(new Function(_a_R_dT->function_space(1)));

  SpecialFacetFunction* R_dT;
  if (f_e[0]->value_rank() == 0)
    R_dT = new SpecialFacetFunction(f_e);
  else if (f_e[0]->value_rank() == 1)
    R_dT = new SpecialFacetFunction(f_e, f_e[0]->value_dimension(0));
  else
  {
    R_dT = new SpecialFacetFunction(f_e, f_e[0]->value_dimension(0));
    error("Not implemented for tensor-valued functions");
  }

  // Compute residual representation
  residual_representation(R_T, *R_dT, u);

  // Interpolate dual extrapolation into primal test (dual trial space)
  Function Pi_E_z_h(_a_star->function_space(1));
  Pi_E_z_h.interpolate(*_Ez_h);

  // Attach coefficients to error indicator form
  _eta_T->set_coefficient(0, *_Ez_h);
  _eta_T->set_coefficient(1, R_T);
  _eta_T->set_coefficient(2, *R_dT);
  _eta_T->set_coefficient(3, Pi_E_z_h);

  // Assemble error indicator form
  assemble(indicators, *_eta_T);

  // Take absolute value of indicators: FIXME. Must be better way.
  double abs;
  for (uint i=0; i < indicators.size(); i++)
  {
    abs = std::abs(indicators.getitem(i));
    indicators.setitem(i, abs);
  }

  // Delete stuff
  for (uint i=0; i <= R_T.geometric_dimension(); i++)
    delete f_e[i];
  delete R_dT;
}
//-----------------------------------------------------------------------------
void ErrorControl::residual_representation(Function& R_T,
                                           SpecialFacetFunction& R_dT,
                                           const Function& u)
{
  begin("Computing residual representation");

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

  begin("Computing cell residual representation");

  // Attach primal approximation to left-hand side form (residual) if
  // necessary
  if (_is_linear) {
    const uint num_coeffs = _L_R_T->num_coefficients();
    _L_R_T->set_coefficient(num_coeffs - 2, u);
  }

  // Extract trial space for strong cell residual and mesh
  const FunctionSpace& V(R_T.function_space());
  const Mesh& mesh(V.mesh());

  // Define Matrix and Vector and Vector for cell-residual problems of
  // dimension equal to the local dimension of the function spaces
  const uint N = V.element().space_dimension();

  // Hope that Armadillo is the best at solving small linear systems
  arma::mat A(N, N);
  arma::vec b(N);
  arma::vec x(N);

  std::vector<uint> exterior_facets;
  std::vector<uint> interior_facets;

  // Solve local problem for each cell in mesh
  const uint D = mesh.topology().dim();

  // Create data structures for local assembly data
  UFC ufc_lhs(*_a_R_T);
  UFC ufc_rhs(*_L_R_T);

  // Local array for dof indices
  boost::scoped_array<uint> dofs(new uint[N]);

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Figure out which facets are interior and which are exterior on
    // this cell by iterating over facets in cell
    exterior_facets.clear();
    interior_facets.clear();

    A.zeros();
    b.zeros();
    for (FacetIterator facet(*cell); !facet.end(); ++facet)
    {
      if (facet->num_entities(D) == 2)
        interior_facets.push_back(facet.pos());
      else
        exterior_facets.push_back(facet.pos());
    }

    // Assemble A
    assemble_cell(A, N, ufc_lhs, *cell, exterior_facets, interior_facets);

    // Assemble b
    assemble_cell(b, N, ufc_rhs, *cell, exterior_facets, interior_facets);

    // solve A x = b
    x = arma::solve(A, b);

    // Tabulate dofs for w on cell
    V.dofmap().tabulate_dofs(dofs.get(), ufc_lhs.cell, cell->index());

    // Plug x into global vector
    for (uint i=0; i < N; i++)
      R_T.vector().setitem(dofs[i], x[i]);
  }

  end();
}
//-----------------------------------------------------------------------------
void ErrorControl::compute_facet_residual(SpecialFacetFunction& R_dT,
                                          const Function& u,
                                          const Function& R_T)
{
  begin("Computing facet residual representation");

  // Extract function space for facet residual approximation
  const FunctionSpace& V(R_dT[0]->function_space());
  const uint N = V.element().space_dimension();

  // Extract mesh
  const Mesh& mesh(V.mesh());
  int q = mesh.topology().dim();

  // Extract dimension of cell cone space
  int n = _C->element().space_dimension();

  arma::mat A(N, N);
  arma::vec b(N);
  arma::vec x(N);

  std::vector<uint> exterior_facets;
  std::vector<uint> interior_facets;

  // Local array for dof indices
  boost::scoped_array<uint> dofs(new uint[N]);

  // Extract number of coefficients on right-hand side (for use with
  // attaching coefficients)
  const uint L_R_dT_num_coefficients = _L_R_dT->num_coefficients();

  // Attach primal approximation if linear (already attached
  // otherwise).
  if (_is_linear)
    _L_R_dT->set_coefficient(L_R_dT_num_coefficients - 3, u);

  for (int e=0; e < (q+1); e++)
  {
    // Construct b_e
    Function b_e(_C);
    for (uint k = 0; k < mesh.num_cells(); k++)
      b_e.vector().setitem(n*(k+1) - (q+1) + e, 1.0);
    b_e.vector().apply("insert");

    // Attach b_e to _a_R_dT and _L_R_dT
    _a_R_dT->set_coefficient(0, b_e);
    _L_R_dT->set_coefficient(L_R_dT_num_coefficients - 2, R_T);
    _L_R_dT->set_coefficient(L_R_dT_num_coefficients - 1, b_e);

    UFC ufc_lhs(*_a_R_dT);
    UFC ufc_rhs(*_L_R_dT);

    // Solve local problem for each cell in mesh
    const uint D = mesh.topology().dim();
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Figure out which facets are interior and which are exterior on
      // this cell by iterating over facets in cell
      exterior_facets.clear();
      interior_facets.clear();

      A.zeros();
      b.zeros();

      for (FacetIterator facet(*cell); !facet.end(); ++facet)
      {
        if (facet->num_entities(D) == 2)
          interior_facets.push_back(facet.pos());
        else
          exterior_facets.push_back(facet.pos());
      }

      // Assemble A
      assemble_cell(A, N, ufc_lhs, *cell, exterior_facets, interior_facets);

      // Assemble b
      assemble_cell(b, N, ufc_rhs, *cell, exterior_facets, interior_facets);

      //Non-singularize
      for(uint i=0; i < N; i ++)
      {
        if (std::abs(A(i, i)) < 1.e-10)
        {
          A(i, i) = 1.0;
          b(i) = 0.0;
        }
      }

      // solve A x = b
      x = arma::solve(A, b);

      // Tabulate dofs for w on cell
      V.dofmap().tabulate_dofs(dofs.get(), ufc_lhs.cell, cell->index());

      // Plug x into global vector
      for (uint i=0; i < N; i++)
        R_dT[e]->vector().setitem(dofs[i], x[i]);
    }
  }
  end();
}
//-----------------------------------------------------------------------------
void ErrorControl::assemble_cell(arma::mat& A, const uint N,
                                 UFC& ufc,
                                 const Cell& cell,
                                 std::vector<uint> exterior_facets,
                                 std::vector<uint> interior_facets)
{
  uint local_facet = 0;

  // Iterate over cell_integral(s) (assume max 1 for now)
  if (ufc.form.num_cell_integrals() == 1)
  {
    // Extract cell integral
    ufc::cell_integral* integral = ufc.cell_integrals[0];

    // Update ufc object to this cell
    ufc.update(cell);

    // Tabulate local tensor
    integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell);

    // Stuff a_ufc.A into A
    for (uint i=0; i < N; i++)
    {
      for (uint j=0; j < N; j++) {
        A(i, j) += ufc.A[N*i + j];
      }
    }
  }

  // Iterate over exterior_facet integral(s) (assume max 1 for now)
  if (ufc.form.num_exterior_facet_integrals() == 1)
  {
    // Extract exterior facet integral
    ufc::exterior_facet_integral* ef_integral = ufc.exterior_facet_integrals[0];

    // Iterate over given exterior facets
    for (uint k=0; k < exterior_facets.size(); k++)
    {
      // Get local index of facet with respect to the cell
      local_facet = exterior_facets[k];

      // Update to current cell and facet
      ufc.update(cell, local_facet);

      // Tabulate exterior facet tensor
      ef_integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell, local_facet);

      // Stuff a_ufc.A into A
      for (uint i=0; i < N; i++)
      {
        for (uint j=0; j < N; j++)
          A(i, j) += ufc.A[N*i + j];
      }
    }
  }

  // Iterate over interior facet integral(s) (assume max 1 for now)
  if (ufc.form.num_interior_facet_integrals() == 1)
  {
    // Extract exterior facet integral
    ufc::interior_facet_integral* if_integral = ufc.interior_facet_integrals[0];

    // Iterate over given interior facets
    for (uint k=0; k < interior_facets.size(); k++)
    {
      // Get local index of facet with respect to the cell
      local_facet = interior_facets[k];

      // Update to current pair of cells and facets
      ufc.update(cell, local_facet, cell, local_facet);

      // Tabulate exterior interior facet tensor on macro element
      if_integral->tabulate_tensor(ufc.macro_A.get(), ufc.macro_w,
                                ufc.cell0, ufc.cell1,
                                local_facet, local_facet);

      // Stuff a_ufc.A into A
      for (uint i=0; i < N; i++)
      {
        for (uint j=0; j < N; j++)
          A(i, j) += ufc.macro_A[2*N*i + j];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void ErrorControl::assemble_cell(arma::vec& b, const uint N,
                                 UFC& ufc,
                                 const Cell& cell,
                                 std::vector<uint> exterior_facets,
                                 std::vector<uint> interior_facets)
{
  // Iterate over cell_integral(s) (assume max 1 for now)
  if (ufc.form.num_cell_integrals() == 1)
  {
    // Extract cell integral
    ufc::cell_integral* integral = ufc.cell_integrals[0];

    // Update ufc object to this cell
    ufc.update(cell);

    // Tabulate local tensor
    integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell);

    // Stuff a_ufc.A into b
    for (uint i=0; i < N; i++)
      b(i) += ufc.A[i];
  }

  // Iterate over exterior_facet integral(s) (assume max 1 for now)
  if (ufc.form.num_exterior_facet_integrals() == 1)
  {
    // Extract exterior facet integral
    ufc::exterior_facet_integral* ef_integral = ufc.exterior_facet_integrals[0];

    // Iterate over given exterior facets
    for (uint k=0; k < exterior_facets.size(); k++)
    {
      // Get local index of facet with respect to the cell
      const uint local_facet = exterior_facets[k];

      // Update to current cell and facet
      ufc.update(cell, local_facet);

      // Tabulate exterior facet tensor
      ef_integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell, local_facet);

      // Stuff a_ufc.A into A
      for (uint i=0; i < N; i++)
        b(i) += ufc.A[i];
    }
  }

  // Iterate over interior facet integral(s) (assume max 1 for now)
  if (ufc.form.num_interior_facet_integrals() == 1)
  {
    // Extract interior facet integral
    ufc::interior_facet_integral* if_integral = ufc.interior_facet_integrals[0];

    // Iterate over given exterior facets
    for (uint k=0; k < interior_facets.size(); k++)
    {
      // Get local index of facet with respect to the cell
      const uint local_facet = interior_facets[k];

      // Update to current pair of cells and facets
      ufc.update(cell, local_facet, cell, local_facet);

      // Tabulate exterior interior facet tensor on macro element
      if_integral->tabulate_tensor(ufc.macro_A.get(), ufc.macro_w,
                                ufc.cell0, ufc.cell1,
                                local_facet, local_facet);

      // Stuff correct pieces from a_ufc.A into b
      for (uint i=0; i < N; i++)
        b(i) += ufc.macro_A[i];
    }
  }

}
//-----------------------------------------------------------------------------
