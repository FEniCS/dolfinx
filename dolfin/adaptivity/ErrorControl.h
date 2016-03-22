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
// First added:  2010-08-19
// Last changed: 2012-09-03

#ifndef __ERROR_CONTROL_H
#define __ERROR_CONTROL_H

#include <vector>
#include <memory>

#include <dolfin/common/Hierarchical.h>
#include <dolfin/common/Variable.h>
#include <dolfin/fem/LinearVariationalSolver.h>
#include "adapt.h"

namespace dolfin
{

  class DirichletBC;
  class Form;
  class Function;
  class FunctionSpace;
  class SpecialFacetFunction;
  class Vector;

  template <typename T> class MeshFunction;

  /// (Goal-oriented) Error Control class.

  /// The notation used here follows the notation in "Automated
  /// goal-oriented error control I: stationary variational problems",
  /// ME Rognes and A Logg, 2010-2011.

  class ErrorControl : public Hierarchical<ErrorControl>, public Variable
  {
  public:

    /// Create error control object
    ///
    /// *Arguments*
    ///     a_star (_Form_)
    ///        the bilinear form for the dual problem
    ///     L_star (_Form_)
    ///        the linear form for the dual problem
    ///     residual (_Form_)
    ///        a functional for the residual (error estimate)
    ///     a_R_T (_Form_)
    ///        the bilinear form for the strong cell residual problem
    ///     L_R_T (_Form_)
    ///        the linear form for the strong cell residual problem
    ///     a_R_dT (_Form_)
    ///        the bilinear form for the strong facet residual problem
    ///     L_R_dT (_Form_)
    ///        the linear form for the strong facet residual problem
    ///     eta_T (_Form_)
    ///        a linear form over DG_0 for error indicators
    ///     is_linear (bool)
    ///        true iff primal problem is linear
    ErrorControl(std::shared_ptr<Form> a_star,
                 std::shared_ptr<Form> L_star,
                 std::shared_ptr<Form> residual,
                 std::shared_ptr<Form> a_R_T,
                 std::shared_ptr<Form> L_R_T,
                 std::shared_ptr<Form> a_R_dT,
                 std::shared_ptr<Form> L_R_dT,
                 std::shared_ptr<Form> eta_T,
                 bool is_linear);

    /// Destructor.
    ~ErrorControl() {}

    /// Default parameter values:
    static Parameters default_parameters()
    {
      Parameters p("error_control");

      // Set parameters for dual solver
      Parameters p_dual(LinearVariationalSolver::default_parameters());
      p_dual.rename("dual_variational_solver");
      p.add(p_dual);

      return p;
    }


    /// Estimate the error relative to the goal M of the discrete
    /// approximation 'u' relative to the variational formulation by
    /// evaluating the weak residual at an approximation to the dual
    /// solution.
    ///
    /// *Arguments*
    ///     u (_Function_)
    ///        the primal approximation
    ///
    ///     bcs (std::vector<_DirichletBC_>)
    ///         the primal boundary conditions
    ///
    /// *Returns*
    ///     double
    ///         error estimate
    double estimate_error(const Function& u,
           const std::vector<std::shared_ptr<const DirichletBC> > bcs);

    /// Compute error indicators
    ///
    /// *Arguments*
    ///     indicators (_MeshFunction_ <double>)
    ///         the error indicators (to be computed)
    ///
    ///     u (_Function_)
    ///         the primal approximation
    void compute_indicators(MeshFunction<double>& indicators,
                            const Function& u);

    /// Compute strong representation (strong cell and facet
    /// residuals) of the weak residual.
    ///
    /// *Arguments*
    ///     R_T (_Function_)
    ///         the strong cell residual (to be computed)
    ///
    ///     R_dT (_SpecialFacetFunction_)
    ///         the strong facet residual (to be computed)
    ///
    ///     u (_Function_)
    ///         the primal approximation
    void residual_representation(Function& R_T,
                                 SpecialFacetFunction& R_dT,
                                 const Function& u);

    /// Compute representation for the strong cell residual
    /// from the weak residual
    ///
    /// *Arguments*
    ///     R_T (_Function_)
    ///         the strong cell residual (to be computed)
    ///
    ///     u (_Function_)
    ///         the primal approximation
    void compute_cell_residual(Function& R_T, const Function& u);

    /// Compute representation for the strong facet residual from the
    /// weak residual and the strong cell residual
    ///
    /// *Arguments*
    ///     R_dT (_SpecialFacetFunction_)
    ///         the strong facet residual (to be computed)
    ///
    ///     u (_Function_)
    ///         the primal approximation
    ///
    ///     R_T (_Function_)
    ///         the strong cell residual
    void compute_facet_residual(SpecialFacetFunction& R_dT,
                                const Function& u,
                                const Function& R_T);

    /// Compute dual approximation defined by dual variational
    /// problem and dual boundary conditions given by homogenized primal
    /// boundary conditions.
    ///
    /// *Arguments*
    ///     z (_Function_)
    ///         the dual approximation (to be computed)
    ///
    ///     bcs (std::vector<_DirichletBC_>)
    ///         the primal boundary conditions
    void compute_dual(Function& z,
         const std::vector<std::shared_ptr<const DirichletBC> > bcs);

    /// Compute extrapolation with boundary conditions
    ///
    /// *Arguments*
    ///     z (_Function_)
    ///         the extrapolated function (to be computed)
    ///
    ///     bcs (std::vector<_DirichletBC_>)
    ///         the dual boundary conditions
    void compute_extrapolation(const Function& z,
         const std::vector<std::shared_ptr<const DirichletBC> > bcs);

    friend std::shared_ptr<ErrorControl>
      adapt(const ErrorControl& ec,
            std::shared_ptr<const Mesh> adapted_mesh,
            bool adapt_coefficients);

  private:

    void apply_bcs_to_extrapolation(const std::vector<std::shared_ptr<const DirichletBC> > bcs);

    // Bilinear and linear form for dual problem
    std::shared_ptr<Form> _a_star;
    std::shared_ptr<Form> _L_star;

    // Functional for evaluating residual (error estimate)
    std::shared_ptr<Form> _residual;

    // Bilinear and linear form for computing cell residual R_T
    std::shared_ptr<Form> _a_R_T;
    std::shared_ptr<Form> _L_R_T;

    // Bilinear and linear form for computing facet residual R_dT
    std::shared_ptr<Form> _a_R_dT;
    std::shared_ptr<Form> _L_R_dT;

    // Linear form for computing error indicators
    std::shared_ptr<Form> _eta_T;

    // Computed extrapolation
    std::shared_ptr<Function> _Ez_h;

    bool _is_linear;

    // Function spaces for extrapolation, cell bubble and cell cone:
    std::shared_ptr<const FunctionSpace> _extrapolation_space;
    std::shared_ptr<const FunctionSpace> _bubble_space;
    std::shared_ptr<const FunctionSpace> _cone_space;

    // Functions for cell bubble, cell cone, computed cell residual,
    // computed facet residual, and interpolated extrapolated(!) dual:
    std::shared_ptr<Function> _cell_bubble;
    std::shared_ptr<Function> _cell_cone;
    std::shared_ptr<Function> _R_T;
    std::shared_ptr<SpecialFacetFunction> _R_dT;
    std::shared_ptr<Function> _Pi_E_z_h;
  };
}

#endif
