// Copyright (C) 2010 Marie E. Rognes
// Licensed under the GNU LGPL Version 3.0 or any later version
//
// First added:  2010-08-19
// Last changed: 2011-02-16

#ifndef __ERROR_CONTROL_H
#define __ERROR_CONTROL_H

#include <vector>
#include <boost/shared_ptr.hpp>

#include <dolfin/common/Hierarchical.h>

namespace dolfin
{

  class BoundaryCondition;
  class Form;
  class Function;
  class FunctionSpace;
  class SpecialFacetFunction;
  class Vector;

  /// (Goal-oriented) Error Control class.

  /// The notation used here follows the notation in "Automated
  /// goal-oriented error control I: stationary variational problems",
  /// ME Rognes and A Logg, 2010-2011.

  class ErrorControl : public Hierarchical<ErrorControl>
  {
  public:

    /// Create error control
    ErrorControl(boost::shared_ptr<Form> a_star,
                 boost::shared_ptr<Form> L_star,
                 boost::shared_ptr<Form> residual,
                 boost::shared_ptr<Form> a_R_T,
                 boost::shared_ptr<Form> L_R_T,
                 boost::shared_ptr<Form> a_R_dT,
                 boost::shared_ptr<Form> L_R_dT,
                 boost::shared_ptr<Form> eta_T,
                 bool is_linear);

    /// Destructor.
    ~ErrorControl() { /* Do nothing */};

    /// Estimate the error relative to the goal M of the discrete
    /// approximation 'u' relative to the variational formulation by
    /// evaluating the weak residual at an approximation to the dual
    /// solution.
    ///
    /// *Arguments*
    ///     u (_Function_)
    ///        the primal approximation
    ///
    ///     bcs (std::vector<const _BoundaryCondition_*>)
    ///         the primal boundary conditions
    ///
    /// *Returns*
    ///     double
    ///         error estimate
    double estimate_error(const Function& u,
                          std::vector<const BoundaryCondition*> bcs);

    /// Compute error indicators
    ///
    /// *Arguments*
    ///     indicators (_Vector_)
    ///         the error indicators (to be computed)
    ///
    ///     u (_Function_)
    ///         the primal approximation
    void compute_indicators(Vector& indicators, const Function& u);

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
    ///     bcs (std::vector<const _BoundaryCondition_*>)
    ///         the primal boundary conditions
    void compute_dual(Function& z,
                      std::vector<const BoundaryCondition*> bcs);

    /// Compute extrapolation with boundary conditions
    ///
    /// *Arguments*
    ///     z (_Function_)
    ///         the extrapolated function (to be computed)
    ///
    ///     bcs (std::vector<const _BoundaryCondition_*>)
    ///         the dual boundary conditions
    void compute_extrapolation(const Function& z,
                               std::vector<const BoundaryCondition*> bcs);

    friend dolfin::ErrorControl& dolfin::adapt(ErrorControl& ec,
                                               boost::shared_ptr<const Mesh> refined_mesh);


  private:

    // Bilinear and linear form for dual problem
    boost::shared_ptr<Form> _a_star;
    boost::shared_ptr<Form> _L_star;

    // Functional for evaluating residual (error estimate)
    boost::shared_ptr<Form> _residual;

    // Bilinear and linear form for computing cell residual R_T
    boost::shared_ptr<Form> _a_R_T;
    boost::shared_ptr<Form> _L_R_T;

    // Bilinear and linear form for computing facet residual R_dT
    boost::shared_ptr<Form> _a_R_dT;
    boost::shared_ptr<Form> _L_R_dT;

    // Linear form for computing error indicators
    boost::shared_ptr<Form> _eta_T;

    // Computed extrapolation
    boost::shared_ptr<Function> _Ez_h;

    bool _is_linear;

    // Pointers to other function spaces (FIXME: Only out-of-scope
    // motivated)
    boost::shared_ptr<const FunctionSpace> _E;
    boost::shared_ptr<const FunctionSpace> _C;

    // Pointers to other functions (FIXME: Only out-of-scope
    // motivated)
    boost::shared_ptr<Function> _cell_cone;
    boost::shared_ptr<Function> _R_T;
    boost::shared_ptr<SpecialFacetFunction> _R_dT;
    boost::shared_ptr<Function> _Pi_E_z_h;

  };

}

#endif
