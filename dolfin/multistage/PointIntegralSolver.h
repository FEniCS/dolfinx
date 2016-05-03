// Copyright (C) 2013 Johan Hake
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
// First added:  2013-02-15
// Last changed: 2016-04-27

#ifndef __POINTINTEGRALSOLVER_H
#define __POINTINTEGRALSOLVER_H

#include <memory>
#include <set>
#include <vector>

#include <dolfin/common/Variable.h>
#include <dolfin/fem/Assembler.h>

namespace dolfin
{

  /// This class is a time integrator for general Runge Kutta forms,
  /// which only includes Point integrals with piecewise linear test
  /// functions. Such problems are disconnected at the vertices and
  /// can therefore be solved locally.

  // Forward declarations
  class MultiStageScheme;
  class UFC;

  class PointIntegralSolver : public Variable
  {
  public:

    /// Constructor
    /// FIXME: Include version where one can pass a Solver and/or
    /// Parameters
    explicit PointIntegralSolver(std::shared_ptr<MultiStageScheme> scheme);

    /// Destructor
    ~PointIntegralSolver();

    /// Step solver with time step dt
    void step(double dt);

    /// Step solver an interval using dt as time step
    void step_interval(double t0, double t1, double dt);

    /// Return the MultiStageScheme
    std::shared_ptr<MultiStageScheme> scheme() const
    { return _scheme; }

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("point_integral_solver");

      p.add("reset_stage_solutions", true);

      // Set parameters for NewtonSolver
      Parameters pn("newton_solver");
      pn.add("maximum_iterations", 40);
      pn.add("always_recompute_jacobian", false);
      pn.add("recompute_jacobian_each_solve", true);
      pn.add("relaxation_parameter", 1., 0., 1.);
      pn.add("relative_tolerance", 1e-10, 1e-20, 2.);
      pn.add("absolute_tolerance", 1e-15, 1e-20, 2.);

      pn.add("kappa", 0.1, 0.05, 1.0);
      pn.add("eta_0", 1., 1e-15, 1.0);
      pn.add("max_relative_previous_residual", 1e-1, 1e-5, 1.);
      pn.add("reset_each_step", true);
      pn.add("report", false);
      pn.add("report_vertex", 0, 0, 32767);
      pn.add("verbose_report", false);

      p.add(pn);

      return p;
    }

    // Reset newton solver
    void reset_newton_solver();

    // Reset stage solutions
    void reset_stage_solutions();

    // Return number of computations of jacobian
    std::size_t num_jacobian_computations() const
    { return _num_jacobian_computations; }

  private:

    // In-place LU factorization of jacobian matrix
    void _lu_factorize(std::vector<double>& A);

    // Forward backward substitution, assume that mat is already
    // in place LU factorized
    void _forward_backward_subst(const std::vector<double>& A,
                                 const std::vector<double>& b,
                                 std::vector<double>& x) const;

    // Compute jacobian using passed UFC form
    void _compute_jacobian(std::vector<double>& jac,
                           const std::vector<double>& u,
                           unsigned int local_vert, UFC& loc_ufc,
                           const Cell& cell, const ufc::cell& ufc_cell,
                           int coefficient_index,
                           const std::vector<double>& coordinate_dofs);

    // Compute the norm of a vector
    double _norm(const std::vector<double>& vec) const;

    // Check the forms making sure they only include piecewise linear
    // test functions
    void _check_forms();

    // Build map between vertices, cells and the corresponding local
    // vertex and initialize UFC data for each form
    void _init();

    // Solve an explicit stage
    void _solve_explicit_stage(std::size_t vert_ind, unsigned int stage,
                               const ufc::cell& ufc_cell,
                               const std::vector<double>& coordinate_dofs);

    // Solve an implicit stage
    void _solve_implicit_stage(std::size_t vert_ind, unsigned int stage,
                               const Cell& cell, const ufc::cell& ufc_cell,
                               const std::vector<double>& coordinate_dofs);

    void
      _simplified_newton_solve(std::size_t vert_ind, unsigned int stage,
                               const Cell& cell, const ufc::cell& ufc_cell,
                               const std::vector<double>& coordinate_dofs);

    // The MultiStageScheme
    std::shared_ptr<MultiStageScheme> _scheme;

    // Reference to mesh
    std::shared_ptr<const Mesh> _mesh;

    // The dofmap (Same for all stages and forms)
    const GenericDofMap& _dofmap;

    // Size of ODE system
    const std::size_t _system_size;

    // Offset into local dofmap
    // FIXME: Consider put in local loop
    const unsigned int _dof_offset;

    // Number of stages
    const unsigned int _num_stages;

    // Local to local dofs to be used in tabulate entity dofs
    std::vector<std::size_t> _local_to_local_dofs;

    // Vertex map between vertices, cells and corresponding local
    // vertex
    std::vector<std::pair<std::size_t, unsigned int>> _vertex_map;

    // Local to global dofs used when solution is fanned out to global
    // vector
    std::vector<dolfin::la_index> _local_to_global_dofs;

    // Local stage solutions
    std::vector<std::vector<double>> _local_stage_solutions;

    // Local solutions
    std::vector<double> _u0;
    std::vector<double> _residual;
    std::vector<double> _y;
    std::vector<double> _dx;

    // UFC objects, one for each form
    std::vector<std::vector<std::shared_ptr<UFC>>> _ufcs;

    // UFC objects for the last form
    std::shared_ptr<UFC> _last_stage_ufc;

    // Solution coefficient index in form
    std::vector<std::vector<int>> _coefficient_index;

    // Flag which is set to false once the jacobian has been computed
    std::vector<bool> _recompute_jacobian;

    // Jacobians/LU factorized jacobians matrices
    std::vector<std::vector<double>> _jacobians;

    // Variable used in the estimation of the error of the newton
    // iteration for the first iteration (important for linear
    // problems!)
    double _eta;

    // Number of computations of Jacobian
    std::size_t _num_jacobian_computations;

  };

}

#endif
