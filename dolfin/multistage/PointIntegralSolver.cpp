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
// Last changed: 2014-10-14

#include <cmath>
#include <algorithm>
#include <memory>

#include <dolfin/log/log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/Constant.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/UFC.h>

#include "MultiStageScheme.h"
#include "PointIntegralSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PointIntegralSolver::PointIntegralSolver(std::shared_ptr<MultiStageScheme> scheme) :
  Variable("PointIntegralSolver", "unnamed"), _scheme(scheme),
  _mesh(_scheme->last_stage()->mesh()),
  _dofmap(*_scheme->last_stage()->function_space(0)->dofmap()),
  _system_size(_dofmap.num_entity_dofs(0)),
  _dof_offset(_mesh->type().num_entities(0)),
  _num_stages(_scheme->stage_forms().size()),
  _local_to_local_dofs(_system_size),
  _vertex_map(), _local_to_global_dofs(_system_size),
  _local_stage_solutions(_scheme->stage_solutions().size()),
  _u0(_system_size), _residual(_system_size), _y(_system_size),
  _dx(_system_size),
  _ufcs(), _coefficient_index(), _recompute_jacobian(),
  _jacobians(), _eta(1.0), _num_jacobian_computations(0)
{
  Timer construct_pis("Construct PointIntegralSolver");

  // Set parameters
  parameters = default_parameters();

  _check_forms();
  _init();
}
//-----------------------------------------------------------------------------
PointIntegralSolver::~PointIntegralSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::reset_newton_solver()
{
  _eta = parameters("newton_solver")["eta_0"];

  for (unsigned int i=0; i < _recompute_jacobian.size(); i++)
    _recompute_jacobian[i] = true;
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::reset_stage_solutions()
{
  // Iterate over stage forms
  for (unsigned int stage=0; stage<_num_stages; stage++)
  {
    // Reset global stage solutions
    *_scheme->stage_solutions()[stage]->vector() = 0.0;

    // Reset local stage solutions
    for (unsigned int row=0; row < _system_size; row++)
      _local_stage_solutions[stage][row] = 0.0;
  }
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::step(double dt)
{
  dolfin_assert(_mesh);

  const bool reset_stage_solutions_ = parameters["reset_stage_solutions"];
  const bool reset_newton_solver_
    = parameters("newton_solver")["reset_each_step"];

  // Check for reseting stage solutions
  if (reset_stage_solutions_)
    reset_stage_solutions();

  // Check for reseting newtonsolver for each time step
  if (reset_newton_solver_)
    reset_newton_solver();

  Timer t_step("PointIntegralSolver::step");

  dolfin_assert(dt > 0.0);

  // Update time constant of scheme
  *_scheme->dt() = dt;

  // Time at start of timestep
  const double t0 = *_scheme->t();

  // Get ownership range
  const dolfin::la_index local_dof_size = _dofmap.ownership_range().second
    - _dofmap.ownership_range().first;

  // Iterate over vertices
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;
  for (std::size_t vert_ind = 0; vert_ind < _mesh->num_vertices(); ++vert_ind)
  {
    // Cell containing vertex
    const Cell cell(*_mesh, _vertex_map[vert_ind].first);
    cell.get_coordinate_dofs(coordinate_dofs);
    cell.get_cell_data(ufc_cell);

    // Get all dofs for cell
    // FIXME: Should we include logics about empty dofmaps?
    const ArrayView<const dolfin::la_index> cell_dofs
      = _dofmap.cell_dofs(cell.index());

    // Tabulate local-local dofmap
    _dofmap.tabulate_entity_dofs(_local_to_local_dofs, 0,
                                 _vertex_map[vert_ind].second);

    // Fill local to global dof map and check that the dof is owned
    bool owns_all_dofs = true;
    for (unsigned int row = 0; row < _system_size; row++)
    {
      _local_to_global_dofs[row] = cell_dofs[_local_to_local_dofs[row]];
      if (_local_to_global_dofs[row] >= local_dof_size)
      {
        owns_all_dofs = false;
        break;
      }
    }

    // If not owning all dofs
    if (!owns_all_dofs)
      continue;

    // Iterate over stage forms
    for (unsigned int stage = 0; stage < _num_stages; stage++)
    {
      // Update cell
      // TODO: Pass suitable bool vector here to avoid tabulating all
      // coefficient dofs:
      _ufcs[stage][0]->update(cell, coordinate_dofs, ufc_cell);
      //some_integral.enabled_coefficients());

      // Check if we have an explicit stage (only 1 form)
      if (_ufcs[stage].size() == 1)
      {
        _solve_explicit_stage(vert_ind, stage, ufc_cell, coordinate_dofs);
      }
      // or an implicit stage (2 forms)
      else
      {
        _solve_implicit_stage(vert_ind, stage, cell, ufc_cell,
                              coordinate_dofs);
      }
    }

    Timer t_last_stage("Last stage: tabulate_tensor");

    // Last stage point integral
    const ufc::vertex_integral& integral
      = *_last_stage_ufc->default_vertex_integral;

    // Update coefficients for last stage
    // TODO: Pass suitable bool vector here to avoid tabulating all
    // coefficient dofs:
    _last_stage_ufc->update(cell, coordinate_dofs, ufc_cell);
    //integral.enabled_coefficients());

    // Tabulate cell tensor
    integral.tabulate_tensor(_last_stage_ufc->A.data(), _last_stage_ufc->w(),
                             coordinate_dofs.data(),
                             _vertex_map[vert_ind].second,
                             ufc_cell.orientation);

    // Update solution with a tabulation of the last stage
    for (unsigned int row = 0; row < _system_size; row++)
      _y[row] = _last_stage_ufc->A[_local_to_local_dofs[row]];

    // Update global solution with last stage
    _scheme->solution()->vector()->set_local(_y.data(),
                                             _local_to_global_dofs.size(),
                                             _local_to_global_dofs.data());
  }

  for (unsigned int stage=0; stage<_num_stages; stage++)
    _scheme->stage_solutions()[stage]->vector()->apply("insert");

  _scheme->solution()->vector()->apply("insert");

  // Update time
  *_scheme->t() = t0 + dt;
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::_solve_explicit_stage(std::size_t vert_ind,
                                                unsigned int stage,
                                                const ufc::cell& ufc_cell,
                                                const std::vector<double>& coordinate_dofs)
{
  Timer t_expl("Explicit stage");

  // Local vertex ind
  const unsigned int local_vert = _vertex_map[vert_ind].second;

  // Point integral
  const ufc::vertex_integral& integral
    = *_ufcs[stage][0]->default_vertex_integral;

  // Tabulate cell tensor
  Timer t_expl_tt("Explicit stage: tabulate_tensor");
  integral.tabulate_tensor(_ufcs[stage][0]->A.data(), _ufcs[stage][0]->w(),
                           coordinate_dofs.data(), local_vert,
                           ufc_cell.orientation);
  t_expl_tt.stop();

  // Extract vertex dofs from tabulated tensor and put them into the
  // local stage solution vector
  // Extract vertex dofs from tabulated tensor
  for (unsigned int row = 0; row < _system_size; row++)
  {
    _local_stage_solutions[stage][row]
      = _ufcs[stage][0]->A[_local_to_local_dofs[row]];
  }


  // FIXME: This below is dodgy and will sooner or later break in parallel (GNW)
  // Put solution back into global stage solution vector
  // NOTE: This so an UFC.update (coefficient restriction) would just
  // work
  _scheme->stage_solutions()[stage]->vector()->set_local(
    _local_stage_solutions[stage].data(), _system_size,
    _local_to_global_dofs.data());
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::_solve_implicit_stage(std::size_t vert_ind,
                                                unsigned int stage,
                                                const Cell& cell,
                                                const ufc::cell& ufc_cell,
                                                const std::vector<double>& coordinate_dofs)
{
  Timer t_impl("Implicit stage");

  // Do a simplified newton solve
  _simplified_newton_solve(vert_ind, stage, cell, ufc_cell, coordinate_dofs);

  // Put solution back into global stage solution vector
  _scheme->stage_solutions()[stage]->vector()->set_local(_local_stage_solutions[stage].data(),
                                                         _system_size,
                                                         _local_to_global_dofs.data());
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::step_interval(double t0, double t1, double dt)
{
  if (dt <= 0.0)
  {
    dolfin_error("PointIntegralSolver.cpp",
                 "stepping PointIntegralSolver",
                 "Expecting a positive dt");
  }

  if (t0 >= t1)
  {
    dolfin_error("PointIntegralSolver.cpp",
                 "stepping PointIntegralSolver",
                 "Expecting t0 to be smaller than t1");
  }

  // Set start time
  *_scheme->t() = t0;
  double t = t0;
  double next_dt = std::min(t1-t, dt);

  // Step interval
  while (t + next_dt <= t1)
  {
    if (next_dt < DOLFIN_EPS)
      break;
    step(next_dt);
    t = *_scheme->t();
    next_dt = std::min(t1-t, dt);
  }
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::_compute_jacobian(std::vector<double>& jac,
                                            const std::vector<double>& u,
                                            unsigned int local_vert,
                                            UFC& loc_ufc, const Cell& cell,
                                            const ufc::cell& ufc_cell,
                                            int coefficient_index,
                                            const std::vector<double>& coordinate_dofs)
{
  const ufc::vertex_integral& J_integral = *loc_ufc.default_vertex_integral;

  //Timer _timer_compute_jac("Implicit stage: Compute jacobian");
  //Timer t_impl_update("Update_cell");
  // TODO: Pass suitable bool vector here to avoid tabulating all
  // coefficient dofs:
  loc_ufc.update(cell, coordinate_dofs, ufc_cell);
  //J_integral.enabled_coefficients());
  //t_impl_update.stop();

  // If there is a solution coefficient in the Jacobian form
  if (coefficient_index > 0)
  {
    // Put solution back into restricted coefficients before tabulate
    // new jacobian
    for (unsigned int row = 0; row < _system_size; row++)
      loc_ufc.w()[coefficient_index][_local_to_local_dofs[row]] = u[row];
  }

  // Tabulate Jacobian
  Timer t_impl_tt_jac("Implicit stage: tabulate_tensor (J)");
  J_integral.tabulate_tensor(loc_ufc.A.data(), loc_ufc.w(),
                             coordinate_dofs.data(),
                             local_vert,
                             ufc_cell.orientation);
  t_impl_tt_jac.stop();

  // Extract vertex dofs from tabulated tensor
  //Timer t_impl_update_jac("Implicit stage: update_jac");
  for (unsigned int row = 0; row < _system_size; row++)
  {
    for (unsigned int col = 0; col < _system_size; col++)
    {
      jac[row*_system_size + col] = loc_ufc.A[_local_to_local_dofs[row]*
                                              _dof_offset*_system_size +
                                              _local_to_local_dofs[col]];
    }
  }
  //t_impl_update_jac.stop();

  // LU factorize Jacobian
  //Timer lu_factorize("Implicit stage: LU factorize");
  _lu_factorize(jac);
  _num_jacobian_computations += 1;
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::_lu_factorize(std::vector<double>& A)
{
  // Local variables
  double sum;
  const int system_size = _system_size;
  for (int k = 1; k < system_size; k++)
  {
    for (int i = 0; i <= k-1; ++i)
    {
      sum = 0.0;
      for (int r = 0; r <= i-1; r++)
        sum += A[i*_system_size+r]*A[r*_system_size+k];
      A[i*_system_size+k] -=sum;

      sum = 0.0;
      for (int r = 0; r <= i-1; r++)
        sum += A[k*_system_size+r]*A[r*_system_size+i];
      A[k*_system_size+i] = (A[k*_system_size+i]-sum)/A[i*_system_size+i];
    }

    sum = 0.0;
    for (int r = 0; r <= k-1; r++)
      sum += A[k*_system_size+r]*A[r*_system_size+k];
    A[k*_system_size + k] -= sum;
  }
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::_forward_backward_subst(const std::vector<double>& A,
                                                  const std::vector<double>& b,
                                                  std::vector<double>& x) const
{
  // solves Ax = b with forward backward substitution, provided that
  // A is already LU factorized

  double sum;
  x[0] = b[0];

  // Forward
  for (unsigned int i=1; i < _system_size; ++i)
  {
    sum = 0.0;
    for (unsigned int j=0; j <= i-1; ++j)
      sum = sum + A[i*_system_size+j]*x[j];

    x[i] = b[i] -sum;
  }

  const unsigned int _system_size_m_1 = _system_size-1;
  x[_system_size_m_1] = x[_system_size_m_1]/A[_system_size_m_1*_system_size+_system_size_m_1];

  // Backward
  for (int i = _system_size-2; i >= 0; i--)
  {
    sum = 0;
    for (unsigned int j=i+1; j < _system_size; ++j)
      sum = sum +A[i*_system_size+j]*x[j];

    x[i] = (x[i]-sum)/A[i*_system_size+i];
  }
}
//-----------------------------------------------------------------------------
double PointIntegralSolver::_norm(const std::vector<double>& vec) const
{
  double l2_norm = 0;

  for (unsigned int i = 0; i < vec.size(); ++i)
    l2_norm += vec[i]*vec[i];

  l2_norm = std::sqrt(l2_norm);
  return l2_norm;
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::_check_forms()
{
  // Iterate over stage forms and check they include point integrals
  std::vector<std::vector<std::shared_ptr<const Form>>>& stage_forms
    = _scheme->stage_forms();
  for (unsigned int i = 0; i < stage_forms.size(); i++)
  {
    for (unsigned int j = 0; j < stage_forms[i].size(); j++)
    {
      // Check form includes at least PointIntegral
      if (!stage_forms[i][j]->ufc_form()->has_vertex_integrals())
      {
        dolfin_error("PointIntegralSolver.cpp",
                     "constructing PointIntegralSolver",
                     "Expecting form to have at least 1 PointIntegral");
      }

      // Num dofs per vertex
      const GenericDofMap& dofmap
        = *stage_forms[i][j]->function_space(0)->dofmap();
      const unsigned int dofs_per_vertex = dofmap.num_entity_dofs(0);
      const unsigned int vert_per_cell
        = _mesh->topology()(_mesh->topology().dim(), 0).size(0);

      if (vert_per_cell*dofs_per_vertex != dofmap.max_element_dofs())
      {
        dolfin_error("PointIntegralSolver.cpp",
                     "constructing PointIntegralSolver",
                     "Expecting test space to only have dofs on vertices");
      }
    }
  }
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::_init()
{
  dolfin_assert(_mesh);

  // Get stage forms
  std::vector<std::vector<std::shared_ptr<const Form>>>& stage_forms
    = _scheme->stage_forms();

  // Init local stage solutions
  for (unsigned int stage = 0; stage < _num_stages; stage++)
    _local_stage_solutions[stage].resize(_system_size);

  // Init coefficient index and ufcs
  _coefficient_index.resize(stage_forms.size());
  _ufcs.resize(stage_forms.size());

  // Initiate jacobian matrices
  if (_scheme->implicit())
  {
    // First count the number of distinct jacobians
    int max_jacobian_index = 0;
    for (unsigned int stage = 0; stage < _num_stages; stage++)
    {
      max_jacobian_index = std::max(_scheme->jacobian_index(stage),
                                    max_jacobian_index);
    }

    // Create memory for jacobians
    _jacobians.resize(max_jacobian_index+1);
    for (int i=0; i<=max_jacobian_index; i++)
      _jacobians[i].resize(_system_size*_system_size);
    _recompute_jacobian.resize(max_jacobian_index+1, true);
  }

  // Create last stage UFC form
  _last_stage_ufc = std::make_shared<UFC>(*_scheme->last_stage());

  // Iterate over stages and collect information
  for (unsigned int stage = 0; stage < stage_forms.size(); stage++)
  {
    // Create a UFC object for first form
    _ufcs[stage].push_back(std::make_shared<UFC>(*stage_forms[stage][0]));

    //  If implicit stage
    if (stage_forms[stage].size()==2)
    {
      // Create a UFC object for second form
      _ufcs[stage].push_back(std::make_shared<UFC>(*stage_forms[stage][1]));

      // Find coefficient index for each of the two implicit forms
      for (unsigned int i = 0; i < 2; i++)
      {
        for (unsigned int j = 0;  j < stage_forms[stage][i]->num_coefficients();
             j++)
        {
          // Check if stage solution is the same as coefficient j
          if (stage_forms[stage][i]->coefficients()[j]->id()
              == _scheme->stage_solutions()[stage]->id())
          {
            _coefficient_index[stage].push_back(j);
            break;
          }
        }
      }

      // Check that nonlinear form includes a coefficient index
      // (otherwise it might be some error in the form)
      if (_coefficient_index[stage].size() == 0)
      {
        dolfin_error("PointIntegralSolver.cpp",
                     "constructing PointIntegralSolver",
                     "Expecting nonlinear form of stage: %d to be dependent "\
                     "on the stage solution.", stage);
      }
    }
  }

  // Build vertex map
  _vertex_map.resize(_mesh->num_vertices());

  // Init mesh connections
  _mesh->init(0);
  const unsigned int dim_t = _mesh->topology().dim();

  // Iterate over vertices and collect cell and local vertex
  // information
  for (VertexIterator vert(*_mesh); !vert.end(); ++vert)
  {
    // First look for cell where the vert is local vert 0
    bool local_vert_found = false;
    for (CellIterator cell(*vert); !cell.end(); ++cell )
    {
      // If the first local vertex is the same as the global vertex
      if (cell->entities(0)[0]==vert->index())
      {
        _vertex_map[vert->index()].first = cell->index();
        _vertex_map[vert->index()].second = 0;
        local_vert_found = true;
        break;
      }
    }

    // If no cell exist where vert corresponds to local vert 0 just
    // grab local cell 0 and find what local vert the global vert
    // corresponds to
    if (!local_vert_found)
    {
      const Cell cell0(*_mesh, vert->entities(dim_t)[0]);
      _vertex_map[vert->index()].first = cell0.index();

      unsigned int local_vert_index = 0;
      for (VertexIterator local_vert(cell0); !local_vert.end(); ++local_vert)
      {
        // If local vert is found
        if (vert->index() == local_vert->index())
        {
          // Store local vertex index
          _vertex_map[vert->index()].second = local_vert_index;
          break;
        }

        // Bump index
        local_vert_index++;
      }
    }
  }
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::_simplified_newton_solve(
  std::size_t vert_ind, unsigned int stage,
  const Cell& cell,
  const ufc::cell& ufc_cell,
  const std::vector<double>& coordinate_dofs)
{
  //Timer _timer_newton_solve("Implicit stage: Newton solve");
  const Parameters& newton_solver_params = parameters("newton_solver");
  const size_t report_vertex = newton_solver_params["report_vertex"];
  const double kappa = newton_solver_params["kappa"];
  const double rtol = newton_solver_params["relative_tolerance"];
  const double atol = newton_solver_params["absolute_tolerance"];
  std::size_t max_iterations = newton_solver_params["maximum_iterations"];
  const double max_relative_previous_residual
    = newton_solver_params["max_relative_previous_residual"];
  const double relaxation = newton_solver_params["relaxation_parameter"];
  const bool report = newton_solver_params["report"];
  const bool verbose_report = newton_solver_params["verbose_report"];
  bool always_recompute_jacobian
    = newton_solver_params["always_recompute_jacobian"];
  const unsigned int local_vert = _vertex_map[vert_ind].second;
  UFC& loc_ufc_F = *_ufcs[stage][0];
  UFC& loc_ufc_J = *_ufcs[stage][1];
  const int coefficient_index_F = _coefficient_index[stage][0];
  const int coefficient_index_J = _coefficient_index[stage].size()==2 ?
    _coefficient_index[stage][1] : -1;
  const unsigned int jac_index = _scheme->jacobian_index(stage);
  std::vector<double>& jac = _jacobians[jac_index];

  if (newton_solver_params["recompute_jacobian_each_solve"])
    _recompute_jacobian[jac_index] = true;

  bool newton_solve_restared = false;
  unsigned int newton_iterations = 0;
  double relative_previous_residual = 0., residual, prev_residual = 1.,
    initial_residual = 1., relative_residual = 1.;

  // Get point integrals
  const ufc::vertex_integral& F_integral = *loc_ufc_F.default_vertex_integral;

  // Local solution
  std::vector<double>& u = _local_stage_solutions[stage];

  // Update with previous local solution and make a backup of solution
  // to be used in a potential restarting of newton solver
  for (unsigned int row=0; row < _system_size; row++)
  {
    _u0[row] = u[row]
      = loc_ufc_F.w()[coefficient_index_F][_local_to_local_dofs[row]];
  }

  do
  {
    // Tabulate residual
    Timer t_impl_tt_F("Implicit stage: tabulate_tensor (F)");
    F_integral.tabulate_tensor(loc_ufc_F.A.data(), loc_ufc_F.w(),
                               coordinate_dofs.data(),
                               local_vert,
                               ufc_cell.orientation);
    t_impl_tt_F.stop();

    // Extract vertex dofs from tabulated tensor, together with the old stage
    // solution
    for (unsigned int row=0; row < _system_size; row++)
      _residual[row] = loc_ufc_F.A[_local_to_local_dofs[row]];

    residual = _norm(_residual);
    if (newton_iterations == 0)
      initial_residual = residual;//std::max(residual, DOLFIN_EPS);

    relative_residual = residual/initial_residual;

    // Check for relative residual convergence together with a check for absolute tolerance
    // The absolute tolerance check is combined a check that the residual is not changing
    // FIXME: Make the relative_previous_residual check configurable
    if (relative_residual < rtol || (residual < atol && relative_previous_residual > 0.99))
      break;

    // Should we recompute jacobian
    if (_recompute_jacobian[jac_index] || always_recompute_jacobian)
    {
      _compute_jacobian(jac, u, local_vert, loc_ufc_J, cell, ufc_cell,
                        coefficient_index_J, coordinate_dofs);
      _recompute_jacobian[jac_index] = false;
    }

    // Perform linear solve By forward backward substitution
    //Timer forward_backward_substitution("Implicit stage: fb substitution");
    _forward_backward_subst(jac, _residual, _dx);
    //forward_backward_substitution.stop();

    // Newton_Iterations == 0
    if (newton_iterations == 0)
    {
      // On first iteration we need an approximation of eta. We take
      // the one from previous step and increase it slightly. This is
      // important for linear problems which only should require 1
      // iteration to converge.
      _eta = _eta > DOLFIN_EPS ? _eta : DOLFIN_EPS;
      _eta = std::pow(_eta, 0.8);
    }
    // 2nd time around
    else
    {
      // How fast are we converging?
      relative_previous_residual = residual/prev_residual;

      if (always_recompute_jacobian)
      {
        if ((report && vert_ind == report_vertex) || verbose_report)
        {
          info("Newton solver after %d iterations. vertex: %d, "
               "relative_previous_residual: %.3f, "
               "relative_residual: %.3e, residual: %.3e.",
               newton_iterations, vert_ind, relative_previous_residual,
               relative_residual, residual);
        }
      }

      // If we diverge
      else if (relative_previous_residual > 1)
      {
        if ((report && vert_ind == report_vertex) || verbose_report)
        {
          info("Newton solver diverges after %d iterations. vertex: %d, "
               "relative_previous_residual: %.3f, "
               "relative_residual: %.3e, residual: %.3e.",
               newton_iterations, vert_ind, relative_previous_residual,
               relative_residual, residual);
        }

        // If we have not restarted newton solve previously
        if (!newton_solve_restared)
        {
          if ((report && vert_ind == report_vertex) || verbose_report)
            info("Restarting newton solve for vertex: %d", vert_ind);

          // Reset flags
          newton_solve_restared = true;
          always_recompute_jacobian = true;

          // Reset solution
          for (unsigned int row=0; row < _system_size; row++)
          {
            loc_ufc_F.w()[coefficient_index_F][_local_to_local_dofs[row]]
              = u[row] = _u0[row];
          }

          // Update variables
          _eta = parameters("newton_solver")["eta_0"];
          newton_iterations = 0;
          relative_previous_residual = prev_residual = initial_residual
            = relative_residual = 1.0;
          max_iterations = 400;
          continue;
        }

      }

      // We converge too slow
      else if (relative_previous_residual >= max_relative_previous_residual ||
               residual > (kappa*rtol*(1 - relative_previous_residual) /
                           std::pow(relative_previous_residual,
                                    max_iterations - newton_iterations)))
      {
        if ((report && vert_ind == report_vertex) || verbose_report)
        {
          info("Newton solver converges too slow at iteration %d. vertex: %d, "
               "relative_previous_residual: %.3f, "
               "relative_residual: %.3e, residual: %.3e. Recomputing jacobian.",
               newton_iterations, vert_ind, relative_previous_residual,
               relative_residual, residual);
        }
        _recompute_jacobian[jac_index] = true;
      }
      else
      {
        if ((report && vert_ind == report_vertex) || verbose_report)
        {
          info("Newton solver after %d iterations. vertex: %d, "
               "relative_previous_residual: %.3f, "
               "relative_residual: %.3e, residual: %.3e.",
               newton_iterations, vert_ind, relative_previous_residual,
               relative_residual, residual);
        }
        // Update eta
        _eta = relative_previous_residual/(1.0 - relative_previous_residual);
      }
    }

    // No convergence
    if (newton_iterations > max_iterations)
    {
      if (report)
      {
        info("Newton solver did not converge after %d iterations. vertex: %d, "
             "relative_previous_residual: %.3f, "
             "relative_residual: %.3e, residual: %.3e.", max_iterations, vert_ind,
             relative_previous_residual, relative_residual, residual);
      }
      dolfin_error("PointIntegralSolver.h",
                   "Newton point integral solver",
                   "Newton solver in PointIntegralSolver exceeded maximal iterations");
    }

    // Update solution
    if (std::abs(1.0 - relaxation) < DOLFIN_EPS)
      for (unsigned int i=0; i < u.size(); i++)
        u[i] -= _dx[i];
    else
      for (unsigned int i=0; i < u.size(); i++)
        u[i] -= relaxation*_dx[i];

    // Put solution back into restricted coefficients before tabulate
    // new residual
    for (unsigned int row=0; row < _system_size; row++)
      loc_ufc_F.w()[coefficient_index_F][_local_to_local_dofs[row]] = u[row];

    prev_residual = residual;
    newton_iterations++;

  } while(_eta*relative_residual >= kappa*rtol);

  if ((report && vert_ind == report_vertex) || verbose_report)
  {
    info("Newton solver converged after %d iterations. vertex: %d, "\
         "relative_previous_residual: %.3f, relative_residual: %.3e, "\
         "residual: %.3e.", newton_iterations, vert_ind,
         relative_previous_residual, relative_residual, residual);
  }
}
//-----------------------------------------------------------------------------
