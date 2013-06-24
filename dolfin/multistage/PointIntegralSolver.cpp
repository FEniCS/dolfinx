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
// Last changed: 2013-06-24

#include <cmath>
#include <boost/make_shared.hpp>

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
PointIntegralSolver::PointIntegralSolver(boost::shared_ptr<MultiStageScheme> scheme) : 
  Variable("PointIntegralSolver", "unnamed"), _scheme(scheme), 
  _mesh(_scheme->stage_forms()[0][0]->mesh()),
  _dofmap(*_scheme->stage_forms()[0][0]->function_space(0)->dofmap()), 
  _system_size(_dofmap.num_entity_dofs(0)), _dof_offset(_mesh.type().num_entities(0)), 
  _num_stages(_scheme->stage_forms().size()), _local_to_local_dofs(_system_size),
  _vertex_map(), _local_to_global_dofs(_system_size), 
  _local_stage_solutions(_scheme->stage_solutions().size()), 
  _F(_system_size), _y(_system_size), _dx(_system_size), 
  _ufcs(), _coefficient_index(), _recompute_jacobian(true), 
  _jac(), _eta(1e-10), _num_jacobian_computations(0)
{
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
void PointIntegralSolver::reset()
{
  _num_jacobian_computations = 0;
  _eta = parameters("newton_solver")["eta_0"];
  _recompute_jacobian = true;
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::step(double dt)
{
  Timer t_step("PointIntegralSolver::step");
  
  dolfin_assert(dt > 0.0);

  // Update time constant of scheme
  *_scheme->dt() = dt;

  // Time at start of timestep
  const double t0 = *_scheme->t();

  // Update ghost values
  _update_ghost_values();

  //const std::size_t num_threads = dolfin::parameters["num_threads"];

  // Iterate over vertices
  //Progress p("Solving local point integral problems", _mesh.num_vertices());
  
  //#pragma omp parallel for schedule(guided, 20) private(_jac_L, _jac_U, _jac, _local_to_global_dofs, _local_to_local_dofs, _ufcs)
  for (std::size_t vert_ind=0; vert_ind < _mesh.num_vertices(); ++vert_ind)
  {

    // FIXME: Some debug output
    std::cout << std::endl << std::endl << "Vertex: " << vert_ind << std::endl;
    
    Timer t_vert("Step: update vert");

    // Cell containing vertex
    const Cell cell(_mesh, _vertex_map[vert_ind].first);

    // Get all dofs for cell
    // FIXME: Shold we include logics about empty dofmaps?
    const std::vector<dolfin::la_index>& cell_dofs = _dofmap.cell_dofs(cell.index());

    // Tabulate local-local dofmap
    _dofmap.tabulate_entity_dofs(_local_to_local_dofs, 0, _vertex_map[vert_ind].second);
    
    // Fill local to global dof map
    for (unsigned int row=0; row < _system_size; row++)
    {
      _local_to_global_dofs[row] = cell_dofs[_local_to_local_dofs[row]];
    }

    t_vert.stop();

    // Iterate over stage forms
    for (unsigned int stage=0; stage<_num_stages; stage++)
    {

      // FIXME: Debug:
      _scheme->stage_solutions()[stage]->vector()->get_local(&_y[0], 
							     _local_to_global_dofs.size(), 
							     &_local_to_global_dofs[0]);
      
      // FIXME: Some debug output
      std::cout << "Stage solution before solve[" << stage << "]: ";
      for (unsigned int row=0; row < _system_size; row++)
      {
	std::cout << _y[row] << ", ";
      }
      
      std::cout << std::endl;

      // FIXME: Some debug output
      std::cout << "Local stage solution before solve[" << stage << "]: ";
      for (unsigned int row=0; row < _system_size; row++)
      {
	std::cout << _local_stage_solutions[stage][row] << ", ";
      }
      
      std::cout << std::endl;

      // Update cell
      Timer t_impl_update("Update_cell");
      //for (unsigned int i=0; i < _ufcs[stage].size(); i++)
      _ufcs[stage][0]->update(cell);

      t_impl_update.stop();

      // Check if we have an explicit stage (only 1 form)
      if (_ufcs[stage].size()==1)
      {
	_solve_explicit_stage(vert_ind, stage);
      }
    
      // or an implicit stage (2 forms)
      else
      {
	_solve_implicit_stage(vert_ind, stage, cell);
      }

      // FIXME: Debug:
      _scheme->stage_solutions()[stage]->vector()->get_local(&_y[0], 
							     _local_to_global_dofs.size(), 
							     &_local_to_global_dofs[0]);
      
      // FIXME: Some debug output
      std::cout << "Stage solution after solve[" << stage << "]: ";
      for (unsigned int row=0; row < _system_size; row++)
      {
	std::cout << _y[row] << ", ";
      }
      
      std::cout << std::endl;

      // FIXME: Some debug output
      std::cout << "Local stage solution after solve[" << stage << "]: ";
      for (unsigned int row=0; row < _system_size; row++)
      {
	std::cout << _local_stage_solutions[stage][row] << ", ";
      }
      
      std::cout << std::endl;

    }

    Timer t_last_stage("Last stage: tabulate_tensor");

    // Update coeffcients for last stage
    _last_stage_ufc->update(cell);

    // Last stage point integral
    const ufc::point_integral& integral = *_last_stage_ufc->default_point_integral;

    // Tabulate cell tensor
    integral.tabulate_tensor(&_last_stage_ufc->A[0], _last_stage_ufc->w(), 
			     &_last_stage_ufc->cell.vertex_coordinates[0], 
			     _vertex_map[vert_ind].second);
    
    // Update solution with a tabulation of the last stage
    for (unsigned int row=0; row < _system_size; row++)
      _y[row] = _last_stage_ufc->A[_local_to_local_dofs[row]];

    // Update global solution with last stage
    _scheme->solution()->vector()->set(&_y[0], _local_to_global_dofs.size(), 
				       &_local_to_global_dofs[0]);
    t_last_stage.stop();
    
    //p++;
  }

  // Update time
  *_scheme->t() = t0 + dt;
  
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::_solve_explicit_stage(std::size_t vert_ind, 
						unsigned int stage)
{
  Timer t_expl("Explicit stage");

  // Local vertex ind
  const unsigned int local_vert = _vertex_map[vert_ind].second;

  // Point integral
  const ufc::point_integral& integral = *_ufcs[stage][0]->default_point_integral;

  // Update to current cell
  Timer t_expl_update("Explicit stage: update_cell");
  t_expl_update.stop();

  // Tabulate cell tensor
  Timer t_expl_tt("Explicit stage: tabulate_tensor");
  integral.tabulate_tensor(&_ufcs[stage][0]->A[0], _ufcs[stage][0]->w(), 
			   &_ufcs[stage][0]->cell.vertex_coordinates[0], 
			   local_vert);
  t_expl_tt.stop();

  // Extract vertex dofs from tabulated tensor and put them into the local 
  // stage solution vector
  // Extract vertex dofs from tabulated tensor
  for (unsigned int row=0; row < _system_size; row++)
  {
    _local_stage_solutions[stage][row] = _ufcs[stage][0]->A[_local_to_local_dofs[row]];
  }

  // Put solution back into global stage solution vector
  // NOTE: This so an update (coefficient restriction) would just work
  _scheme->stage_solutions()[stage]->vector()->set(
		 &_local_stage_solutions[stage][0], _system_size, 
		 &_local_to_global_dofs[0]);

}
//-----------------------------------------------------------------------------
void PointIntegralSolver::_solve_implicit_stage(std::size_t vert_ind,
						unsigned int stage,
						const Cell& cell)
{
	
  Timer t_impl("Implicit stage");
	
  // Local vertex ind
  const unsigned int local_vert = _vertex_map[vert_ind].second;

  // Set initial convergence
  convergence_criteria_t convergence = diverge;

  // Local counter for jacobian calculations
  unsigned int jacobian_calculations = 0;

  std::size_t max_jacobian_computations = parameters("newton_solver")["maximum_jacobian_computations"];

  // Local solution
  std::vector<double>& u = _local_stage_solutions[stage];
	
  // Do until not converged
  while (convergence != converged)
  {
    
    // Recompute jacobian if convergence is too slow
    if (_recompute_jacobian)
    {
      
      _compute_jacobian(_jac, u, local_vert, *_ufcs[stage][1], cell,
			_coefficient_index[stage].size()==2 ?	\
			_coefficient_index[stage][1] : -1);
      jacobian_calculations += 1;
      
    }
    
    // Do a simplified newton solve
    convergence = _simplified_newton_solve(u, vert_ind, *_ufcs[stage][0], 
					   _coefficient_index[stage][0], cell);
    
    // First time we do not converge and it is the second time around
    if (convergence != converged && jacobian_calculations > max_jacobian_computations)
    {

      if (convergence == too_slow)
	error("Newton solver converged too slowly, after jacobian "\
	      "has been re-computed %d times.", jacobian_calculations);
      
      // If we diverge
      if (convergence == diverge)
	error("Newton solver in PointIntegralSolver diverged, after jacobian "\
	      "has been re-computed %d times.", jacobian_calculations);

      // If we exceed max iterations
      if (jacobian_calculations > 1 && convergence == exceeds_max_iter)
	error("Newton solver in PointIntegralSolver exeeded maximal iterations, "\
	      "after jacobian has been re-computed %d times.", jacobian_calculations);

    }

  }

  Timer t_impl_set("Implicit stage: set");

  // Put solution back into global stage solution vector
  _scheme->stage_solutions()[stage]->vector()->set(&u[0], u.size(), 
						   &_local_to_global_dofs[0]);


}
//-----------------------------------------------------------------------------
void PointIntegralSolver::step_interval(double t0, double t1, double dt)
{
  if (dt<=0.0)
  {
    dolfin_error("PointIntegralSolver.cpp",
		 "stepping PointIntegralSolver",
		 "Expecting a positive dt");
  }

  if (t0>=t1)
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
					    UFC& loc_ufc, 
					    const Cell& cell, int coefficient_index)
{
  Timer t_impl_update("Update_cell");
  loc_ufc.update(cell);
  t_impl_update.stop();

  const ufc::point_integral& J_integral = *loc_ufc.default_point_integral;

  // If there is a solution coefficient in the jacobian form
  if (coefficient_index>0)
  {
    // Put solution back into restricted coefficients before tabulate new jacobian
    for (unsigned int row=0; row < _system_size; row++)
    {
      loc_ufc.w()[coefficient_index][_local_to_local_dofs[row]] = u[row];
    }
  }

  // Tabulate Jacobian
  Timer t_impl_tt_jac("Implicit stage: tabulate_tensor (J)");
  J_integral.tabulate_tensor(&loc_ufc.A[0], loc_ufc.w(), 
			     &loc_ufc.cell.vertex_coordinates[0], 
			     local_vert);
  t_impl_tt_jac.stop();

  // Extract vertex dofs from tabulated tensor
  Timer t_impl_update_jac("Implicit stage: update_jac");
  for (unsigned int row=0; row < _system_size; row++)
    for (unsigned int col=0; col < _system_size; col++)
      jac[row*_system_size + col] = loc_ufc.A[_local_to_local_dofs[row]*
					      _dof_offset*_system_size +
					      _local_to_local_dofs[col]];
  t_impl_update_jac.stop();

  // LU factorize Jacobian
  Timer lu_factorize("Implicit stage: LU factorize");
  _lu_factorize(jac);
  _recompute_jacobian = false;
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
      {
        sum += A[i*_system_size+r]*A[r*_system_size+k];
      }

      A[i*_system_size+k] -=sum;
      
      sum = 0.0;
      for (int r = 0; r <= i-1; r++)
      {
        sum += A[k*_system_size+r]*A[r*_system_size+i];
      }
    
      A[k*_system_size+i] = (A[k*_system_size+i]-sum)/A[i*_system_size+i];

    }

    sum = 0.0;
    for (int r = 0; r <= k-1; r++)
    {
      sum += A[k*_system_size+r]*A[r*_system_size+k];
    }

    A[k*_system_size+k] -= sum;

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
    {
      sum = sum + A[i*_system_size+j]*x[j];
    }

    x[i] = b[i] -sum;
  }

  const unsigned int _system_size_m_1 = _system_size-1;
  x[_system_size_m_1] = x[_system_size_m_1]/\
    A[_system_size_m_1*_system_size+_system_size_m_1];

  // Backward
  for (int i = _system_size-2; i >= 0; i--)
  {
    sum = 0;
    for (unsigned int j=i+1; j < _system_size; ++j)
    {
      sum = sum +A[i*_system_size+j]*x[j];
    }
  
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
void PointIntegralSolver::_update_ghost_values()
{
  // Update off-process coefficients
  for (unsigned int i=0; i < _num_stages; i++)
  {
    for (unsigned int j=0; j < _scheme->stage_forms()[i].size(); j++)
    {
      const std::vector<boost::shared_ptr<const GenericFunction> >
	coefficients = _scheme->stage_forms()[i][j]->coefficients();
      
      for (unsigned int k = 0; k < coefficients.size(); ++k)
      {
	coefficients[k]->update();
      }
    }
  }
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::_check_forms()
{
  // Iterate over stage forms and check they include point integrals
  std::vector<std::vector<boost::shared_ptr<const Form> > >& stage_forms = 
    _scheme->stage_forms();
  for (unsigned int i=0; i < stage_forms.size(); i++)
  {
    for (unsigned int j=0; j < stage_forms[i].size(); j++)
    {

      // Check form includes at least PointIntegral
      if (!stage_forms[i][j]->ufc_form()->has_point_integrals())
      {
	dolfin_error("PointIntegralSolver.cpp",
		     "constructing PointIntegralSolver",
		     "Expecting form to have at least 1 PointIntegral");
      }

      // Num dofs per vertex
      const GenericDofMap& dofmap = *stage_forms[i][j]->function_space(0)->dofmap();
      const unsigned int dofs_per_vertex = dofmap.num_entity_dofs(0);
      const unsigned int vert_per_cell = _mesh.topology()(_mesh.topology().dim(), 0).size(0);
      
      if (vert_per_cell*dofs_per_vertex != dofmap.max_cell_dimension())
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
  
  // Get stage forms
  std::vector<std::vector<boost::shared_ptr<const Form> > >& stage_forms = 
    _scheme->stage_forms();

  // Init local stage solutions
  for (unsigned int stage=0; stage < _num_stages; stage++)
    _local_stage_solutions[stage].resize(_system_size);

  // Init coefficient index and ufcs
  _coefficient_index.resize(stage_forms.size());
  _ufcs.resize(stage_forms.size());

  // Initiate jacobian matrices
  if (_scheme->implicit())
  {
    _jac.resize(_system_size*_system_size);
  }

  // Create last stage UFC form
  _last_stage_ufc = boost::make_shared<UFC>(*_scheme->last_stage());

  // Iterate over stages and collect information
  for (unsigned int stage=0; stage < stage_forms.size(); stage++)
  {

    // Create a UFC object for first form
    _ufcs[stage].push_back(boost::make_shared<UFC>(*stage_forms[stage][0]));
      
    //  If implicit stage
    if (stage_forms[stage].size()==2)
    {
      
      // Create a UFC object for second form
      _ufcs[stage].push_back(boost::make_shared<UFC>(*stage_forms[stage][1]));
      
      // Find coefficient index for each of the two implicit forms
      for (unsigned int i=0; i < 2; i++)
      {
	for (unsigned int j=0; j<stage_forms[stage][i]->num_coefficients(); j++)
	{
	
	  // Check if stage solution is the same as coefficient j
	  if (stage_forms[stage][i]->coefficients()[j]->id() == 
	      _scheme->stage_solutions()[stage]->id())
	  {
	    _coefficient_index[stage].push_back(j);
	    break;
	  }
	}
      }
    }
  }
  
  // Build vertex map
  _vertex_map.resize(_mesh.num_vertices());
  
  // Init mesh connections
  _mesh.init(0);
  const unsigned int dim_t = _mesh.topology().dim();

  // Iterate over vertices and collect cell and local vertex information
  for (VertexIterator vert(_mesh); !vert.end(); ++vert)
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
    
    // If no cell exist where vert corresponds to local vert 0 just grab 
    // local cell 0 and find what local vert the global vert corresponds to
    if (!local_vert_found)
    {
      const Cell cell0(_mesh, vert->entities(dim_t)[0]);
      _vertex_map[vert->index()].first = cell0.index();
      
      unsigned int local_vert_index = 0;
      for (VertexIterator local_vert(cell0); !local_vert.end(); ++local_vert)
      {

	// If local vert is found
	if (vert->index()==local_vert->index())
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
PointIntegralSolver::convergence_criteria_t \
PointIntegralSolver::_simplified_newton_solve(std::vector<double>& u, 
					      std::size_t vert_ind, 
					      UFC& loc_ufc,
					      unsigned int coefficient_index, 
					      const Cell& cell)
{
  
  const Parameters& newton_solver_params = parameters("newton_solver");
  const std::string convergence_criterion = newton_solver_params["convergence_criterion"];
  const double kappa = newton_solver_params["kappa"];
  const double atol = newton_solver_params["absolute_tolerance"];
  const std::size_t max_iterations = newton_solver_params["maximum_iterations"];
  const double max_relative_residual = newton_solver_params["max_relative_residual"];
  const double relaxation = newton_solver_params["relaxation_parameter"];
  const bool report = newton_solver_params["report"];
  const unsigned int local_vert = _vertex_map[vert_ind].second;


  unsigned int newton_iterations = 0;
  double relative_residual = 1.0, residual, prev_residual = 1.0;

  // Get point integrals
  const ufc::point_integral& F_integral = *loc_ufc.default_point_integral;
        
  do
  {

    // Tabulate residual 
    Timer t_impl_tt_F("Implicit stage: tabulate_tensor (F)");
    F_integral.tabulate_tensor(&loc_ufc.A[0], loc_ufc.w(), 
			       &loc_ufc.cell.vertex_coordinates[0], 
			       local_vert);
    t_impl_tt_F.stop();
  
    // Extract vertex dofs from tabulated tensor, together with the old stage 
    // solution
    for (unsigned int row=0; row < _system_size; row++)
      _F[row] = loc_ufc.A[_local_to_local_dofs[row]];

    // Perform linear solve By forward backward substitution
    Timer forward_backward_substitution("Implicit stage: fb substituion");
    _forward_backward_subst(_jac, _F, _dx);
    forward_backward_substitution.stop();

    // Residual (residual or incremental)
    if (convergence_criterion == "residual")
      residual = _norm(_F);
    else 
      residual = _norm(_dx);

    // Check for residual convergence
    if (residual < atol)
      return converged;

    // Newton_Iterations == 0
    if (newton_iterations == 0) 
    {
      // On first iteration we need an approximation of eta. We take
      // the one from previous step and increase it slightly. This is
      // important for linear problems which only should recquire 1
      // iteration to converge.
      _eta = _eta > DOLFIN_EPS ? _eta : DOLFIN_EPS;
      _eta = std::pow(_eta, 0.8);
    }

    // 2nd time around
    else
    {
      // How fast are we converging?
      relative_residual = residual/prev_residual;

      // If we are not converging fast enough we flag the jacobian to be recomputed
      _recompute_jacobian = relative_residual >= max_relative_residual;
      
      // If we diverge
      if (relative_residual > 1)
      {
	if (report && vert_ind == 0)
	  info("Newton solver diverges with relative_residual: %.3f, residual " \
	       "%.3e, after %d iterations. Recomputing jacobian.", 
	       relative_residual, residual, newton_iterations);
	_recompute_jacobian = true;
        return diverge;
      }
      
      // We converge too slow 
      // NOTE: This code only works if we have the possibility to reduce time 
      // NOTE: step. For now this is not implemented and we therefore 
      // NOTE: out-comment it.
      /*if (residual > (kappa*atol*(1 - relative_residual)		\
		      /std::pow(relative_residual, max_iterations - newton_iterations)))
      {
	
	if (report && vert_ind == 0)
	  info("Newton solver converges too slow with relative_residual: " \
	       "%.3e at iteration %d, residual: %.2e.", relative_residual, 
	       newton_iterations, residual);
	_recompute_jacobian = true;
	return too_slow;
	}*/
      
      // Update eta 
      _eta = relative_residual/(1.0 - relative_residual);
    }
    
    // No convergence
    if (newton_iterations > max_iterations)
    {
      if (report && vert_ind == 0)
	info("Newton solver did not converged after %d iterations. "  \
	     "relative_residual: %.3f, residual: %.3e. Recomputing jacobian.", \
	     max_iterations, relative_residual, residual);
      _recompute_jacobian = true;
      return exceeds_max_iter;
    }
    
    // Update solution
    if (std::abs(1.0 - relaxation) < DOLFIN_EPS)
      for (unsigned int i=0; i < u.size(); i++)
	u[i] -= _dx[i];
    else
      for (unsigned int i=0; i < u.size(); i++)
	u[i] -= relaxation*_dx[i];
     
    // Put solution back into restricted coefficients before tabulate new residual
    for (unsigned int row=0; row < _system_size; row++)
      loc_ufc.w()[coefficient_index][_local_to_local_dofs[row]] = u[row];

    prev_residual = residual;
    newton_iterations++;

  } while(_eta*residual >= kappa*atol);
  
  return converged;
}
//-----------------------------------------------------------------------------
