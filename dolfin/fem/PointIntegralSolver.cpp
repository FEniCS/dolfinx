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
// Last changed: 2013-03-06

#include <cmath>
#include <armadillo>
#include <boost/make_shared.hpp>

#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/Constant.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/GenericDofMap.h>

#include "UFC.h"
#include "ButcherScheme.h"

#include "PointIntegralSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PointIntegralSolver::PointIntegralSolver(boost::shared_ptr<ButcherScheme> scheme) : 
  _scheme(scheme)
{
  _check_forms();
  _init();
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::step(double dt)
{
  dolfin_assert(dt > 0.0);

  // Update time constant of scheme
  *_scheme->dt() = dt;

  // Extract mesh
  const Mesh& mesh = _scheme->stage_forms()[0][0]->mesh();

  // Collect ref to dof map only need one as we require same trial and test 
  // space for all forms
  const GenericDofMap& dofmap = *_scheme->stage_forms()[0][0]->function_space(0)->dofmap();
  
  // Get size of system (num dofs per vertex)
  const unsigned int N = dofmap.num_entity_dofs(0);
  const unsigned int dof_offset = mesh.type().num_entities(0);
  const unsigned int num_stages = _scheme->stage_forms().size();

  // Local solution vector at start of time step
  arma::vec u0;
  u0.set_size(N);

  // Local stage solutions
  std::vector<arma::vec> local_stage_solutions(_scheme->stage_solutions().size());
  for (unsigned int stage=0; stage<num_stages; stage++)
    local_stage_solutions[stage].set_size(N);

  /// Local to local dofs to be used in tabulate entity dofs
  std::vector<std::size_t> local_to_local_dofs(N);

  // Local to global dofs used when solution is fanned out to global vector
  std::vector<dolfin::la_index> local_to_global_dofs(N);
  
  // Iterate over vertices
  Progress p("Solving local point integral problems", mesh.num_vertices());
  for (std::size_t vert_ind=0; vert_ind< mesh.num_vertices(); ++vert_ind)
  {

    // Cell containing vertex
    const Cell cell(mesh, _vertex_map[vert_ind].first);

    // Get all dofs for cell
    // FIXME: Shold we include logics about empty dofmaps?
    const std::vector<dolfin::la_index>& cell_dofs = dofmap.cell_dofs(cell.index());

    // Local vertex ind
    const unsigned int local_vert = _vertex_map[vert_ind].second;

    // Tabulate local-local dofmap
    dofmap.tabulate_entity_dofs(local_to_local_dofs, 0, local_vert);
    
    // Fill local to global dof map
    for (unsigned int row=0; row<N; row++)
    {
      local_to_global_dofs[row] = cell_dofs[local_to_local_dofs[row]];
    }

    // Iterate over stage forms
    for (unsigned int stage=0; stage<num_stages; stage++)
    {

      // Check if we have an explicit stage (only 1 form)
      if (_ufcs[stage].size()==1)
      {

	// Point integral
	const ufc::point_integral& integral = *_ufcs[stage][0]->default_point_integral;

	// Update to current cell
	_ufcs[stage][0]->update(cell);

	// Tabulate cell tensor
	integral.tabulate_tensor(&_ufcs[stage][0]->A[0], _ufcs[stage][0]->w(), 
				 &_ufcs[stage][0]->cell.vertex_coordinates[0], 
				 local_vert);

	// Extract vertex dofs from tabulated tensor and put them into the local 
	// stage solution vector
	// Extract vertex dofs from tabulated tensor
	for (unsigned int row=0; row < N; row++)
	  local_stage_solutions[stage](row) = _ufcs[stage][0]->A[local_to_local_dofs[row]];

	// Put solution back into global stage solution vector
	_scheme->stage_solutions()[stage]->vector()->set(
			    local_stage_solutions[stage].memptr(), N, 
			    &local_to_global_dofs[0]);
      }
    
      // or an implicit stage (2 forms)
      else
      {
	
	// Local solution
	arma::vec& u = local_stage_solutions[stage];
	
	// FIXME: Include as some sort of solver parameters
	unsigned int newton_iteration = 0;
	bool newton_converged = false;
	const std::size_t maxiter = 10;
	//const bool recompute_jacobian = true;
	const double relaxation = 1.0;
	const std::string convergence_criterion = "residual";
	const double rtol = 1e-10;
	const double atol = 1e-10;
	const bool report = false;
	//const int num_threads = 0;

	/// Most recent residual and intitial residual
	double residual = 1.0;
	double residual0 = 1.0;
	double relative_residual;
      
	//const double relaxation = 1.0;
      
	// Initialize la structures
	arma::mat J(N, N);
	arma::vec F;
	arma::vec dx;
	F.set_size(N);
	dx.set_size(N);
      
	// Get point integrals
	const ufc::point_integral& F_integral = *_ufcs[stage][0]->default_point_integral;
	const ufc::point_integral& J_integral = *_ufcs[stage][1]->default_point_integral;
      
	// Update to current cell. This only need to be done once for each stage and 
	// vertex
	_ufcs[stage][0]->update(cell);
	_ufcs[stage][1]->update(cell);

	// Tabulate an initial residual solution
	F_integral.tabulate_tensor(&_ufcs[stage][0]->A[0], _ufcs[stage][0]->w(), 
				   &_ufcs[stage][0]->cell.vertex_coordinates[0], 
				   local_vert);

	// Extract vertex dofs from tabulated tensor, together with the old stage 
	// solution
	for (unsigned int row=0; row < N; row++)
	{
	  F(row) = _ufcs[stage][0]->A[local_to_local_dofs[row]];

	  // Grab old value of stage solution as an initial start value. This 
	  // value was also used to tabulate the initial value of the F_integral above 
	  // and we therefore just grab it from the restricted coeffcients
	  u(row) = _ufcs[stage][0]->w()[_coefficient_index[stage][0]][local_to_local_dofs[row]];
	}

	// Start iterations
	while (!newton_converged && newton_iteration < maxiter)
	{
        
	  // Tabulate Jacobian
	  J_integral.tabulate_tensor(&_ufcs[stage][1]->A[0], _ufcs[stage][1]->w(), 
				     &_ufcs[stage][1]->cell.vertex_coordinates[0], 
				     local_vert);
      
	  // Extract vertex dofs from tabulated tensor
	  for (unsigned int row=0; row < N; row++)
	    for (unsigned int col=0; col < N; col++)
	      J(row, col) = _ufcs[stage][1]->A[local_to_local_dofs[row]*dof_offset*N+
					       local_to_local_dofs[col]];

          // Perform linear solve and update total number of Krylov iterations
          arma::solve(dx, J, F);
	  
	  // Compute resdiual
	  if (convergence_criterion == "residual")
	    residual = arma::norm(F, 2);
	  else if (convergence_criterion == "incremental")
	    residual = arma::norm(dx, 2);
	  else
	    error("Unknown Newton convergence criterion");

          // If initial residual
          if (newton_iteration == 0)
	    residual0 = residual;
	  
	  // Relative residual
	  relative_residual = residual / residual0;
	  
	  // Update solution
          if (std::abs(1.0 - relaxation) < DOLFIN_EPS)
            u -= dx;
          else
            u -= relaxation*dx;
        
	  // Update number of iterations
	  ++newton_iteration;
	  
	  // Put solution back into restricted coefficients before tabulate new residual
	  for (unsigned int row=0; row < N; row++)
	    _ufcs[stage][0]->w()[_coefficient_index[stage][0]][local_to_local_dofs[row]] = u(row);

	  // Tabulate new residual 
	  F_integral.tabulate_tensor(&_ufcs[stage][0]->A[0], _ufcs[stage][0]->w(), 
				     &_ufcs[stage][0]->cell.vertex_coordinates[0], 
				     local_vert);

	  // Extract vertex dofs from tabulated tensor
	  for (unsigned int row=0; row < N; row++)
	    F(row) = _ufcs[stage][0]->A[local_to_local_dofs[row]];

	  // Output iteration number and residual (only first vertex)
	  if (report && (newton_iteration > 0) && (vert_ind == 0))
	  {
	    info("Point solver newton iteration %d: r (abs) = %.3e (tol = %.3e) "\
		 "r (rel) = %.3e (tol = %.3e)", newton_iteration, residual, atol, 
		 relative_residual, rtol);
	  }
	  
	  // Return true of convergence criterion is met
	  if (relative_residual < rtol || residual < atol)
	    newton_converged = true;

	  // If there is a solution coefficient in the jacobian form
	  if (_coefficient_index[stage].size()==2)
	  {
	    // Put solution back into restricted coefficients before tabulate new jacobian
	    for (unsigned int row=0; row < N; row++)
	      _ufcs[stage][1]->w()[_coefficient_index[stage][1]][local_to_local_dofs[row]] = u(row);
	  }

	}
      
        if (newton_converged)
        {
          // Put solution back into global stage solution vector
          _scheme->stage_solutions()[stage]->vector()->set(u.memptr(), u.size(), 
							   &local_to_global_dofs[0]);
	}
        else
        {
          info("Last iteration before error %d: r (abs) = %.3e (tol = %.3e) "
	       "r (rel) = %.3e (tol = %.3e)", newton_iteration, residual, atol, 
	       relative_residual, rtol);
	  error("Newton solver in PointIntegralSolver did not converge.");
        }
      }
    }

    // Get local u0 solution and add the stage derivatives
    _scheme->solution()->vector()->get_local(&u0[0], u0.size(), 
					     &local_to_global_dofs[0]);
    
    // Do the last stage and put back into solution vector
    FunctionAXPY last_stage = _scheme->last_stage()*dt;
  
    // Axpy local solution vectors
    for (unsigned int stage=0; stage < num_stages; stage++)
      u0 += last_stage.pairs()[stage].first*local_stage_solutions[stage] ;
    
    // Update global solution with last stage
    _scheme->solution()->vector()->set(u0.memptr(), local_to_global_dofs.size(), 
				       &local_to_global_dofs[0]);
    
    p++;
  }

  // Update time
  const double t = *_scheme->t();
  *_scheme->t() = t + dt;
  
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
      const Mesh& mesh = *stage_forms[i][j]->function_space(0)->mesh();
      const GenericDofMap& dofmap = *stage_forms[i][j]->function_space(0)->dofmap();
      const unsigned int dofs_per_vertex = dofmap.num_entity_dofs(0);
      const unsigned int vert_per_cell = mesh.topology()(mesh.topology().dim(), 0).size(0);
      
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

  // Init coefficient index and ufcs
  _coefficient_index.resize(stage_forms.size());
  _ufcs.resize(stage_forms.size());

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
      for (unsigned int i=0; i<2; i++)
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
  
  // Extract mesh
  const Mesh& mesh = stage_forms[0][0]->mesh();
  _vertex_map.resize(mesh.num_vertices());
  
  // Init mesh connections
  mesh.init(0);
  const unsigned int dim_t = mesh.topology().dim();

  // Iterate over vertices and collect cell and local vertex information
  for (VertexIterator vert(mesh); !vert.end(); ++vert)
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
      const Cell cell0(mesh, vert->entities(dim_t)[0]);
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
