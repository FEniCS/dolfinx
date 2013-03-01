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
// Last changed: 2013-03-01

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
  
  // Local stage solutions
  std::vector<arma::vec> _local_solutions(_scheme->stage_solutions().size());
  for (std::vector<arma::vec>::iterator it = _local_solutions.begin(); 
       it != _local_solutions.end(); it++)
    it->resize(N);

  /// Local_dofs to be used in tabulate entity dofs
  std::vector<std::size_t> local_dofs;

  // Iterate over vertices
  Progress p("Solving local point integral problems", mesh.num_vertices());
  for (std::size_t vert_ind; vert_ind< mesh.num_vertices(); ++vert_ind)
  {

    // Cell containing vertex
    const Cell cell(mesh, _vertex_map[vert_ind].first);
    
    // Local vertex ind
    const unsigned int local_vert = _vertex_map[vert_ind].second;

    // Tabulate local-local dofmap
    dofmap.tabulate_entity_dofs(local_dofs, 0, local_vert);
	
    // Iterate over stage forms
    for (unsigned int i=0; i < _ufcs.size(); i++)
    {
      // Check if we have an explicit stage (only 1 form)
      if (_ufcs[i].size()==1)
      {

	// Point integral
	const ufc::point_integral& integral = *_ufcs[i][0]->default_point_integral;

	// Update to current cell
	_ufcs[i][0]->update(cell);

	// FIXME: Shold we include logics about empty dofmaps?
	
	// Tabulate cell tensor
	integral.tabulate_tensor(&_ufcs[i][0]->A[0], _ufcs[i][0]->w(), \
				 _ufcs[i][0]->cell, local_vert);

	// Extract vertex dofs from tabulated tensor
	unsigned int j=0;
	for (std::vector<std::size_t>::const_iterator it = local_dofs.begin(); \
	     it != local_dofs.end(); it++)
	  _local_solutions[j++] = *it;

      }
    
      // or an implicit stage (2 forms)
      else
      {
	// FIXME: Include solver parameters
	// Do a nonlinear solve
	//solve(*stage_forms[i][0] == 0, *stage_solutions[i], bcs, *stage_forms[i][1]);
      }
    }

    p++;
  }


  // Do the last stage
  FunctionAXPY last_stage = _scheme->last_stage()*dt;
  
  // Update solution with last stage
  GenericVector& solution_vector = *_scheme->solution()->vector();
  
  // Start from item 2 and axpy 
  for (std::vector<std::pair<double, const Function*> >::const_iterator \
	 it=last_stage.pairs().begin();
       it!=last_stage.pairs().end(); it++)
  {
    solution_vector.axpy(it->first, *(it->second->vector()));
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
  std::vector<std::vector<boost::shared_ptr<const Form> > >& stage_forms = \
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
  std::vector<std::vector<boost::shared_ptr<const Form> > >& stage_forms = \
    _scheme->stage_forms();

  // Create a UFC object for each form
  for (unsigned int i=0; i < stage_forms.size(); i++)
    for (unsigned int j=0; j < stage_forms.size(); j++)
      _ufcs[i].push_back(boost::make_shared<UFC>(*stage_forms[i][j]));
  
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
