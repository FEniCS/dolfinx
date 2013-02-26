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
// Last changed: 2013-02-26

#include <cmath>

#include <dolfin/log/log.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/Constant.h>
#include <dolfin/la/GenericVector.h>

#include "assemble.h"
#include "solve.h"
#include "BoundaryCondition.h"
#include "ButcherScheme.h"
#include "PointIntegralSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PointIntegralSolver::PointIntegralSolver(boost::shared_ptr<ButcherScheme> scheme) : 
  _scheme(scheme)
{
  _check_forms();
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::step(double dt)
{
  dolfin_assert(dt > 0.0);

  // Update time constant of scheme
  *_scheme->dt() = dt;

  // Get scheme data
  std::vector<std::vector<boost::shared_ptr<const Form> > >& stage_forms = \
    _scheme->stage_forms();
  std::vector<boost::shared_ptr<Function> >& stage_solutions = _scheme->stage_solutions();
  std::vector<const BoundaryCondition* > bcs = _scheme->bcs();
  
  // Extract mesh
  const Mesh& mesh = stage_forms[0][0]->mesh();

  // Iterate over vertices
  for (VertexIterator vert(mesh); !vert.end(); ++vert)
  {

    // Iterate over stage forms
    for (unsigned int i=0; i < stage_forms.size(); i++)
    {
      // Check if we have an explicit stage (only 1 form)
      if (stage_forms[i].size()==1)
      {

	// Just do an assemble
	_assembler.assemble(*stage_solutions[i]->vector(), *stage_forms[i][0]);
      
	// Apply boundary conditions
	for (unsigned int j = 0; j < bcs.size(); j++)
	{
	  dolfin_assert(bcs[j]);
	  bcs[j]->apply(*stage_solutions[i]->vector());
	}
      
      }
    
      // or an implicit stage (2 forms)
      else
      {
	// FIXME: Include solver parameters
	// Do a nonlinear solve
	solve(*stage_forms[i][0] == 0, *stage_solutions[i], bcs, *stage_forms[i][1]);
      }
    }
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
      //const ufc::dofmap& ufc_dofmap = _stage_forms[i][j]->functionspace(0)->dofmap()
      //const unsigned int dofs_per_vertex = ufc_dofmap->num_entity_dofs(0);
      //const unsigned int vert_per_cell = mesh.topology()(top_dim, 0).size(0);
      //
      //if (vert_per_cell*dofs_per_vertex != _ufc_dofmap->max_local_dimension())
      //{
      //	dolfin_error("PointIntegralSolver.cpp",
      //		     "constructing PointIntegralSolver",
      //		     "Expecting test space to only have dofs on vertices");
      //}

    }
  }
}
//-----------------------------------------------------------------------------
void PointIntegralSolver::_build_vertex_map()
{
  
}
//-----------------------------------------------------------------------------
