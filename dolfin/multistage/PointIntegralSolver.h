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
// Last changed: 2013-05-16

#ifndef __POINTINTEGRALSOLVER_H
#define __POINTINTEGRALSOLVER_H

#include <vector>
#include <armadillo>
#include <boost/shared_ptr.hpp>

#include <dolfin/common/Variable.h>
#include <dolfin/function/FunctionAXPY.h>
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
    /// FIXME: Include version where one can pass a Solver and/or Parameters
    PointIntegralSolver(boost::shared_ptr<MultiStageScheme> scheme);

    /// Step solver with time step dt
    void step(double dt);

    /// Step solver an interval using dt as time step
    void step_interval(double t0, double t1, double dt);

    /// Return the MultiStageScheme
    boost::shared_ptr<MultiStageScheme> scheme()const
    {return _scheme;}

    /// Default parameter values
    static Parameters default_parameters()
    {
      
      Parameters p("point_integral_solver");

      // Get default parameters from NewtonSolver
      p.add(NewtonSolver::default_parameters());
      p("newton_solver").add("reuse_jacobian", true);
      p("newton_solver").add("iterations_to_retabulate_jacobian", 4);
      
      return p;
    }

  private:

    // Update ghost values
    void _update_ghost_values();

    // Check the forms making sure they only include piecewise linear
    // test functions
    void _check_forms();

    // Build map between vertices, cells and the correspondning local vertex
    // and initialize UFC data for each form
    void _init();

    // Solve an explicit stage
    void _solve_explicit_stage(std::size_t vert_ind,
			       unsigned int stage);

    // Solve an implicit stage
    void _solve_implict_stage(std::size_t vert_ind,
			      unsigned int stage);

    // The MultiStageScheme
    boost::shared_ptr<MultiStageScheme> _scheme;

    // Reference to mesh
    const Mesh& _mesh;

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
    
    // Vertex map between vertices, cells and corresponding local vertex
    std::vector<std::pair<std::size_t, unsigned int> > _vertex_map;

    // Local to global dofs used when solution is fanned out to global vector
    std::vector<dolfin::la_index> _local_to_global_dofs;
  
    // Local stage solutions 
    std::vector<arma::vec> _local_stage_solutions;

    // UFC objects, one for each form
    std::vector<std::vector<boost::shared_ptr<UFC> > > _ufcs;

    // Solution coefficient index in form
    std::vector<std::vector<int> > _coefficient_index;

    // Flag for retabulation of J
    bool _retabulate_J;
    
    // Jacobian and LU factorized jacobian matrices
    arma::mat _J;
    arma::mat _J_L;
    arma::mat _J_U;

  };

}

#endif
