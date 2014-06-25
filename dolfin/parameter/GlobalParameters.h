// Copyright (C) 2009-2011 Anders Logg
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
// Modified by Fredrik Valdmanis, 2011
//
// First added:  2009-07-02
// Last changed: 2013-06-21

#ifndef __GLOBAL_PARAMETERS_H
#define __GLOBAL_PARAMETERS_H

#include "Parameters.h"
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/la/LUSolver.h>

namespace dolfin
{

  /// This class defines the global DOLFIN parameter database.

  class GlobalParameters : public Parameters
  {
  public:

    /// Constructor
    GlobalParameters();

    /// Destructor
    virtual ~GlobalParameters();

    /// Parse parameters from command-line
    virtual void parse(int argc, char* argv[]);

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("dolfin");

      // General

      // Prefix for timer tasks
      p.add("timer_prefix", "");

      // Allow extrapolation in function interpolation
      p.add("allow_extrapolation", false);

      // Use exact or linear interpolation in ODESolution::eval()
      p.add("exact_interpolation", true);

      // Output

      // Print standard output on all processes
      p.add("std_out_all_processes", true);

      // Line width relative to edge length in SVG output
      p.add("relative_line_width", 0.025);

      // Number of threads to run, 0 = run serial version
      p.add("num_threads", 0);

      // DOF reordering when running in serial
      p.add("reorder_dofs_serial", true);

      // Allowed dofs ordering libraries and set default
      std::set<std::string> allowed_dof_ordering_libraries;
      allowed_dof_ordering_libraries.insert("Boost");
      allowed_dof_ordering_libraries.insert("random");
      allowed_dof_ordering_libraries.insert("SCOTCH");
      std::string default_dof_ordering_library = "Boost";
      #ifdef HAS_SCOTCH
      default_dof_ordering_library = "SCOTCH";
      #endif

      // Add dof ordering library
      p.add("dof_ordering_library", default_dof_ordering_library,
            allowed_dof_ordering_libraries);

      // Print the level of thread support provided by the MPI library
      p.add("print_mpi_thread_support_level", false);

      // Allowed modes of ghost cell support
      std::set<std::string> allowed_ghost_modes;
      allowed_ghost_modes.insert("shared_facet");
      allowed_ghost_modes.insert("shared_vertex");
      allowed_ghost_modes.insert("None");
      p.add("ghost_mode", "None", allowed_ghost_modes);

      // Allowed partitioners (not necessarily installed)
      std::set<std::string> allowed_mesh_partitioners;
      allowed_mesh_partitioners.insert("ParMETIS");
      allowed_mesh_partitioners.insert("SCOTCH");
      allowed_mesh_partitioners.insert("Zoltan_RCB");
      allowed_mesh_partitioners.insert("Zoltan_PHG");
      allowed_mesh_partitioners.insert("None");

      // Set default graph/mesh partitioner
      std::string default_mesh_partitioner = "SCOTCH";
      #ifdef HAS_PARMETIS
        #ifndef HAS_SCOTCH
        default_mesh_partitioner = "ParMETIS";
        #endif
      #endif

      // Add mesh/graph partitioner
      p.add("mesh_partitioner", default_mesh_partitioner,
            allowed_mesh_partitioners);

      // Approaches to partitioning (following Zoltan syntax)
      // but applies to both Zoltan PHG and ParMETIS
      std::set<std::string> allowed_partitioning_approaches;
      allowed_partitioning_approaches.insert("PARTITION");
      allowed_partitioning_approaches.insert("REPARTITION");
      allowed_partitioning_approaches.insert("REFINE");

      p.add("partitioning_approach",
            "PARTITION",
            allowed_partitioning_approaches);

      #ifdef HAS_PARMETIS
      // Repartitioning parameter, determines how strongly to hold on
      // to cells when shifting between processes
      p.add("ParMETIS_repartitioning_weight", 1000.0);
      #endif

      #ifdef HAS_TRILINOS
      // Zoltan PHG partitioner parameters
      p.add("Zoltan_PHG_REPART_MULTIPLIER", 1.0);
      #endif

      // Graph coloring
      std::set<std::string> allowed_coloring_libraries;
      allowed_coloring_libraries.insert("Boost");
      #ifdef HAS_TRILINOS
      allowed_coloring_libraries.insert("Zoltan");
      #endif
      p.add("graph_coloring_library", "Boost", allowed_coloring_libraries);

      // Mesh refinement
      std::set<std::string> allowed_refinement_algorithms;
      std::string default_refinement_algorithm("recursive_bisection");
      allowed_refinement_algorithms.insert("bisection");
      allowed_refinement_algorithms.insert("iterative_bisection");
      allowed_refinement_algorithms.insert("recursive_bisection");
      allowed_refinement_algorithms.insert("regular_cut");
      p.add("refinement_algorithm",
            default_refinement_algorithm,
            allowed_refinement_algorithms);

      // Linear algebra
      std::set<std::string> allowed_backends;
      std::string default_backend("uBLAS");
      allowed_backends.insert("uBLAS");
      allowed_backends.insert("STL");
      #ifdef HAS_PETSC
      allowed_backends.insert("PETSc");
      default_backend = "PETSc";
      p.add("use_petsc_signal_handler", false);
      #endif
      #ifdef HAS_PETSC_CUSP
      allowed_backends.insert("PETScCusp");
      #endif
      #ifdef HAS_TRILINOS
      allowed_backends.insert("Epetra");
        #ifndef HAS_PETSC
        default_backend = "Epetra";
        #endif
      #endif
      p.add("linear_algebra_backend",
            default_backend,
            allowed_backends);

      // Add nested parameter sets
      p.add(KrylovSolver::default_parameters());
      p.add(LUSolver::default_parameters());

      return p;
    }

  };

  /// The global parameter database
  extern GlobalParameters parameters;

}

#endif
