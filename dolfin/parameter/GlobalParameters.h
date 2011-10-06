// Copyright (C) 2009 Anders Logg
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
// Last changed: 2011-09-13

#ifndef __GLOBAL_PARAMETERS_H
#define __GLOBAL_PARAMETERS_H

#include "Parameters.h"

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
      p.add("timer_prefix", "");                             // Prefix for timer tasks
      p.add("plot_filename_prefix", "dolfin_plot_data");     // Prefix for temporary plot files
      p.add("allow_extrapolation", false);                   // Allow extrapolation in function interpolation
      p.add("exact_interpolation", true);                    // Use exact or linear interpolation in ODESolution::eval()

      // Ouput
      p.add("std_out_all_processes", true);                  // Print stanard output on all processes

      // JIT compiler
      p.add("optimize_form", false);                         // Use optimization -O2 when compiling generated code
      p.add("optimize_use_dofmap_cache", false);             // Store dof maps in cache for reuse
      p.add("optimize_use_tensor_cache", false);             // Store tensors in cache for reuse
      p.add("optimize", false);                              // All of the above

      // Multi-core
      p.add("num_threads", 0);                               // Number of threads to run, 0 = run serial version

      // Graph partitioner
      p.add("mesh_partitioner", "ParMETIS");

      // Graph coloring
      p.add("graph_coloring_library", "Boost");

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
      #endif
      //#ifdef PETSC_HAVE_CUSP // FIXME: Find functioning test
      allowed_backends.insert("PETScCusp");
      //#endif
      #ifdef HAS_TRILINOS
      allowed_backends.insert("Epetra");
        #ifndef HAS_PETSC
        default_backend = "Epetra";
        #endif
      #endif
      #ifdef HAS_MTL4
      allowed_backends.insert("MTL4");
      #endif
      #ifdef HAS_TRILINOS
      allowed_backends.insert("Epetra");
      #endif
      p.add("linear_algebra_backend",
            default_backend,
            allowed_backends);

      return p;
    }

  };

  /// The global parameter database
  extern GlobalParameters parameters;

}

#endif
