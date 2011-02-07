// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-07-02
// Last changed: 2011-02-07

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

      // JIT compiler
      p.add("optimize_form", false);                         // Use optimization -O2 when compiling generated code
      p.add("optimize_use_dof_map_cache", false);            // Store dof maps in cache for reuse
      p.add("optimize_use_tensor_cache", false);             // Store tensors in cache for reuse
      p.add("optimize", false);                              // All of the above

      // Multi-core
      p.add("num_threads", 0);                               // Number of threads to run, 0 = run serial version

      // Graph partitioner
      p.add("mesh_partitioner", "ParMETIS");

      // Mesh refinement
      std::set<std::string> allowed_refinement_algorithms;
      std::string default_refinement_algorithm("recursive_bisection");
      allowed_refinement_algorithms.insert("bisection");
      allowed_refinement_algorithms.insert("iterative_bisection");
      allowed_refinement_algorithms.insert("recursive_bisection");
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
#ifdef HAS_TRILINOS
      allowed_backends.insert("Epetra");
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

      // Floating-point precision (only relevant when using GMP)
      #ifdef HAS_GMP
      p.add("floating_point_precision", 30);                 // Use higher precision for GMP (can be changed)
      #else
      p.add("floating_point_precision", 16);                 // Use double precision when GMP is not available
      #endif

      return p;
    }

  };

  /// The global parameter database
  extern GlobalParameters parameters;

}

#endif
