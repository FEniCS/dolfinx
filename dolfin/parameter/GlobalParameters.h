// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-07-02
// Last changed: 2010-04-15

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

      // Linear algebra
      #ifdef HAS_PETSC
      p.add("linear_algebra_backend", "PETSc");              // Use PETSc if available
      #else
      p.add("linear_algebra_backend", "uBLAS");              // Otherwise, use uBLAS
      #endif

      // Solvers
      p.add("error_on_nonconvergence", true);                // Issue an error if solver does not converge (otherwise warning)

      // Floating-point precision (only relevant when using GMP)
      #ifdef HAS_GMP
      p.add("floating_point_precision", 30);                 // Use higher precision for GMP (can be changed)
      #else
      p.add("floating_point_precision", 16);                 // Use double precision when GMP is not available
      #endif

      // Graph partitioner
      p.add("mesh_partitioner", "ParMETIS");

      return p;
    }

  };

  /// The global parameter database
  extern GlobalParameters parameters;

}

#endif
