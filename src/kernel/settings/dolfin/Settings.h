// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-05-06
// Last changed: 2005

#ifndef __SETTINGS_H
#define __SETTINGS_H

#include <iostream>
#include <dolfin/constants.h>
#include <dolfin/ParameterList.h>

namespace dolfin
{
  
  /// Pre-defined global parameters within DOLFIN.

  class Settings : public ParameterList
  {
  public:
    
    Settings() : ParameterList()
    {
      dolfin_info("Initializing DOLFIN parameter database.");

      // General parameters

      add(Parameter::REAL, "progress step", 0.1);
      add(Parameter::BOOL, "save each mesh", false);

      // Parameters for multi-adaptive solver

      add(Parameter::BOOL, "fixed time step", false);
      add(Parameter::BOOL, "solve dual problem", false);
      add(Parameter::BOOL, "save solution", true);
      add(Parameter::BOOL, "adaptive samples", false);
      add(Parameter::BOOL, "automatic modeling", false);
      add(Parameter::BOOL, "implicit", false);
      add(Parameter::BOOL, "matrix piecewise constant", true);
      add(Parameter::BOOL, "monitor convergence", false);

      add(Parameter::INT, "order", 1);
      add(Parameter::INT, "number of samples", 101);
      add(Parameter::INT, "sample density", 1);
      add(Parameter::INT, "maximum iterations", 100);
      add(Parameter::INT, "maximum local iterations", 2);
      add(Parameter::INT, "average samples", 1000);

      add(Parameter::REAL, "tolerance", 0.1);
      add(Parameter::REAL, "start time", 0.0);
      add(Parameter::REAL, "end time", 10.0);      
      add(Parameter::REAL, "discrete tolerance", 0.001);
      add(Parameter::REAL, "discrete tolerance factor", 0.001);
      add(Parameter::REAL, "initial time step", 0.01);
      add(Parameter::REAL, "maximum time step", 0.1);
      add(Parameter::REAL, "partitioning threshold", 0.1);
      add(Parameter::REAL, "interval threshold", 0.9);
      add(Parameter::REAL, "time step conservation", 5.0);
      add(Parameter::REAL, "sparsity check increment", 0.01);
      add(Parameter::REAL, "average length", 0.1);
      add(Parameter::REAL, "average tolerance", 0.1);
      
      add(Parameter::STRING, "method", "cg");
      add(Parameter::STRING, "solver", "default");
      add(Parameter::STRING, "linear solver", "default");
      add(Parameter::STRING, "file name", "primal.m");

      // Parameters for homotopy solver
      add(Parameter::INT,    "homotopy maximum size", std::numeric_limits<int>::max());
      add(Parameter::INT,    "homotopy maximum degree", std::numeric_limits<int>::max());
      add(Parameter::REAL,   "homotopy solution tolerance", 1e-12);
      add(Parameter::REAL,   "homotopy divergence tolerance", 10.0);
      add(Parameter::BOOL,   "homotopy randomize", true);
      add(Parameter::BOOL,   "homotopy monitoring", false);
      add(Parameter::STRING, "homotopy solution file", "solution.data");
      
    }
    
  };
  
}

#endif
