// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SETTINGS_HH
#define __SETTINGS_HH

#include "ParameterList.hh"
#include <dolfin.h>

///
class Settings : public ParameterList {
public:
  
  /// Constructor
  Settings() : ParameterList() {
	 
	 // Create the parameters and specify default values

	 Add("start time",              type_double, 0.0);
	 Add("final time",              type_double, 0.0);
	 Add("space dimension",         type_int,    3);
	 Add("output samples",          type_int,    10);
	 Add("debug level",             type_int,    0);
	 Add("grid file",               type_string, "grid.inp");
	 Add("output file prefix",      type_string, "dolfin");
	 Add("output file type",        type_string, "");
	 Add("output file primal",      type_string, "");
	 Add("output file dual",        type_string, "");
	 Add("output file settings",    type_string, "");
	 Add("output file residual",    type_string, "");
	 Add("output file matlab",      type_string, "");
	 Add("output file matlab data", type_string, "");
	 Add("problem",                 type_string, "unknown");
	 Add("problem description",     type_string, "Problem description is not specified.");
	 Add("display",                 type_string, "terminal");
	 Add("solve primal",            type_int,    1);
	 Add("solve dual",              type_int,    0);
	 Add("write data",              type_int,    0);
	 Add("write residuals",         type_int,    0);

	 bc_function = 0;
	 
  };
  
  /// Destructor
  ~Settings() {};

  // Initialization of parameters
  void Initialize();

  // Boundary condition function
  dolfin_bc (*bc_function) (real x, real y, real z, int node, int component);
  
};

extern Settings *settings;

#endif
