// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SETTINGS_HH
#define __SETTINGS_HH

#include "ParameterList.h"
#include <dolfin.h>

// Data types
typedef double real;
enum bc_type { dirichlet , neumann};

namespace dolfin {
  
  // Boundary conditions
  class dolfin_bc{
  public:
	 dolfin_bc(){ type = neumann; val = 0.0; }  
	 bc_type type;
	 real val;
  };
  
  ///
  class Settings : public ParameterList {
  public:
	 
	 /// Constructor
	 Settings() : ParameterList() {
		
		// Create the parameters and specify default values
		
		add( Parameter::REAL,   "start time",              0.0 );
		add( Parameter::REAL,   "final time",              0.0 );
		add( Parameter::INT,    "space dimension",         3 );
		add( Parameter::INT,    "output samples",          10 );
		add( Parameter::INT,    "debug level",             0 );
		add( Parameter::STRING, "grid file",               "grid.inp" );
		add( Parameter::STRING, "output file prefix",      "dolfin" );
		add( Parameter::STRING, "output file type",        "" );
		add( Parameter::STRING, "output file primal",      "" );
		add( Parameter::STRING, "output file dual",        "" );
		add( Parameter::STRING, "output file settings",    "" );
		add( Parameter::STRING, "output file residual",    "" );
		add( Parameter::STRING, "output file matlab",      "" );
		add( Parameter::STRING, "output file matlab data", "" );
		add( Parameter::STRING, "problem",                 "unknown");
		add( Parameter::STRING, "problem description",     "Problem description is not specified." );
		add( Parameter::STRING, "display",                 "terminal" );
		add( Parameter::INT,    "solve primal",            1 );
		add( Parameter::INT,    "solve dual",              0 );
		add( Parameter::INT,    "write data",              0 );
		add( Parameter::INT,    "write residuals",         0 );
	 	
		bc_function = 0;
		
	 };
	 
	 /// Destructor
	 ~Settings() {};
	 
	 // Initialization of parameters
	 void init();
	 
	 // Boundary condition function
	 dolfin_bc (*bc_function) (real x, real y, real z, int node, int component);
	 
  };
  
  extern Settings *settings;

}
  
#endif
