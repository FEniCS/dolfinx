// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SETTINGS_H
#define __SETTINGS_H

#include <dolfin/constants.h>
#include <dolfin/ParameterList.h>

namespace dolfin {
  
  ///
  class Settings : public ParameterList {
  public:
	 
	 /// Constructor
	 Settings() : ParameterList() {
		
		// Create default parameters
		
		add(Parameter::REAL, "start time", 0.0);
		add(Parameter::REAL, "end time",   10.0);

		add(Parameter::BCFUNCTION, "boundary condition", 0);
		
	 }
	 
  };
  
}

#endif
