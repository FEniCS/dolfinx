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
    
    Settings() : ParameterList() {
      
      add(Parameter::REAL, "start time", 0.0);
      add(Parameter::REAL, "end time",   10.0);
      
      add(Parameter::INT, "max no krylov restarts", 100);
      add(Parameter::INT, "max no stored krylov vectors", 100);
      add(Parameter::INT, "max no cg iterations", 1000); 
      
      add(Parameter::BCFUNCTION, "boundary condition", 0);
      
    }
    
  };
  
}

#endif
