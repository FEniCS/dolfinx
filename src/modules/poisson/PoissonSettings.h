#ifndef __POISSON_SETTINGS_H
#define __POISSON_SETTINGS_H

#include "Settings.h"

namespace dolfin {
  
  class PoissonSettings : public Settings {
  public:
	 
	 PoissonSettings() : Settings() {

		add(Parameter::FUNCTION, "source", 0);
		
	 }
	 
  };

}

#endif
