#ifndef __SETTINGS_POISSON_HH
#define __SETTINGS_POISSON_HH

#include "Settings.hh"

namespace dolfin {
  
  class SettingsPoisson : public Settings {
  public:
	 
	 SettingsPoisson() : Settings() {
		
		AddFunction("source");
		
	 };
	 
  };

}

#endif
