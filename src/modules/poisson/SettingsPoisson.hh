#ifndef __SETTINGS_POISSON_HH
#define __SETTINGS_POISSON_HH

#include "Settings.hh"

///
class SettingsPoisson : public Settings {
public:
  
  SettingsPoisson() : Settings() {

	 AddFunction("source");
	 
  };
  
};

#endif
