#ifndef __SETTINGS_TEMPLATE_HH
#define __SETTINGS_TEMPLATE_HH

#include "Settings.hh"

///
class SettingsTemplate : public Settings {
public:
  
  SettingsTemplate() : Settings() {

	Add("my parameter", type_real, 42.0);
	 
  };
  
};

#endif
