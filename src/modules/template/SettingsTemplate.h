#ifndef __SETTINGS_TEMPLATE_H
#define __SETTINGS_TEMPLATE_H

#include "Settings.h"

namespace dolfin {
  
  class SettingsTemplate : public Settings {
  public:
	 
	 SettingsTemplate() : Settings() {
		
		Add("my parameter", type_real, 42.0);
		
	 };
	 
  };

}

#endif
