#ifndef __TEMPLATE_SETTINGS_H
#define __TEMPLATE_SETTINGS_H

#include <dolfin/Settings.h>

namespace dolfin {
  
  class TemplateSettings : public Settings {
  public:
	 
	 TemplateSettings() : Settings() {
		
		add(Parameter::REAL, "my parameter", 42.0);
		
	 }
	 
  };

}

#endif
