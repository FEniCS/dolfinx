#ifndef __SETTINGS_CONV_DIFF_HH
#define __SETTINGS_CONV_DIFF_HH

#include "Settings.h"

namespace dolfin {
  
  class SettingsConvDiff : public Settings {
  public:
	 
	 SettingsConvDiff() : Settings() {
		
		AddFunction("source");
		AddFunction("diffusivity");
		AddFunction("x-convection");
		AddFunction("y-convection");
		AddFunction("z-convection");
	 
	 };
	 
  };

}

#endif
