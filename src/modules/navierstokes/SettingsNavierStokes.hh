#ifndef __SETTINGS_NAVIER_STOKES_HH
#define __SETTINGS_NAVIER_STOKES_HH

#include "Settings.hh"

///
class SettingsNavierStokes : public Settings {
public:
  
  /// Constructor
  SettingsNavierStokes() : Settings() {

	 // Create the parameters and specify default values

    Add("reynolds number",                       type_real,   100.0);
	 
    Add("turbulent flow",                        type_int, 0);
    Add("turbulent inflow",                      type_int, 0);
	 
    Add("compute reynolds stresses",             type_int, 0);
    Add("write reynolds stresses",               type_int, 0);

    Add("compute projections",                   type_int, 0);
    Add("variational multscale eddy viscosity",  type_int, 0);
    Add("smagorinsky eddy viscosity",            type_int, 0);

    Add("write couette pertubations",            type_int, 0);
    Add("write poiseuille pertubations",         type_int, 0);

    // FIXME: All the below are related to boundary conditions
	 
    Add("backward facing step",                  type_int, 0);
    Add("bluff body",                            type_int, 0);
    Add("jet",                                   type_int, 0);
    Add("driven cavity",                         type_int, 0);
    Add("LES benchmark",                         type_int, 0);

    Add("slip bc x0",                            type_int, 0);
    Add("slip bc x1",                            type_int, 0);
    Add("slip bc y0",                            type_int, 0);
    Add("slip bc y1",                            type_int, 0);
    Add("slip bc z0",                            type_int, 0);
    Add("slip bc z1",                            type_int, 0);

    Add("no slip bc x0",                         type_int, 0);
    Add("no slip bc x1",                         type_int, 0);
    Add("no slip bc y0",                         type_int, 0);
    Add("no slip bc y1",                         type_int, 0);
    Add("no slip bc z0",                         type_int, 0);
    Add("no slip bc z1",                         type_int, 0);

    Add("periodic bc x",                         type_int, 0);
    Add("periodic bc y",                         type_int, 0);
    Add("periodic bc z",                         type_int, 0);

    Add("inflow bc x0",                          type_int, 0);
    Add("outflow bc x0",                         type_int, 0);
    Add("outflow bc x1",                         type_int, 0);

    Add("couette bc y",                          type_int, 0);

    Add("driven cavity bc x0",                   type_int, 0);
    Add("driven cavity bc y1",                   type_int, 0);

    Add("dual drag bc",                          type_int, 0);
 
    Add("bnd x0",                                type_real, 0.0);
    Add("bnd x1",                                type_real, 1.0);
    Add("bnd y0",                                type_real, 0.0);
    Add("bnd y1",                                type_real, 1.0);
    Add("bnd z0",                                type_real, 0.0);
    Add("bnd z1",                                type_real, 1.0);

	 AddFunction("fx");
	 AddFunction("fy");
	 AddFunction("fz");
	 
  };
  
  /// Destructor
  ~SettingsNavierStokes() {};
};

#endif
