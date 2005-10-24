#include <dolfin.h>
#include "SettingsGlue.h"

void glueset(std::string name, dolfin::real val)
{
  dolfin::dolfin_set(name.c_str(), val);
}

void glueset(std::string name, int val)
{
  dolfin::dolfin_set(name.c_str(), val);
}

void glueset(std::string name, bool val)
{
  dolfin::dolfin_set(name.c_str(), val);
}

void glueset(std::string name, std::string val)
{
  dolfin::dolfin_set(name.c_str(), val.c_str());
}
