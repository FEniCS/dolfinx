#include <iostream>

#include <dolfin.h>
#include "SettingsGlue.h"

void glueset(std::string name, dolfin::real val)
{
  dolfin::set(name.c_str(), val);
}

void glueset(std::string name, int val)
{
  dolfin::set(name.c_str(), val);
}

void glueset(std::string name, bool val)
{
  dolfin::set(name.c_str(), val);
}

void glueset(std::string name, std::string val)
{
  dolfin::set(name.c_str(), val.c_str());
}
