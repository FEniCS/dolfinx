#include <iostream>

#include <dolfin.h>
#include "SettingsGlue.h"

void glueset(std::string name, dolfin::Parameter val)
{
  dolfin::set(name, val);
}

dolfin::Parameter glueget(std::string name)
{
  return dolfin::get(name);
}
