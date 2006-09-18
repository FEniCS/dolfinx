#include <iostream>

#include <dolfin.h>
#include "dolfin_glue.h"

void glueset(std::string name, dolfin::real val)
{
  dolfin::set(name, val);
}

void glueset(std::string name, int val)
{
  dolfin::set(name, val);
}

void glueset(std::string name, std::string val)
{
  dolfin::set(name, val.c_str());
}

void glueset_bool(std::string name, bool val)
{
  dolfin::set(name, val);
}


dolfin::Parameter glueget(std::string name)
{
  return dolfin::get(name);
}

void load_parameters(std::string filename)
{
  dolfin::File file(filename);
  file >> dolfin::ParameterSystem::parameters;
}
