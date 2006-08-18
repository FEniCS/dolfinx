#include <iostream>

#include <dolfin.h>
#include "dolfin_glue.h"

// This function is required when linking to libraries which contain 
// FORTRAN code
extern "C"
{
  int MAIN__()  { return(0); }
}

// These functions are a fix to avoid undefined symbol errors
extern "C"
{
  int umfpack_di_transpose()  { return(0); }
  int umfpack_di_solve()  { return(0); }
}


void glueset(std::string name, dolfin::real val)
{
  dolfin::set(name, val);
}

void glueset(std::string name, int val)
{
  dolfin::set(name, val);
}

void glueset(std::string name, bool val)
{
  dolfin::set(name, val);
}

void glueset(std::string name, std::string val)
{
  dolfin::set(name, val.c_str());
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
