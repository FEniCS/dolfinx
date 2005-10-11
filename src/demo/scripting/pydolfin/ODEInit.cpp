#include <dolfin.h>
#include "ODEInit.h"

void odeinit(std::string method, int order)
{
  dolfin::dolfin_set("method", method.c_str());
  dolfin::dolfin_set("order", order);
}
