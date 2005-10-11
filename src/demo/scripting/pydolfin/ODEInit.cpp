#include <dolfin.h>
#include "ODEInit.h"

void odeinit()
{
  dolfin::dolfin_set("method", "cg");
  dolfin::dolfin_set("order", 1);
}
