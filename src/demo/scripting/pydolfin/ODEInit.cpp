#include <dolfin.h>
#include "ODEInit.h"

void odeinit()
{
  dolfin::dolfin_set("method", "mcg");
}
