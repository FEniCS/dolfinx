#ifdef ENABLE_PROJECTION_LIBRARY_H
#include "projection_library.h"
#else

#include "ProjectionLibrary.h"

dolfin::Form* dolfin::ProjectionLibrary::create_projection_a(const char* signature)
{
  error("Projection library not available, try building DOLFIN with enableProjectionLibrary=yes.");
  return 0;
}

dolfin::Form* dolfin::ProjectionLibrary::create_projection_L(const char* signature, Function& f)
{
  error("Projection library not available, try building DOLFIN with enableProjectionLibrary=yes.");
  return 0;
}

#endif
