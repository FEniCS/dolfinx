// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-12
// Last changed: 2008-10-23

#ifdef ENABLE_PROJECTION_LIBRARY_H
#include "projection_library.inc"
#else

#include <dolfin/log/log.h>
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
