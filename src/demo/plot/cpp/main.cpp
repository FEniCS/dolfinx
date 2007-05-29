// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-29
// Last changed: 2007-05-29

#include <dolfin.h>

using namespace dolfin;

int main()
{
  Mesh mesh("cow.xml.gz");
  plot(mesh);
}
