// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

// Temporary: FIXME
Display *display = new Terminal(0);

using namespace dolfin;

int main(int argc, char **argv)
{
  File file("example.dolfin");
  Vector x;

  file >> x;

  // Temporary: FIXME
  delete display;
}
