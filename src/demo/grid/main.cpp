// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

// Temporary: FIXME
Display *display = new Terminal(0);

using namespace dolfin;

int main(int argc, char **argv)
{
  Grid grid;
  File file("grid.xml.gz");

  file >> grid;

  cout << grid << endl;

  grid.show();
}
