// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace std;
using namespace dolfin;

int main()
{
  Grid grid;
  File file("grid.xml.gz");

  file >> grid;

  cout << grid << endl;
  
  grid.show();

  return 0;
}
