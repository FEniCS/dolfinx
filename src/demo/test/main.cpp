// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  dolfin_set("output", "plain text");
  cout << "Test" << " test" << endl;

  PETScManager::init();

  NewMatrix A(100, 100);
  cout << A << endl;

  return 0;
}
