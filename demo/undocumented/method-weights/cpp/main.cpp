// Copyright (C) 2003-2005 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2003-10-21
// Last changed: 2009-08-11

#include <cstring>
#include <dolfin.h>

using namespace dolfin;

int main(int argc, char** argv)
{

  if (argc != 3)
  {
    dolfin::cout << "Usage: dolfin-ode method q' where method is one of" << dolfin::endl;
    dolfin::cout << "cg or dg, and q is the order" << dolfin::endl;
    return 1;
  }

  int q = atoi(argv[2]);

  if (strcmp(argv[1], "cg") == 0)
  {
    cGqMethod cGq(q);
    info(cGq, true);
  }
  else if (strcmp(argv[1], "dg") == 0)
  {
    dGqMethod dGq(q);
    info(dGq, true);
  }
  else
  {
    dolfin::cout << "Unknown method." << dolfin::endl;
    return 1;
  }

  return 0;
}
