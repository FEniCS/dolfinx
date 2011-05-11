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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2003-06-04
// Last changed: 2008-11-16

#include <cstring>
#include <dolfin.h>

using namespace dolfin;

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    dolfin::cout << "Usage: dolfin-quadrature rule n' where rule is one of" << dolfin::endl;
    dolfin::cout << "gauss, radau, lobatto, and n is the number of points" << dolfin::endl;
    return 1;
  }

  int n = atoi(argv[2]);

  if (strcmp(argv[1], "gauss") == 0)
  {
    GaussQuadrature q(n);
    info(q);
  }
  else if (strcmp(argv[1], "radau") == 0)
  {
    RadauQuadrature q(n);
    info(q);
  }
  else if (strcmp(argv[1], "lobatto") == 0)
  {
    LobattoQuadrature q(n);
    info(q);
  }
  else
  {
    dolfin::cout << "Unknown quadrature rule." << dolfin::endl;
    return 1;
  }

  return 0;
}
