// Copyright (C) 2006 Ola Skavhaug
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
// Modified by Anders Logg, 2007.
//
// First added:  2006-11-29
// Last changed: 2009-09-15

#include <dolfin.h>

using namespace dolfin;

int main()
{
  Mesh mesh("../mesh2D.xml.gz");

  // Read mesh function from file (new style)
  File in("../meshfunction.xml");
  MeshFunction<double> f(mesh);
  in >> f;

  // Write mesh function to file (new style)
  File out("meshfunction_out.xml");
  out << f;

  // Plot mesh function
  plot(f);
}
