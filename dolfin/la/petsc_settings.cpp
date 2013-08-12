// Copyright (C) 2013 Garth N. Wells
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
// First added:  2013-08-12
// Last changed:

#ifdef HAS_PETSC

#include "petsc_settings.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::set_petsc_option(std::string option)
{
  std::string s = "";
  set_petsc_option<std::string>(option, s);
}
//-----------------------------------------------------------------------------
void dolfin::set_petsc_option(std::string option, bool value)
{
  set_petsc_option<bool>(option, value);
}
//-----------------------------------------------------------------------------
void dolfin::set_petsc_option(std::string option, int value)
{
  set_petsc_option<int>(option, value);
}
//-----------------------------------------------------------------------------
void dolfin::set_petsc_option(std::string option, double value)
{
  set_petsc_option<double>(option, value);
}
//-----------------------------------------------------------------------------
void dolfin::set_petsc_option(std::string option, std::string value)
{
  set_petsc_option<std::string>(option, value);
}
//-----------------------------------------------------------------------------


#endif
