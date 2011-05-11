// Copyright (C) 2006-2009 Johan Hake
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
// First added:  2006-02-06
// Last changed: 2011-03-14

//-----------------------------------------------------------------------------
// Include code to generate a __swigversion__ and a __dolfinversion__ 
// attributes, from defines during compile time, to the cpp module
//-----------------------------------------------------------------------------
%inline %{
int dolfin_swigversion() { return  SWIGVERSION; }
std::string dolfin_version() {return DOLFIN_VERSION;}
%}

%pythoncode %{
tmp = hex(dolfin_swigversion())
__swigversion__ = "%d.%d.%d"%(tuple(map(int, [tmp[-5], tmp[-3], tmp[-2:]])))
__dolfinversion__ = dolfin_version()
del tmp, dolfin_swigversion, dolfin_version
%}
