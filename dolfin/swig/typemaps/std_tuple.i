/* -*- C -*- */
// Copyright (C) 2015 Jan Blechta
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

//-----------------------------------------------------------------------------
// Out typemaps for std::tuple of primitives
//-----------------------------------------------------------------------------
%typemap(out) std::tuple<double, double, double>
{
  $result = Py_BuildValue("ddd",
                          std::get<0>($1), std::get<1>($1), std::get<2>($1));
}
%typemap(out) std::tuple<std::size_t, double, double, double>
{
  $result = Py_BuildValue("iddd",
                          std::get<0>($1), std::get<1>($1),
                          std::get<2>($1), std::get<3>($1));
}
