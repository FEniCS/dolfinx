/* -*- C -*- */
// Copyright (C) 2012 Johan Hake
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
// First added:  2012-11-01
// Last changed: 2014-11-24

%ignore dolfin::GenericFile::operator>> (std::vector<int>& x);
%ignore dolfin::GenericFile::operator>> (std::vector<std::size_t>& x);
%ignore dolfin::GenericFile::operator>> (std::vector<double>& x);
%ignore dolfin::GenericFile::operator>> (std::map<std::size_t, int>& map);
%ignore dolfin::GenericFile::operator>> (std::map<std::size_t, std::size_t>& map);
%ignore dolfin::GenericFile::operator>> (std::map<std::size_t, double>& map);
%ignore dolfin::GenericFile::operator>> (std::map<std::size_t, std::vector<int> >& array_map);
%ignore dolfin::GenericFile::operator>> (std::map<std::size_t, std::vector<std::size_t> >& array_map);
%ignore dolfin::GenericFile::operator>> (std::map<std::size_t, std::vector<double> >& array_map);
%ignore dolfin::GenericFile::operator<< (const std::vector<int>& x);
%ignore dolfin::GenericFile::operator<< (const std::vector<std::size_t>& x);
%ignore dolfin::GenericFile::operator<< (const std::vector<double>& x);
%ignore dolfin::GenericFile::operator<< (const std::map<std::size_t, int>& map);
%ignore dolfin::GenericFile::operator<< (const std::map<std::size_t, std::size_t>& map);
%ignore dolfin::GenericFile::operator<< (const std::map<std::size_t, double>& map);
%ignore dolfin::GenericFile::operator<< (const std::map<std::size_t, std::vector<int> >& array_map);
%ignore dolfin::GenericFile::operator<< (const std::map<std::size_t, std::vector<std::size_t> >& array_map);
%ignore dolfin::GenericFile::operator<< (const std::map<std::size_t, std::vector<double> >& array_map);

%ignore dolfin::HDF5Attribute::get;
%ignore dolfin::HDF5Attribute::set;

%ignore dolfin::X3DOMParameters::get_color_map_array;
%rename (_set_color_map) dolfin::X3DOMParameters::set_color_map;
