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
// Last changed: 2012-11-01

%ignore dolfin::GenericFile::operator>> (std::vector<int>& x);
%ignore dolfin::GenericFile::operator>> (std::vector<uint>& x);
%ignore dolfin::GenericFile::operator>> (std::vector<double>& x);
%ignore dolfin::GenericFile::operator>> (std::map<uint, int>& map);
%ignore dolfin::GenericFile::operator>> (std::map<uint, uint>& map);
%ignore dolfin::GenericFile::operator>> (std::map<uint, double>& map);
%ignore dolfin::GenericFile::operator>> (std::map<uint, std::vector<int> >& array_map);
%ignore dolfin::GenericFile::operator>> (std::map<uint, std::vector<uint> >& array_map);
%ignore dolfin::GenericFile::operator>> (std::map<uint, std::vector<double> >& array_map);
%ignore dolfin::GenericFile::operator<< (const std::vector<int>& x);
%ignore dolfin::GenericFile::operator<< (const std::vector<uint>& x);
%ignore dolfin::GenericFile::operator<< (const std::vector<double>& x);
%ignore dolfin::GenericFile::operator<< (const std::map<uint, int>& map);
%ignore dolfin::GenericFile::operator<< (const std::map<uint, uint>& map);
%ignore dolfin::GenericFile::operator<< (const std::map<uint, double>& map);
%ignore dolfin::GenericFile::operator<< (const std::map<uint, std::vector<int> >& array_map);
%ignore dolfin::GenericFile::operator<< (const std::map<uint, std::vector<uint> >& array_map);
%ignore dolfin::GenericFile::operator<< (const std::map<uint, std::vector<double> >& array_map);
