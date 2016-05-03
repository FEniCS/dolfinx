// Copyright (C) 2015 Tormod Landet
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
// First added:  2015-09-22
//
// This file adds an easy to use wrapper for the LocalAssembler::assemble
// routine that can used from Python

#ifndef __ASSEMBLE_LOCAL_H
#define __ASSEMBLE_LOCAL_H

namespace dolfin
{
  /// Assemble form to local tensor on a cell
  void assemble_local(const Form& a,
                      const Cell& cell,
                      std::vector<double>& tensor);
}

#endif
