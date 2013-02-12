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
// First added:  2013-02-12
// Last changed:

#ifndef __LOCAL_SOLVER_H
#define __LOCAL_SOLVER_H

namespace dolfin
{

  // Forward declarations
  class GenericVector;
  class Form;

  class LocalSolver
  {
  public:

    void solve(GenericVector& u, const Form& a, const Form& L,
               bool symmetric=false) const;

  };

}

#endif
