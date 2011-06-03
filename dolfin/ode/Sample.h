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
// First added:  2003-11-20
// Last changed: 2005

#ifndef __SAMPLE_H
#define __SAMPLE_H

#include <string>
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  class TimeSlab;

  /// Sample of solution values at a given point.

  class Sample : public Variable
  {
  public:

    /// Constructor
    Sample(TimeSlab& timeslab, real t, std::string name, std::string label);

    /// Destructor
    ~Sample();

    /// Return number of components
    uint size() const;

    /// Return time t
    real t() const;

    /// Return value of component with given index
    real u(uint index) const;

    /// Return time step for component with given index
    real k(uint index) const;

    /// Return residual for component with given index
    real r(uint index) const;

  private:

    TimeSlab& timeslab;
    real time;

  };

}

#endif
