// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg
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
// First added:  2003-07-15
// Last changed: 2005

#ifndef __PYTHON_FILE_H
#define __PYTHON_FILE_H

#include <string>
#include <tr1/tuple>
#include <dolfin/common/types.h>
#include "GenericFile.h"
#include <boost/ref.hpp>

namespace dolfin
{
  typedef boost::reference_wrapper<Array<double> > RealArrayRef;

  class Sample;

  // Represents input/output of data in a format readable by Python
  // (Numeric). The data is written to several files (the variable
  // name is appended to the base file name) to enable incremental
  // output in an efficient way.

  class PythonFile : public GenericFile
  {
  public:

    PythonFile(const std::string filename);
    virtual ~PythonFile();

    // Input

    // Output

    void operator<< (const std::pair<double, RealArrayRef>);

    std::string filename_t, filename_u, filename_k, filename_r;

  };

}

#endif
