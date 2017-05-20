// Copyright (C) 2005-2007 Garth N. Wells
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
// Modified by Nuno Lopes 2008.
//
// First added:  2008-05-29


#ifndef __RAW_FILE_H
#define __RAW_FILE_H

#include <fstream>
#include "GenericFile.h"

namespace dolfin
{

  /// Output of data in raw binary format

  class RAWFile : public GenericFile
  {
  public:

    /// Constructor
    explicit RAWFile(const std::string filename);

    /// Destructor
    ~RAWFile();

    /// Output MeshFunction (int)
    /// @param meshfunction
    ///  MeshFunction
    void operator<< (const MeshFunction<int>& meshfunction);

    /// Output MeshFunction (double)
    /// @param meshfunction
    ///  MeshFunction
    void operator<< (const MeshFunction<double>& meshfunction);

    /// Output Function
    /// @param u
    ///  Function
    void operator<< (const Function& u);

  private:

    void ResultsWrite(const Function& u) const;
    void rawNameUpdate(const int counter);

    template<typename T>
    void MeshFunctionWrite(T& meshfunction);

    // raw filename
    std::string raw_filename;

  };

}

#endif
