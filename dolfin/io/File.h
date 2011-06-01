// Copyright (C) 2002-2008 Johan Hoffman and Anders Logg
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
// Modified by Garth N. Wells, 2005-2009.
// Modified by Magnus Vikstrom 2007
// Modified by Nuno Lopes 2008
// Modified by Ola Skavhaug 2009
//
// First added:  2002-11-12
// Last changed: 2010-02-10

#ifndef __FILE_H
#define __FILE_H

#include <ostream>
#include <string>
#include <utility>
#include <boost/scoped_ptr.hpp>
#include "GenericFile.h"

namespace dolfin
{

  /// A File represents a data file for reading and writing objects.
  /// Unless specified explicitly, the format is determined by the
  /// file name suffix.

  /// A list of objects that can be read/written to file can be found in
  /// GenericFile.h

  class File
  {
  public:

    /// File formats
    enum Type {xml, vtk, python, raw, xyz, binary};

    /// Create a file with given name
    File(const std::string filename, std::string encoding = "ascii");

    /// Create a file with given name and type (format)
    File(const std::string filename, Type type, std::string encoding = "ascii");

    /// Create a outfile object writing to stream
    File(std::ostream& outstream);

    /// Destructor
    ~File();

    /// Read from file
    template<class T> void operator>>(T& t)
    {
      file->read();
      *file >> t;
    }

    /// Write object to file
    template<class T> void operator<<(const T& t)
    {
      file->write();
      *file << t;
    }

    /// Check if file exists
    static bool exists(std::string filename);

  private:

    // Pointer to implementation (envelop-letter design)
    boost::scoped_ptr<GenericFile> file;

  };

}

#endif
