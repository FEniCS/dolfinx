// Copyright (C) 2002-2008 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2009.
// Modified by Magnus Vikstrom 2007
// Modified by Nuno Lopes 2008
// Modified by Ola Skavhaug 2009
//
// First added:  2002-11-12
// Last changed: 2009-06-15

#ifndef __FILE_H
#define __FILE_H

#include <ostream>
#include <string>
#include "GenericFile.h"

namespace dolfin
{

  /// A File represents a data file for reading and writing objects.
  /// Unless specified explicitly, the format is determined by the
  /// file name suffix.

  class File
  {
  public:

    /// File formats
    enum Type {xml, matlab, octave, vtk, python, raw, xyz};

    /// Create a file with given name
    File(const std::string filename);

    /// Create a file with given name and type (format)
    File(const std::string filename, Type type);

    /// Create a outfile object writing to stream
    File(std::ostream& outstream);

    /// Destructor
    ~File();

    /// Read from file
    template<class T> void operator>> (T& t)
    {
      file->read();
      *file >> t;
    }

    /// Write to file
    template<class T> void operator<< (const T& t)
    {
      file->write();
      *file << t;
    }

  private:

    // Pointer to implementation (envelop-letter design)
    GenericFile* file;

  };

}

#endif
