// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-11-11
// Last changed: 2010-04-29

#ifndef __BINARY_FILE_H
#define __BINARY_FILE_H

#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
#include "GenericFile.h"
#include <tr1/tuple>

namespace dolfin
{

  // Forward declarations
  class GenericVector;
  class Mesh;

  /// This class handles input/output in binary format. This format
  /// is more efficient than DOLFIN XML format but does not support
  /// all data types. Use this format with caution. Often, a plain
  /// text self-documenting format is more suitable for storing data.

  class BinaryFile : public GenericFile
  {
  public:

    /// Constructor
    BinaryFile(const std::string filename);

    /// Destructor
    virtual ~BinaryFile();

    //--- Input ---

    /// Read array
    void operator>> (std::vector<double>& values);

    /// Read vector
    void operator>> (GenericVector& vector);

    /// Read mesh
    void operator>> (Mesh& mesh);

    //--- Output ---

    /// Write array
    void operator<< (const std::vector<double>& values);

    /// Write vector
    void operator<< (const GenericVector& vector);

    /// Write mesh
    void operator<< (const Mesh& mesh);

  private:

    // Open file for reading
    void open_read();

    // Open file for writing
    void open_write();

    // Close file for reading
    void close_read();

    // Close file for writing
    void close_write();

    // Read uint
    uint read_uint();

    // Read array (uint)
    void read_array(uint n, uint* values);

    // Read array (double)
    void read_array(uint n, double* values);

    // Write uint
    void write_uint(uint value);

    // Write array (uint)
    void write_array(uint n, const uint* values);

    // Write array (double)
    void write_array(uint n, const double* values);

    // File for reading
    std::ifstream ifile;

    // File for writing
    std::ofstream ofile;

  };

}

#endif
