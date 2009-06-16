// Copyright (C) 2002-2008 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2009.
// Modified by Magnus Vikstrom 2007
// Modified by Nuno Lopes 2008
// Modified by Ola Skavhaug 2009
//
// First added:  2002-11-12
// Last changed: 2009-06-16

#ifndef __FILE_H
#define __FILE_H

#include <ostream>
#include <string>
#include "GenericFile.h"

#include <string>
#include <dolfin/main/MPI.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/MeshFunction.h>
#include "File.h"
#include "GenericFile.h"
#include "XMLFile.h"
#include "MatlabFile.h"
#include "OctaveFile.h"
#include "PythonFile.h"
#include "PVTKFile.h"
#include "VTKFile.h"
#include "RAWFile.h"
#include "XYZFile.h"

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

    /// Read (operator>> is not templated to avoid SWIG template instantiations)
    void operator>> (GenericVector& t) { read(t); }
    void operator>> (GenericMatrix& t){ read(t); }
    void operator>> (Mesh& t){ read(t); }
    void operator>> (LocalMeshData& t){ read(t); }
    void operator>> (MeshFunction<int>& t){ read(t); }
    void operator>> (MeshFunction<dolfin::uint>& t){ read(t); }
    void operator>> (MeshFunction<double>& t){ read(t); }
    void operator>> (MeshFunction<bool>& t){ read(t); }
    void operator>> (Function& t){ read(t); }
    void operator>> (Sample& t){ read(t); }
    void operator>> (FiniteElementSpec& t){ read(t); }
    void operator>> (ParameterList& t){ read(t); }
    void operator>> (Graph& t){ read(t); }
    void operator>> (FunctionPlotData& t){ read(t); }
    void operator>> (std::vector<int>& t){ read(t); }
    void operator>> (std::vector<dolfin::uint>& t){ read(t); }
    void operator>> (std::vector<double>& t){ read(t); }
    void operator>> (std::map<dolfin::uint, int>& t){ read(t); }
    void operator>> (std::map<dolfin::uint, dolfin::uint>& t){ read(t); }
    void operator>> (std::map<dolfin::uint, double>& t){ read(t); }
    void operator>> (std::map<dolfin::uint, std::vector<int> >& t){ read(t); }
    void operator>> (std::map<dolfin::uint, std::vector<dolfin::uint> >& t){ read(t); }
    void operator>> (std::map<dolfin::uint, std::vector<double> >& t){ read(t); }

    /// Write (operator<< is not templated to avoid SWIG template instantiations)
    void operator<< (const GenericVector& t) { write(t); }
    void operator<< (const GenericMatrix& t){ write(t); }
    void operator<< (const Mesh& t){ write(t); }
    void operator<< (const LocalMeshData& t){ write(t); }
    void operator<< (const MeshFunction<int>& t){ write(t); }
    void operator<< (const MeshFunction<dolfin::uint>& t){ write(t); }
    void operator<< (const MeshFunction<double>& t){ write(t); }
    void operator<< (const MeshFunction<bool>& t){ write(t); }
    void operator<< (const Function& t){ write(t); }
    void operator<< (const Sample& t){ write(t); }
    void operator<< (const FiniteElementSpec& t){ write(t); }
    void operator<< (const ParameterList& t){ write(t); }
    void operator<< (const Graph& t){ write(t); }
    void operator<< (const FunctionPlotData& t){ write(t); }
    void operator<< (const std::vector<int>& t){ write(t); }
    void operator<< (const std::vector<dolfin::uint>& t){ write(t); }
    void operator<< (const std::vector<double>& t){ write(t); }
    void operator<< (const std::map<dolfin::uint, int>& t){ write(t); }
    void operator<< (const std::map<dolfin::uint, dolfin::uint>& t){ write(t); }
    void operator<< (const std::map<dolfin::uint, double>& t){ write(t); }
    void operator<< (const std::map<dolfin::uint, std::vector<int> >& t){ write(t); }
    void operator<< (const std::map<dolfin::uint, std::vector<dolfin::uint> >& t){ write(t); }
    void operator<< (const std::map<dolfin::uint, std::vector<double> >& t){ write(t); }

  private:

    /// Read from file
    template<class T> void read(T& t)
    {
      file->read();
      *file >> t;
    }

    /// Write to file
    template<class T> void write (const T& t)
    {
      file->write();
      *file << t;
    }

    // Pointer to implementation (envelop-letter design)
    GenericFile* file;

  };

}

#endif
