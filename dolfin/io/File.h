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

#include <map>
#include <string>
#include <vector>
#include "dolfin/common/types.h"

#include <string>
#include <dolfin/main/MPI.h>
#include <dolfin/log/dolfin_log.h>
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

  class Mesh;
  class LocalMeshData;
  class Graph;
  template <class T> class MeshFunction;
  class Function;
  class Sample;
  class FiniteElementSpec;
  class ParameterList;
  class GenericFile;
  class FiniteElement;
  class GenericMatrix;
  class GenericVector;
  class FunctionPlotData;

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

    /*
    /// Write to file
    template<class T> void operator>> (const T& t)
    {
      file->write();
      *file << t;
    }
    */

    //--- Output ---

    /// Write vector to file
    void operator<< (const GenericVector& x);

    /// Write matrix to file
    void operator<< (const GenericMatrix& A);

    /// Write mesh to file
    void operator<< (const Mesh& mesh);

    /// Write local mesh data to file
    void operator<< (const LocalMeshData& data);

    /// Write mesh function to file
    void operator<< (const MeshFunction<int>& meshfunction);
    void operator<< (const MeshFunction<uint>& meshfunction);
    void operator<< (const MeshFunction<double>& meshfunction);
    void operator<< (const MeshFunction<bool>& meshfunction);

    /// Write function to file
    void operator<< (const Function& v);

    /// Write ODE sample to file
    void operator<< (const Sample& sample);

    /// Write finite element specification to file
    void operator<< (const FiniteElementSpec& spec);

    /// Write parameter list to file
    void operator<< (const ParameterList& parameters);

    /// Write graph to file
    void operator<< (const Graph& graph);

    /// Write FunctionPlotData to file
    void operator<< (const FunctionPlotData& data);

    /// Write array to file
    void operator<< (const std::vector<int>& x);
    void operator<< (const std::vector<uint>& x);
    void operator<< (const std::vector<double>& x);

    /// Write maps to file
    void operator<< (const std::map<uint, int>& map);
    void operator<< (const std::map<uint, uint>& map);
    void operator<< (const std::map<uint, double>& map);

    /// Write array maps to file
    void operator<< (const std::map<uint, std::vector<int> >& array_map);
    void operator<< (const std::map<uint, std::vector<uint> >& array_map);
    void operator<< (const std::map<uint, std::vector<double> >& array_map);


  private:

    // Pointer to implementation (envelop-letter design)
    GenericFile* file;

  };

}

#endif
