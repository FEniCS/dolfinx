// Copyright (C) 2002-2008 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005, 2006.
// Modified by Magnus Vikstrom 2007
// Modified by Nuno Lopes 2008
// Modified by Ola Skavhaug 2009
//
// First added:  2002-11-12
// Last changed: 2009-03-04

#ifndef __FILE_H
#define __FILE_H

#include <string>

#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>

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
  class BLASFormData;
  class GenericFile;
  class FiniteElement;
  
  /// A File represents a data file for reading and writing objects.
  /// Unless specified explicitly, the format is determined by the
  /// file name suffix.

  class File
  {
  public:
    
    /// File formats
    enum Type {xml, matlab, octave, opendx, vtk, python ,raw, xyz};
    
    /// Create a file with given name
    File(const std::string& filename, bool new_style=false);

    /// Create a file with given name and type (format)
    File(const std::string& filename, Type type);

    /// Destructor
    ~File();

    //--- Input ---
    
    /// Read vector from file
    void operator>> (GenericVector& x);

    /// Read matrix from file
    void operator>> (GenericMatrix& A);

    /// Read mesh from file
    void operator>> (Mesh& mesh);

    /// Read local mesh data from file
    void operator>> (LocalMeshData& data);

    /// Read mesh function from file
    void operator>> (MeshFunction<int>& meshfunction);
    void operator>> (MeshFunction<uint>& meshfunction);
    void operator>> (MeshFunction<double>& meshfunction);
    void operator>> (MeshFunction<bool>& meshfunction);

    /// Read function from file
    void operator>> (Function& u);

    /// Read ODE sample from file
    void operator>> (Sample& sample);
    
    /// Read finite element specification from file
    void operator>> (FiniteElementSpec& spec);

    /// Read parameter list from file
    void operator>> (ParameterList& parameters);

    /// Read FFC BLAS data from file
    void operator>> (BLASFormData& blas);
	 
    /// Read graph from file
    void operator>> (Graph& graph);

    /// Read array from file
    void operator>> (std::vector<int>& x);
    void operator>> (std::vector<uint>& x);
    void operator>> (std::vector<double>& x);

    /// Read maps from file
    void operator>> (std::map<uint, int>& map);
    void operator>> (std::map<uint, uint>& map);
    void operator>> (std::map<uint, double>& map);

    /// Read array maps from file
    void operator>> (std::map<uint, std::vector<int> >& array_map);
    void operator>> (std::map<uint, std::vector<uint> >& array_map);
    void operator>> (std::map<uint, std::vector<double> >& array_map);

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

    /// Write FFC BLAS data to file
    void operator<< (const BLASFormData& blas);
	 
    /// Write graph to file
    void operator<< (const Graph& graph);

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
