// Copyright (C) 2002-2008 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005, 2006.
// Modified by Magnus Vikstrom 2007
//
// First added:  2002-11-12
// Last changed: 2008-03-29

#ifndef __FILE_H
#define __FILE_H

#include <string>

#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>

namespace dolfin
{

  class Mesh;
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
    enum Type { xml, matlab, octave, opendx, vtk, python };
    
    /// Create a file with given name
    File(const std::string& filename);

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

    /// Read mesh function from file
    void operator>> (MeshFunction<int>& meshfunction);
    void operator>> (MeshFunction<unsigned int>& meshfunction);
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

    //--- Output ---

    /// Write vector to file
    void operator<< (GenericVector& x);

    /// Write matrix to file
    void operator<< (GenericMatrix& A);

    /// Write mesh to file
    void operator<< (Mesh& mesh);

    /// Write mesh function to file
    void operator<< (MeshFunction<int>& meshfunction);
    void operator<< (MeshFunction<unsigned int>& meshfunction);
    void operator<< (MeshFunction<double>& meshfunction);
    void operator<< (MeshFunction<bool>& meshfunction);

    /// Write function to file
    void operator<< (Function& u);

    /// Write ODE sample to file
    void operator<< (Sample& sample);

    /// Write finite element specification to file
    void operator<< (FiniteElementSpec& spec);

    /// Write parameter list to file
    void operator<< (ParameterList& parameters);

    /// Write FFC BLAS data to file
    void operator<< (BLASFormData& blas);
	 
    /// Write graph to file
    void operator<< (Graph& graph);
    
  private:
    
    GenericFile* file;
    
  };
  
}

#endif
