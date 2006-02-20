// Copyright (C) 2002-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2005
//
// First added:  2002-11-12
// Last changed: 2006-02-20

#ifndef __FILE_H
#define __FILE_H

#include <string>

namespace dolfin
{
  class Vector;
  class Matrix;
  class Mesh;
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
    enum Type { xml, matlab, matrixmarket, octave, opendx, gid, tecplot, vtk, python };
    
    /// Create a file with given name
    File(const std::string& filename);

    /// Create a file with given name and type (format)
    File(const std::string& filename, Type type);

    /// Destructor
    ~File();

    //--- Input ---
    
    /// Read vector from file
    void operator>> (Vector& x);

    /// Read matrix from file
    void operator>> (Matrix& A);

    /// Read mesh from file
    void operator>> (Mesh& mesh);

    /// Read ODE sample from file
    void operator>> (Sample& sample);
    
    /// Read finite element specification from file
    void operator>> (FiniteElementSpec& spec);

    /// Read parameter list from file
    void operator>> (ParameterList& parameters);

    /// Read FFC BLAS data from file
    void operator>> (BLASFormData& blas);

    /// Read function from file
    void parse(Function& u, FiniteElement& element);

    //--- Output ---

    /// Write vector to file
    void operator<< (Vector& x);

    /// Write matrix to file
    void operator<< (Matrix& A);

    /// Write mesh to file
    void operator<< (Mesh& mesh);

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
    
  private:
    
    GenericFile* file;
    
  };
  
}

#endif
