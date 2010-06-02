// Copyright (C) 2003-2008 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug 2009.
//
// First added:  2003-07-15
// Last changed: 2009-03-16

#ifndef __GENERIC_FILE_H
#define __GENERIC_FILE_H

#include <map>
#include <string>
#include <utility>
#include <vector>
#include "dolfin/common/types.h"

namespace dolfin
{

  class Mesh;
  class LocalMeshData;
  template <class T> class MeshFunction;
  class Function;
  class Sample;
  class Parameters;
  class GenericMatrix;
  class GenericVector;
  class FunctionPlotData;

  class GenericFile
  {
  public:

    GenericFile(const std::string filename);
    virtual ~GenericFile();

    // Input
    virtual void operator>> (GenericVector& x);
    virtual void operator>> (GenericMatrix& A);
    virtual void operator>> (Mesh& mesh);
    virtual void operator>> (LocalMeshData& data);
    virtual void operator>> (MeshFunction<int>& meshfunction);
    virtual void operator>> (MeshFunction<unsigned int>& meshfunction);
    virtual void operator>> (MeshFunction<double>& meshfunction);
    virtual void operator>> (MeshFunction<bool>& meshfunction);
    virtual void operator>> (Sample& sample);
    virtual void operator>> (Parameters& parameters);
    virtual void operator>> (FunctionPlotData& data);
    virtual void operator>> (std::vector<int>& x);
    virtual void operator>> (std::vector<uint>& x);
    virtual void operator>> (std::vector<double>& x);
    virtual void operator>> (std::map<uint, int>& map);
    virtual void operator>> (std::map<uint, uint>& map);
    virtual void operator>> (std::map<uint, double>& map);
    virtual void operator>> (std::map<uint, std::vector<int> >& array_map);
    virtual void operator>> (std::map<uint, std::vector<uint> >& array_map);
    virtual void operator>> (std::map<uint, std::vector<double> >& array_map);

    // Output
    virtual void operator<< (const GenericVector& x);
    virtual void operator<< (const GenericMatrix& A);
    virtual void operator<< (const Mesh& mesh);
    virtual void operator<< (const LocalMeshData& data);
    virtual void operator<< (const MeshFunction<int>& meshfunction);
    virtual void operator<< (const MeshFunction<unsigned int>& meshfunction);
    virtual void operator<< (const MeshFunction<double>& meshfunction);
    virtual void operator<< (const MeshFunction<bool>& meshfunction);
    virtual void operator<< (const Function& u);

    // Output function with time
    virtual void operator<< (const std::pair<const Function*, double> u);

    virtual void operator<< (const Sample& sample);
    virtual void operator<< (const Parameters& parameters);
    virtual void operator<< (const FunctionPlotData& data);
    virtual void operator<< (const std::vector<int>& x);
    virtual void operator<< (const std::vector<uint>& x);
    virtual void operator<< (const std::vector<double>& x);
    virtual void operator<< (const std::map<uint, int>& map);
    virtual void operator<< (const std::map<uint, uint>& map);
    virtual void operator<< (const std::map<uint, double>& map);
    virtual void operator<< (const std::map<uint, std::vector<int> >& array_map);
    virtual void operator<< (const std::map<uint, std::vector<uint> >& array_map);
    virtual void operator<< (const std::map<uint, std::vector<double> >& array_map);

    void read();
    virtual void write();

  protected:

    void read_not_impl(const std::string object);
    void write_not_impl(const std::string object);

    std::string filename;
    std::string type;

    bool opened_read;
    bool opened_write;

    // True if we have written a header
    bool check_header;

    // Counters for the number of times various data has been written
    uint counter;
    uint counter1;
    uint counter2;

  };

}

#endif

