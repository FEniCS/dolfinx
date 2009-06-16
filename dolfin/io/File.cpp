// Copyright (C) 2002-2009 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005-2009.
// Modified by Haiko Etzel 2005.
// Modified by Magnus Vikstrom 2007.
// Modified by Nuno Lopes 2008.
// Modified by Niclas Jansson 2008.
// Modified by Ola Skavhaug 2009.
//
// First added:  2002-11-12
// Last changed: 2009-06-17

#include <string>
#include <dolfin/main/MPI.h>
#include <dolfin/log/dolfin_log.h>
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

using namespace dolfin;

//-----------------------------------------------------------------------------
File::File(const std::string filename)
{
  // Choose file type base on suffix.

  // FIXME: Use correct funtion to find the suffix; using rfind() makes
  //        it essential that the suffixes are checked in the correct order.

  if (filename.rfind(".xml.gz") != filename.npos)
    file = new XMLFile(filename, true);
  else if (filename.rfind(".xml") != filename.npos)
    file = new XMLFile(filename, false);
  else if (filename.rfind(".m") != filename.npos)
    file = new OctaveFile(filename);
  else if (filename.rfind(".py") != filename.npos)
    file = new PythonFile(filename);
  else if (filename.rfind(".pvd") != filename.npos)
  {
    if (MPI::num_processes() > 1)
      file = new PVTKFile(filename);
    else
      file = new VTKFile(filename);
  }
  else if (filename.rfind(".raw") != filename.npos)
    file = new RAWFile(filename);
  else if (filename.rfind(".xyz") != filename.npos)
    file = new XYZFile(filename);
  else
  {
    file = 0;
    error("Unknown file type for \"%s\".", filename.c_str());
  }
}
//-----------------------------------------------------------------------------
File::File(const std::string filename, Type type)
{
  switch (type) 
  {
  case xml:
    file = new XMLFile(filename, false);
    break;
  case matlab:
    file = new MatlabFile(filename);
    break;
  case octave:
    file = new OctaveFile(filename);
    break;
  case vtk:
    file = new VTKFile(filename);
    break;
  case python:
    file = new PythonFile(filename);
    break;
  default:
    file = 0;
    error("Unknown file type for \"%s\".", filename.c_str());
  }
}
//-----------------------------------------------------------------------------
File::File(std::ostream& outstream)
{
  file = new XMLFile(outstream);
}
//-----------------------------------------------------------------------------
File::~File()
{
  delete file;
  file = 0;
}
//-----------------------------------------------------------------------------
// Instantiate templated functions explicitly for PyDolfin
template void File::operator>> <GenericVector> (GenericVector& t);
template void File::operator>> <GenericMatrix> (GenericMatrix& t);
template void File::operator>> <Mesh> (Mesh& t);
template void File::operator>> <LocalMeshData> (LocalMeshData& t);
template void File::operator>> <MeshFunction<int> > (MeshFunction<int>& t);
template void File::operator>> <MeshFunction<dolfin::uint> > (MeshFunction<dolfin::uint>& t);
template void File::operator>> <MeshFunction<double> > (MeshFunction<double>& t);
template void File::operator>> <MeshFunction<bool> > (MeshFunction<bool>& t);
template void File::operator>> <Function> (Function& t);
template void File::operator>> <Sample> (Sample& t);
template void File::operator>> <FiniteElementSpec> (FiniteElementSpec& t);
template void File::operator>> <ParameterList> (ParameterList& t);
template void File::operator>> <Graph> (Graph& t);
template void File::operator>> <FunctionPlotData> (FunctionPlotData& t);
template void File::operator>> <std::vector<int> > (std::vector<int>& t);
template void File::operator>> <std::vector<dolfin::uint> > (std::vector<dolfin::uint>& t);
template void File::operator>> <std::vector<double> > (std::vector<double>& t);
template void File::operator>> <std::map<dolfin::uint, int> > (std::map<dolfin::uint, int>& t);
template void File::operator>> <std::map<dolfin::uint, dolfin::uint> > (std::map<dolfin::uint, dolfin::uint>& t);
template void File::operator>> <std::map<dolfin::uint, double> > (std::map<dolfin::uint, double>& t);
template void File::operator>> <std::map<dolfin::uint, std::vector<int> > > (std::map<dolfin::uint, std::vector<int> >& t);
template void File::operator>> <std::map<dolfin::uint, std::vector<dolfin::uint> > > (std::map<dolfin::uint, std::vector<dolfin::uint> >& t);
template void File::operator>> <std::map<dolfin::uint, std::vector<double> > > (std::map<dolfin::uint, std::vector<double> >& t);

template void File::operator<< <GenericVector> (const GenericVector& t);
template void File::operator<< <GenericMatrix> (const GenericMatrix& t);
template void File::operator<< <Mesh> (const Mesh& t);
template void File::operator<< <LocalMeshData> (const LocalMeshData& t);
template void File::operator<< <MeshFunction<int> > (const MeshFunction<int>& t);
template void File::operator<< <MeshFunction<dolfin::uint> > (const MeshFunction<dolfin::uint>& t);
template void File::operator<< <MeshFunction<double> > (const MeshFunction<double>& t);
template void File::operator<< <MeshFunction<bool> > (const MeshFunction<bool>& t);
template void File::operator<< <Function> (const Function& t);
template void File::operator<< <Sample> (const Sample& t);
template void File::operator<< <FiniteElementSpec> (const FiniteElementSpec& t);
template void File::operator<< <ParameterList> (const ParameterList& t);
template void File::operator<< <Graph> (const Graph& t);
template void File::operator<< <FunctionPlotData> (const FunctionPlotData& t);
template void File::operator<< <std::vector<int> > (const std::vector<int>& t);
template void File::operator<< <std::vector<dolfin::uint> > (const std::vector<dolfin::uint>& t);
template void File::operator<< <std::vector<double> > (const std::vector<double>& t);
template void File::operator<< <std::map<dolfin::uint, int> > (const std::map<dolfin::uint, int>& t);
template void File::operator<< <std::map<dolfin::uint, dolfin::uint> > (const std::map<dolfin::uint, dolfin::uint>& t);
template void File::operator<< <std::map<dolfin::uint, double> > (const std::map<dolfin::uint, double>& t);
template void File::operator<< <std::map<dolfin::uint, std::vector<int> > > (const std::map<dolfin::uint, std::vector<int> >& t);
template void File::operator<< <std::map<dolfin::uint, std::vector<dolfin::uint> > > (const std::map<dolfin::uint, std::vector<dolfin::uint> >& t);
template void File::operator<< <std::map<dolfin::uint, std::vector<double> > > (const std::map<dolfin::uint, std::vector<double> >& t);


