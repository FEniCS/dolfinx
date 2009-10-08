// Copyright (C) 2005-2007 Garth N.Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Nuno Lopes 2008.
// Modified by Martin Alnes 2008.
//
// First added:  2008-05-29

#include <sstream>
#include <fstream>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/la/Vector.h>
#include "RAWFile.h"

using namespace dolfin;

//----------------------------------------------------------------------------
RAWFile::RAWFile(const std::string filename) : GenericFile(filename)
{
  type = "RAW";
}
//----------------------------------------------------------------------------
RAWFile::~RAWFile()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void RAWFile::operator<<(const MeshFunction<int>& meshfunction)
{
  MeshFunctionWrite(meshfunction);
}
//----------------------------------------------------------------------------
void RAWFile::operator<<(const MeshFunction<unsigned int>& meshfunction)
{
  MeshFunctionWrite(meshfunction);
}
//----------------------------------------------------------------------------
void RAWFile::operator<<(const MeshFunction<double>& meshfunction)
{
  MeshFunctionWrite(meshfunction);
}
//----------------------------------------------------------------------------
void RAWFile::operator<<(const Function& u)
{
  // Update raw file name and clear file
  rawNameUpdate(counter);

  // Write results
  ResultsWrite(u);

  // Increase the number of times we have saved the function
  counter++;

  cout << "Saved function " << u.name() << " (" << u.label()
       << ") to file " << filename << " in RAW format." << endl;

}
//----------------------------------------------------------------------------
void RAWFile::ResultsWrite(const Function& u) const
{
  // Type of data (point or cell). Point by default.
  std::string data_type = "point";

  // For brevity
  const FunctionSpace& V = u.function_space();
  const Mesh& mesh(V.mesh());
  const DofMap& dofmap(V.dofmap());

  // Get rank of Function
  const uint rank = u.function_space().element().value_rank();
  if(rank > 1)
    error("Only scalar and vectors functions can be saved in Raw format.");

  // Get number of components
  uint dim = 1;
  for (uint i = 0; i < rank; i++)
    dim *= u.function_space().element().value_dimension(i);

  // Test for cell-based element type
  uint cell_based_dim = 1;
  for (uint i = 0; i < rank; i++)
    cell_based_dim *= mesh.topology().dim();
  if (dofmap.max_local_dimension() == cell_based_dim)
    data_type = "cell";

  // Open file
  std::ofstream fp(raw_filename.c_str(), std::ios_base::app);


  // Write function data at mesh cells
  if (data_type == "cell")
    {
      // Allocate memory for function values at cell centres
      const uint size = mesh.num_cells()*dim;
      double* values = new double[size];

      // Get function values on cells
      u.vector().get_local(values);

      // Write function data at cells
      uint num_cells = mesh.num_cells();
      fp << num_cells << std::endl;

      std::ostringstream ss;
      ss << std::scientific;
      for (CellIterator cell(mesh); !cell.end(); ++cell)
        {
          // Write all components
          ss.str("");
          for (uint i = 0; i < dim; i++)
            ss  <<" "<< values[cell->index() + i*mesh.num_cells()];
          ss << std::endl;
          fp << ss.str();
        }

      delete [] values;
    }
   else if (data_type == "point")
    {
      // Allocate memory for function values at vertices
      const uint size = mesh.num_vertices()*dim;
      double* values = new double[size];

      // Get function values at vertices
      u.compute_vertex_values(values, mesh);

      // Write function data at mesh vertices
      uint num_vertices = mesh.num_vertices();
      fp << num_vertices << std::endl;

      std::ostringstream ss;
      ss << std::scientific;
      for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
        {
          ss.str("");
          for(uint i=0; i<dim; i++)
            ss << " " << values[vertex->index() + i*mesh.num_cells()];

          ss << std::endl;
          fp << ss.str();
        }

      delete [] values;
    }
   else
     error("Unknown RAW data type.");
}
//----------------------------------------------------------------------------
void RAWFile::rawNameUpdate(const int counter)
{
  std::string filestart, extension;
  std::ostringstream fileid, newfilename;

  fileid.fill('0');
  fileid.width(6);

  filestart.assign(filename, 0, filename.find("."));
  extension.assign(filename, filename.find("."), filename.size());

  fileid << counter;
  newfilename << filestart << fileid.str() << ".raw";

  raw_filename = newfilename.str();

  // Make sure file is empty
  FILE* fp = fopen(raw_filename.c_str(), "w");
  if (!fp)
    error("Unable to open file %s", filename.c_str());

  fclose(fp);
}
//----------------------------------------------------------------------------
template<class T>
void RAWFile::MeshFunctionWrite(T& meshfunction)
{
  // Update raw file name and clear file
  rawNameUpdate(counter);

  const Mesh& mesh = meshfunction.mesh();

  if( meshfunction.dim() != mesh.topology().dim() )
    error("RAW output of mesh functions is implemenetd for cell-based functions only.");

  // Open file
  std::ofstream fp(raw_filename.c_str(), std::ios_base::app);

  fp<<mesh.num_cells( ) <<std::endl;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    fp << meshfunction[cell->index()] << std::endl;

  // Close file
  fp.close();

  // Increase the number of times we have saved the mesh function
  counter++;

  cout << "saved mesh function " << counter << " times." << endl;

  cout << "Saved mesh function " << mesh.name() << " (" << mesh.label()
       << ") to file " << filename << " in RAW format." << endl;
}

