// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/timeinfo.h>
#include <dolfin/System.h>
#include <dolfin/Mesh.h>
#include <dolfin/Function.h>
#include <dolfin/NewFunction.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/OpenDXFile.h>

using namespace dolfin;

//-­---------------------------------------------------------------------------
OpenDXFile::OpenDXFile(const std::string filename) : 
  GenericFile(filename),
  save_each_mesh(dolfin_get("save each mesh")),
  event_saving_mesh("Saving mesh to OpenDX file."),
  event_saving_function("Saving function to OpenDX file.")
{
  type = "OpenDX";
  series_pos = 0;
}
//-­---------------------------------------------------------------------------
OpenDXFile::~OpenDXFile()
{
  // Do nothing
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator<<(Mesh& mesh)
{
  event_saving_mesh();
  
  // Open file
  FILE* fp = fopen(filename.c_str(), "r+");
  fseek(fp, 0L, SEEK_END);
  
  // Write header first time
  if ( ftell(fp) == 0 )
    writeHeader(fp);

  // Write mesh
  writeMesh(fp, mesh);

  // Write mesh data
  writeMeshData(fp, mesh);
  
  // Close file
  fclose(fp);
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator<<(Function& u)
{
  event_saving_function();

  // Open file
  FILE* fp = fopen(filename.c_str(), "r+");
  fseek(fp, 0L, SEEK_END);

  // Remove previous time series 
  if ( frames.size() > 0 )
    removeSeries(fp);

  // Write mesh
  if ( frames.size() == 0 || save_each_mesh )
    writeMesh(fp, u.mesh());

  // Write function
  writeFunction(fp, u);

  // Write time series
  writeSeries(fp, u);

  // Close file
  fclose(fp);
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator<<(NewFunction& u)
{
  event_saving_function();

  // Open file
  FILE* fp = fopen(filename.c_str(), "r+");
  fseek(fp, 0L, SEEK_END);

  // Remove previous time series 
  if ( frames.size() > 0 )
    removeSeries(fp);

  // Write mesh
  if ( frames.size() == 0 || save_each_mesh )
    writeMesh(fp, u.mesh());

  // Write function
  writeFunction(fp, u);

  // Write time series
  writeSeries(fp, u);

  // Close file
  fclose(fp);
}
//-­---------------------------------------------------------------------------
void OpenDXFile::writeHeader(FILE* fp)
{
  System system;

  fprintf(fp,"# Output from DOLFIN version %s.\n", DOLFIN_VERSION);
  fprintf(fp,"# Format intended for use with OpenDX (Data Explorer).\n");
  fprintf(fp,"#\n");
  fprintf(fp,"# Saved by %s at %s\n", system.user(), date());
  fprintf(fp,"# on %s (%s) running %s version %s.\n",
	  system.host(), system.mach(), system.name(), system.vers());
  fprintf(fp,"\n");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::writeMesh(FILE* fp, Mesh& mesh)
{
  // Check that we have a tetrahedral mesh
  if ( mesh.type() != Mesh::tetrahedrons )
    dolfin_error("Mesh must be a 3D tetrahedral mesh for OpenDX file format.");

  // Write nodes
  fprintf(fp, "# A list of all node positions\n");
  fprintf(fp, "object \"nodes %d\" class array type float rank 1 shape 3 items %d lsb binary data follows\n", 
	  frames.size(), mesh.noNodes());

  for (NodeIterator n(mesh); !n.end(); ++n)
  {
    Point p = n->coord();
    
    float x = (float) p.x;
    float y = (float) p.y;
    float z = (float) p.z;
    
    fwrite(&x, sizeof(float), 1, fp);
    fwrite(&y, sizeof(float), 1, fp);
    fwrite(&z, sizeof(float), 1, fp);
  }
  fprintf(fp,"\n\n");

  // Write cells
  fprintf(fp, "# A list of all cells (connections)\n");
  fprintf(fp, "object \"cells %d\" class array type int rank 1 shape 4 items %d lsb binary data follows\n",
	  frames.size(), mesh.noCells());

  for (CellIterator c(mesh); !c.end(); ++c)
  {  
    for (NodeIterator n(c); !n.end(); ++n)
    {
      int id  = n->id();
      fwrite(&id, sizeof(int), 1, fp);
    }
  }
  fprintf(fp, "\n");
  fprintf(fp, "attribute \"element type\" string \"tetrahedra\"\n");
  fprintf(fp, "attribute \"ref\" string \"positions\"\n");
  fprintf(fp, "\n");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::writeMeshData(FILE* fp, Mesh& mesh)
{
  // Write data for a given mesh to create an object that can be visualized
  // when no data is associated with the mesh. This is necessary when we only
  // want to save the mesh.

  // Check that we don't try to write mesh data at the same time as we
  // write a time series
  if ( frames.size() > 0 )
    dolfin_error("Mesh data and time series cannot be mixed for OpenDX file format.");

  // Write data (cell diameter)
  fprintf(fp,"# Cell diameter\n");
  fprintf(fp,"object \"diameter\" class array type float rank 0 items %d lsb binary data follows\n",
	  mesh.noCells());
  
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    float value = static_cast<float>(c->diameter());
    fwrite(&value, sizeof(float), 1, fp);
  }
  fprintf(fp, "\n");
  fprintf(fp, "attribute \"dep\" string \"connections\"\n");
  fprintf(fp, "\n");  

  // Write the mesh
  fprintf(fp, "# The mesh\n");
  fprintf(fp, "object \"Mesh\" class field\n");
  fprintf(fp, "component \"positions\" value \"nodes 0\"\n");
  fprintf(fp, "component \"connections\" value \"cells 0\"\n");
  fprintf(fp, "component \"data\" value \"diameter\"\n");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::writeFunction(FILE* fp, Function& u)
{
  // Write header for object
  fprintf(fp,"# Values for [%s] at nodal points, frame %d\n", u.label().c_str(), frames.size());
  fprintf(fp,"object \"data %d\" class array type float rank 1 shape 1 items %d lsb binary data follows\n",
	  frames.size(), u.mesh().noNodes());
  
  // Write data
  for (NodeIterator n(u.mesh()); !n.end(); ++n)
  {
    float value = static_cast<float>(u(*n));
    fwrite(&value, sizeof(float), 1, fp);
  }
  fprintf(fp,"\n");
  fprintf(fp,"attribute \"dep\" string \"positions\"\n\n");
  
  // Write field
  fprintf(fp,"# Field for [%s], frame %d\n", u.label().c_str(), frames.size());
  fprintf(fp,"object \"field %d\" class field\n", frames.size());
  if ( save_each_mesh )
    fprintf(fp,"component \"positions\" value \"nodes %d\"\n", frames.size());
  else
    fprintf(fp,"component \"positions\" value \"nodes 0\"\n");
  if ( save_each_mesh )
    fprintf(fp,"component \"connections\" value \"cells %d\"\n", frames.size());
  else
    fprintf(fp,"component \"connections\" value \"cells 0\"\n");
  fprintf(fp,"component \"data\" value \"data %d\"\n\n", frames.size());
  
  // Add the new frame
  Frame frame(u.time());
  frames.push_back(frame);
}
//-­---------------------------------------------------------------------------
void OpenDXFile::writeFunction(FILE* fp, NewFunction& u)
{
  // Write header for object
  fprintf(fp,"# Values for [%s] at nodal points, frame %d\n", u.label().c_str(), frames.size());
  fprintf(fp,"object \"data %d\" class array type float rank 1 shape 1 items %d lsb binary data follows\n",
	  frames.size(), u.mesh().noNodes());
  
  // Write data
  for (NodeIterator n(u.mesh()); !n.end(); ++n)
  {
    float value = static_cast<float>(u(*n));
    fwrite(&value, sizeof(float), 1, fp);
  }
  fprintf(fp,"\n");
  fprintf(fp,"attribute \"dep\" string \"positions\"\n\n");
  
  // Write field
  fprintf(fp,"# Field for [%s], frame %d\n", u.label().c_str(), frames.size());
  fprintf(fp,"object \"field %d\" class field\n", frames.size());
  if ( save_each_mesh )
    fprintf(fp,"component \"positions\" value \"nodes %d\"\n", frames.size());
  else
    fprintf(fp,"component \"positions\" value \"nodes 0\"\n");
  if ( save_each_mesh )
    fprintf(fp,"component \"connections\" value \"cells %d\"\n", frames.size());
  else
    fprintf(fp,"component \"connections\" value \"cells 0\"\n");
  fprintf(fp,"component \"data\" value \"data %d\"\n\n", frames.size());
  
  // Add the new frame
  Frame frame(u.time());
  frames.push_back(frame);
}
//-­---------------------------------------------------------------------------
void OpenDXFile::writeSeries(FILE* fp, Function& u)
{
  // Get position in file at start of series
  series_pos = ftell(fp);

  // Write the time series
  fprintf(fp,"# Time series for [%s]\n", u.label().c_str());
  fprintf(fp,"object \"Time series\" class series\n");
  
  for (unsigned int i = 0; i < frames.size(); i++)
  {
    fprintf(fp,"member %d value \"field %d\" position %f\n",
	    i, i, frames[i].time);
  }
  
  fprintf(fp,"\n");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::writeSeries(FILE* fp, NewFunction& u)
{
  // Get position in file at start of series
  series_pos = ftell(fp);

  // Write the time series
  fprintf(fp,"# Time series for [%s]\n", u.label().c_str());
  fprintf(fp,"object \"Time series\" class series\n");
  
  for (unsigned int i = 0; i < frames.size(); i++)
  {
    fprintf(fp,"member %d value \"field %d\" position %f\n",
	    i, i, frames[i].time);
  }
  
  fprintf(fp,"\n");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::removeSeries(FILE* fp)
{
  // Remove the previous series (the new one will be placed at the end).
  // This makes sure that we have a valid dx-file even if we kill the
  // program after a few frames. Note that if someone puts a "#"
  // inside a comment we are in trouble.

  fseek(fp, series_pos, SEEK_SET);
  fflush(fp);
}
//-­---------------------------------------------------------------------------
// OpenDXFile::Frame
//-­---------------------------------------------------------------------------
OpenDXFile::Frame::Frame(real time) : time(time)
{
  // Do nothing
}
//-­---------------------------------------------------------------------------
OpenDXFile::Frame::~Frame()
{
  // Do nothing
}
//-­---------------------------------------------------------------------------
