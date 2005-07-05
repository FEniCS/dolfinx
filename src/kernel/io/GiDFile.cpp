// Copyright (C) 2004 Harald Svensson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004.
//
// First added:  2004-03-30
// Last changed: 2004

#include <stdio.h>
#include <dolfin/Mesh.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/GiDFile.h>

using namespace dolfin;

//-­---------------------------------------------------------------------------
GiDFile::GiDFile(const std::string filename) : GenericFile(filename)
{
  type = "GiD";
}
//-­---------------------------------------------------------------------------
GiDFile::~GiDFile()
{
  // Do nothing
}
//-­---------------------------------------------------------------------------
void GiDFile::operator<<(Mesh& mesh)
{
  dolfin_info("Saving mesh to GiD file.");

  // Open file
  FILE* fp = fopen(filename.c_str(), "a");

  // Write nodes
  fprintf(fp, "# Mesh \n");
  fprintf(fp, "MESH dimension 3 ElemType Tetrahedra Nnode 4 \n");
  fprintf(fp, "Coordinates \n");
  fprintf(fp, "#  Node   Coord_X   Coord_Y   Coord_Z \n");
  for (NodeIterator n(mesh); !n.end(); ++n)
  {

    Point   p = n->coord();
    int     nid = n->id();

    float x1 = (float) p.x;
    float x2 = (float) p.y;
    float x3 = (float) p.z;

    fprintf(fp," %6d %.12f %.12f %.12f \n",nid,x1,x2,x3);

  }
  fprintf(fp, "end coordinates \n");
  fprintf(fp,"\n");

  // Write cells
  fprintf(fp, "Elements \n");
  fprintf(fp, "# Element   Node_1   Node_2   Node 3   Node 4  \n");
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    int cid,nid[4];
    if ( c->noNodes() == 4 )
    {
      cid = c->id();
      int i = 0;
      for (NodeIterator n(c); !n.end(); ++n)
      {
        nid[i]  = n->id();
        i = i + 1;
      }
      fprintf(fp," %6d %8d %8d %8d %8d \n",cid,nid[0],nid[1],nid[2],nid[3]);
    }
  }
  fprintf(fp, "End elements \n");
  fprintf(fp, "\n");

  fprintf(fp, "# Mesh \n");
  fprintf(fp, "MESH dimension 3 ElemType Triangle Nnode 3 \n");
  // Write cells
  fprintf(fp, "Elements \n");
  fprintf(fp, "# Element   Node_1   Node_2   Node 3 \n");

  for (CellIterator c(mesh); !c.end(); ++c)
  {
    int cid,nid[3];
    if ( c->noNodes() == 3 )
    {
      cid = c->id();
      int i = 0;
      for (NodeIterator n(c); !n.end(); ++n)
      {
        nid[i]  = n->id();
        i = i + 1;
      }
      fprintf(fp," %6d %8d %8d %8d \n",cid,nid[0],nid[1],nid[2]);
    }
  }
  fprintf(fp, "End elements \n");
  fprintf(fp, "\n");

  // Close file
  fclose(fp);
}
//-­---------------------------------------------------------------------------
/*

FIXME: BROKEN

Needs to be updated for new function class.

void GiDFile::operator<<(Function& u)
{
  dolfin_info("Saving scalar function to GiD file .");

  Mesh* mesh = &(u.mesh());

  // Open file
  FILE* fp = fopen(filename.c_str(), "a");

  fprintf(fp, "# Results \n");
  fprintf(fp, "Result DolfinAnalysis AnalysisName %f Vector OnNodes \n",u.time());
  fprintf(fp, "ComponentNames %s \n",u.name().c_str());
  // Write cells
  fprintf(fp, "Values \n");
  for (NodeIterator n(mesh); !n.end(); ++n)
  {
    int   nid = n->id();
    float value = (float)u(*n);
    fprintf(fp," %d %.16f \n",nid,value);
  }
  fprintf(fp, "End Values \n");

  fprintf(fp, "\n");

  // Close file
  fclose(fp);
}
//-­---------------------------------------------------------------------------
void GiDFile::operator<<(Function::Vector& u)
{
  dolfin_info("Saving vector function to GiD file .");

  Mesh* mesh = &(u(0).mesh());

  // FIXME: Why 3?
  unsigned int noValues = 3; // u.size()/noNodes;

  // Open file
  FILE* fp = fopen(filename.c_str(), "a");

  fprintf(fp, "# Results \n");
  fprintf(fp, "Result DolfinAnalysis AnalysisName %f Vector OnNodes \n",
	  u(0).time());
  fprintf(fp, "ComponentNames %s %s %s \n",
	  u(0).name().c_str(),u(1).name().c_str(),u(2).name().c_str());
  // Write cells
  fprintf(fp, "Values \n");
  for (NodeIterator n(mesh); !n.end(); ++n)
  {
    int   nid   = n->id();
    fprintf(fp,"%d",nid);
    for (unsigned int i = 0; i < noValues; i++)
    {
      float value = (float)u(i)(*n);
      fprintf(fp,"%.16f",value);
    }
  fprintf(fp, "\n");
  }
  fprintf(fp, "End Values \n");

  fprintf(fp, "\n");

  // Close file
  fclose(fp);
}
//-­---------------------------------------------------------------------------
*/
