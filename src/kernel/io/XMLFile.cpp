// Copyright (C) 2002-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Erik Svensson 2003.
//
// First added:  2002-12-03
// Last changed: 2005-12-01

#include <stdarg.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Mesh.h>
#include <dolfin/Parameter.h>
#include <dolfin/ParameterList.h>

#include <dolfin/XMLObject.h>
#include <dolfin/XMLVector.h>
#include <dolfin/XMLMatrix.h>
#include <dolfin/XMLMesh.h>
#include <dolfin/XMLParameterList.h>
#include <dolfin/XMLBLASFormData.h>
#include <dolfin/XMLFile.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLFile::XMLFile(const std::string filename) : GenericFile(filename)
{
  type = "XML";
  xmlObject = 0;
}
//-----------------------------------------------------------------------------
XMLFile::~XMLFile()
{
  if ( xmlObject )
    delete xmlObject;
}
//-----------------------------------------------------------------------------
void XMLFile::operator>>(Vector& x)
{
  if ( xmlObject )
    delete xmlObject;
  xmlObject = new XMLVector(x);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>>(Matrix& A)
{
  if ( xmlObject )
    delete xmlObject;
  xmlObject = new XMLMatrix(A);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>>(Mesh& mesh)
{
  if ( xmlObject )
    delete xmlObject;
  xmlObject = new XMLMesh(mesh);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>>(ParameterList& parameters)
{
  if ( xmlObject )
    delete xmlObject;
  xmlObject = new XMLParameterList(parameters);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>>(BLASFormData& blas)
{
  if ( xmlObject )
    delete xmlObject;
  xmlObject = new XMLBLASFormData(blas);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator<<(Vector& x)
{
  // FIXME: update to new format
  dolfin_error("This function needs to be updated to the new format.");

  /*
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  // Write vector in XML format
  fprintf(fp, "<?xml version=\"1.0\" encoding=\"UTF-8\"?> \n\n" );  
  fprintf(fp, "<dolfin xmlns:dolfin=\"http://www.phi.chalmers.se/dolfin/\"> \n" );
  fprintf(fp, "  <vector size=\" %i \"> \n", x.size() );
  
  for (unsigned int i = 0; i < x.size(); i++) {
    fprintf(fp, "    <element row=\"%i\" value=\"%f\"/>\n", i, x(i) );
    if ( i == (x.size() - 1))
      fprintf(fp, "  </vector>\n");
  }
  
  fprintf(fp, "</dolfin>\n");
  
  // Close file
  fclose(fp);
  
  // Increase the number of times we have saved the vector
  ++x;
  
  cout << "Saved vector " << x.name() << " (" << x.label()
       << ") to file " << filename << " in XML format." << endl;
  */
}
//-----------------------------------------------------------------------------
void XMLFile::operator<<(Matrix& A)
{
  // FIXME: update to new format
  dolfin_error("This function needs to be updated to the new format.");

  /*
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  // Write matrix in sparse XML format
  fprintf(fp, "<?xml version=\"1.0\" encoding=\"UTF-8\"?> \n\n" );
  fprintf(fp, "<dolfin xmlns:dolfin=\"http://www.phi.chalmers.se/dolfin/\"> \n" );
  
  switch( A.type() ) {
  case Matrix::sparse:

    fprintf(fp, "  <sparsematrix rows=\"%i\" columns=\"%i\">\n", A.size(0), A.size(1) );
                                                                                                                             
    for (unsigned int i = 0; i < A.size(0); i++) {
      fprintf(fp, "    <row row=\"%i\" size=\"%i\">\n", i, A.rowsize(i));
      for (unsigned int pos = 0; !A.endrow(i, pos); pos++) {
	unsigned int j = 0;
	real aij = A(i, j, pos);
        fprintf(fp, "      <element column=\"%i\" value=\"%f\"/>\n", j, aij);
        if ( i == (A.size(0) - 1) && pos == (A.rowsize(i)-1)){
           fprintf(fp, "    </row>\n");
           fprintf(fp, "  </sparsematrix>\n");
        }
        else if ( pos == (A.rowsize(i)-1))
          fprintf(fp, "    </row>\n");
      }
    }
    break;

  case Matrix::dense:
    dolfin_error("Cannot write dense matrices to XML files.");
    break;
  default:
    dolfin_error("Unknown matrix type.");
  }
  
  fprintf(fp, "</dolfin>\n");
                                                                                                                             
  // Close file
  fclose(fp);
                                                                                                                             
  // Increase the number of times we have saved the matrix
  ++A;
                                                                                                                             
  cout << "Saved matrix " << A.name() << " (" << A.label()
       << ") to file " << filename << " in XML format." << endl;
  */
}
//-----------------------------------------------------------------------------
void XMLFile::operator<<(Mesh& mesh)
{
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  // Write mesh in XML format
  fprintf(fp, "<?xml version=\"1.0\" encoding=\"UTF-8\"?> \n\n" );  
  fprintf(fp, "<dolfin xmlns:dolfin=\"http://www.phi.chalmers.se/dolfin/\"> \n" );
  fprintf(fp, "  <mesh> \n");

  fprintf(fp, "    <vertices size=\" %i \"> \n", mesh.noVertices());
  
  for(VertexIterator n(&mesh); !n.end(); ++n)
  {
    Vertex &vertex = *n;

    fprintf(fp, "    <vertex name=\"%i\" x=\"%f\" y=\"%f\" z=\"%f\" />\n",
	    vertex.id(), vertex.coord().x, vertex.coord().y, vertex.coord().z);
  }

  fprintf(fp, "    </vertices>\n");

  fprintf(fp, "    <cells size=\" %i \"> \n", mesh.noCells());

  for (CellIterator c(mesh); !c.end(); ++c)
  {
    Cell &cell = *c;

    if ( mesh.type() == Mesh::tetrahedra )
    {
      fprintf(fp, "    <tetrahedron name=\"%i\" n0=\"%i\" n1=\"%i\" n2=\"%i\" n3=\"%i\" />\n",
	      cell.id(), cell.vertex(0).id(), cell.vertex(1).id(), cell.vertex(2).id(), cell.vertex(3).id());
    }
    else
    {
      fprintf(fp, "    <triangle name=\"%i\" n0=\"%i\" n1=\"%i\" n2=\"%i\" />\n",
	      cell.id(), cell.vertex(0).id(), cell.vertex(1).id(),
	      cell.vertex(2).id());
    }
  }

  fprintf(fp, "    </cells>\n");
  
  fprintf(fp, "  </mesh>\n");
  fprintf(fp, "</dolfin>\n");
  
  // Close file
  fclose(fp);
  
  cout << "Saved mesh " << mesh.name() << " (" << mesh.label()
       << ") to file " << filename << " in XML format." << endl;
}
//-----------------------------------------------------------------------------
void XMLFile::operator<<(ParameterList& parameters)
{
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  // Write parameter list in XML format
  fprintf(fp, "<?xml version=\"1.0\" encoding=\"UTF-8\"?> \n\n" );
  fprintf(fp, "<dolfin xmlns:dolfin=\"http://www.fenics.org/dolfin/\"> \n" );
  fprintf(fp, "  <parameters>\n" );

  for (ParameterList::ParameterIterator it = parameters.parameters.begin(); it != parameters.parameters.end(); ++it)
  {
    const Parameter p = it->second;

    switch ( p.type )
    {
    case Parameter::REAL:
      fprintf(fp, "    <parameter name=\"%s\" type=\"real\" value=\"%.16e\"/>\n",
	      p.identifier.c_str(), p.val_real);
      break;
    case Parameter::INT:
      fprintf(fp, "    <parameter name=\"%s\" type=\"int\" value=\"%d\"/>\n",
	      p.identifier.c_str(), p.val_int);
      break;
    case Parameter::BOOL:
      if ( p.val_bool )
	fprintf(fp, "    <parameter name=\"%s\" type=\"bool\" value=\"true\"/>\n",
		p.identifier.c_str());
      else
	fprintf(fp, "    <parameter name=\"%s\" type=\"bool\" value=\"false\"/>\n",
		p.identifier.c_str());
      break;
    case Parameter::STRING:
      fprintf(fp, "    <parameter name=\"%s\" type=\"string\" value=\"%s\"/>\n",
	      p.identifier.c_str(), p.val_string.c_str());
      break;
    default:
      ; // Do nothing
    }

  }
  
  fprintf(fp, "  </parameters>\n" );
  fprintf(fp, "</dolfin>\n");

  // Close file
  fclose(fp);

  cout << "Saved parameters to file " << filename << " in XML format." << endl;
}
//-----------------------------------------------------------------------------
void XMLFile::parseFile()
{
  // Write a message
  xmlObject->reading(filename);

  // Parse file using the SAX interface
  parseSAX();
  
  // Check that we got the data
  if ( !xmlObject->dataOK() )
    dolfin_error("Unable to find data in XML file.");
  
  // Write a message
  xmlObject->done();
}
//-----------------------------------------------------------------------------
void XMLFile::parseSAX()
{
  // Set up the sax handler. Note that it is important that we initialise
  // all (24) fields, even the ones we don't use!
  xmlSAXHandler sax = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  
  // Set up handlers for parser events
  sax.startDocument = sax_start_document;
  sax.endDocument   = sax_end_document;
  sax.startElement  = sax_start_element;
  sax.endElement    = sax_end_element;
  sax.warning       = sax_warning;
  sax.error         = sax_error;
  sax.fatalError    = sax_fatal_error;
  
  // Parse file
  xmlSAXUserParseFile(&sax, (void *) xmlObject, filename.c_str());
}
//-----------------------------------------------------------------------------
// Callback functions for the SAX interface
//-----------------------------------------------------------------------------
void dolfin::sax_start_document(void *ctx)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void dolfin::sax_end_document(void *ctx)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void dolfin::sax_start_element(void *ctx,
			       const xmlChar *name, const xmlChar **attrs)
{
  ( (XMLObject *) ctx )->startElement(name, attrs);
}
//-----------------------------------------------------------------------------
void dolfin::sax_end_element(void *ctx, const xmlChar *name)
{
  ( (XMLObject *) ctx )->endElement(name);
}
//-----------------------------------------------------------------------------
void dolfin::sax_warning(void *ctx, const char *msg, ...)
{
  va_list args;
  
  va_start(args, msg);
  dolfin_info_aptr(msg, args);
  dolfin_warning("Incomplete XML data.");
  va_end(args);
}
//-----------------------------------------------------------------------------
void dolfin::sax_error(void *ctx, const char *msg, ...)
{
  va_list args;
  
  va_start(args, msg);
  dolfin_info_aptr(msg, args);
  dolfin_error("Illegal XML data.");
  va_end(args);
}
//-----------------------------------------------------------------------------
void dolfin::sax_fatal_error(void *ctx, const char *msg, ...)
{
  va_list args;
  
  va_start(args, msg);
  dolfin_info_aptr(msg, args);
  dolfin_error("Illegal XML data.");
  va_end(args);
}
//-----------------------------------------------------------------------------
