#include <dolfin/SparseMatrix.h>
#include "XMLSparseMatrix.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLSparseMatrix::XMLSparseMatrix(SparseMatrix *sparseMatrix) : XMLObject()
{
  this->sparseMatrix = sparseMatrix;
  state = OUTSIDE;
  row = 0;
}
//-----------------------------------------------------------------------------
void XMLSparseMatrix::startElement(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state ){
  case OUTSIDE:

	 if ( xmlStrcasecmp(name,(xmlChar *) "sparsematrix") == 0 ){
		readSparseMatrix(name,attrs);
		state = INSIDE_SPARSE_MATRIX;
	 }
	 
	 break;
  case INSIDE_SPARSE_MATRIX:
	 
	 if ( xmlStrcasecmp(name,(xmlChar *) "row") == 0 ){
		readRow(name,attrs);
		state = INSIDE_ROW;
	 }
	 
	 break;
	 
  case INSIDE_ROW:
	 
	 if ( xmlStrcasecmp(name,(xmlChar *) "element") == 0 )
		readElement(name,attrs);
	 
	 break;
  }
  
}
//-----------------------------------------------------------------------------
void XMLSparseMatrix::endElement(const xmlChar *name)
{
  switch ( state ){
  case INSIDE_SPARSE_MATRIX:
	 
	 if ( xmlStrcasecmp(name,(xmlChar *) "sparsematrix") == 0 ){
		ok = true;
		state = DONE;
	 }
	 
	 break;
  case INSIDE_ROW:

	 if ( xmlStrcasecmp(name,(xmlChar *) "row") == 0 )
		state = INSIDE_SPARSE_MATRIX;
	 
	 break;
  }

}
//-----------------------------------------------------------------------------
void XMLSparseMatrix::readSparseMatrix(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int rows = 0;
  int columns = 0;

  // Parse values
  parseIntegerRequired(name, attrs, "rows",    &rows);
  parseIntegerRequired(name, attrs, "columns", &columns);

  // Set values
  sparseMatrix->init(rows, columns);
}
//-----------------------------------------------------------------------------
void XMLSparseMatrix::readRow(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  row = 0;
  int size = 0;
  
  // Parse values
  parseIntegerRequired(name, attrs, "row", &row);
  parseIntegerRequired(name, attrs, "size", &size);

  // Set values
  sparseMatrix->initRow(row, size);
}
//-----------------------------------------------------------------------------
void XMLSparseMatrix::readElement(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int column = 0;
  real value = 0.0;
  
  // Parse values
  parseIntegerRequired (name, attrs, "column", &column);
  parseRealRequired    (name, attrs, "value",  &value);
  
  // Set values
  (*sparseMatrix)(row,column) = value;
}
//-----------------------------------------------------------------------------
