// Copyright (C) 2005 Haiko Etzel.
// Licensed under the GNU GPL Version 2.
//
// Uses code from http://math.nist.gov/MatrixMarket/mmio/c/mmio.c
// which is in the public domain.
//
// Modified by Anders Logg 2005.
//
// First added:  2005-10-18
// Last changed: 2005-10-26

#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Mesh.h>
#include <dolfin/Parameter.h>
#include <dolfin/ParameterList.h>
#include <dolfin/MTXFile.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MTXFile::MTXFile(const std::string filename) : GenericFile(filename)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MTXFile::~MTXFile()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MTXFile::operator>>(Vector& x)
{
  MM_typecode matcode;
  FILE* f;
  int M,N;
  int i;
  real* val;

  if ((f = fopen(filename.c_str(), "r")) == NULL)
    dolfin_error("Unable to open Matrix Market inputfile.");

  if (MTXFile::mm_read_banner(f, &matcode) != 0)
    dolfin_error("Could not process Matrix Market banner.");

  // Check if Market Market type is implemented.
  if (
      mm_is_sparse   (matcode) ||
      mm_is_complex  (matcode) ||
      mm_is_pattern  (matcode) ||
      mm_is_symmetric(matcode) ||
      mm_is_hermitian(matcode) ||
      mm_is_skew     (matcode)
      )
    dolfin_error("Used Matrix Market type is not implemented.");

  // find out size of vector
  if (mm_read_mtx_array_size(f, &M, &N) !=0)
    dolfin_error("Could not parse matrix size.");

  if (N!=1)
    dolfin_error("Array width for Vector needs to be 1.");

  // resize x
  x.init(M);

  // set pointer to array of Vector
  val=x.array();

  // read from File and store into the Matrix
  for (i=0; i<M; i++)
  {
    fscanf(f, "%lg\n", &val[i]);
  }
  x.restore(val);

  if (f !=stdin) fclose(f);
}
//-----------------------------------------------------------------------------
void MTXFile::operator>>(Matrix& A)
{
  MM_typecode matcode;
  FILE *f;
  int M, N, nz;
  int i;

  if ((f = fopen(filename.c_str(), "r")) == NULL)
    dolfin_error("Unable to open Matrix Market inputfile.");

  if (MTXFile::mm_read_banner(f, &matcode) != 0)
    dolfin_error("Could not process Matrix Market banner.");

  // Check if Market Market type is implemented.
  if (
      mm_is_dense    (matcode) ||
      mm_is_complex  (matcode) ||
      mm_is_pattern  (matcode) ||
      mm_is_symmetric(matcode) ||
      mm_is_hermitian(matcode) ||
      mm_is_skew     (matcode)
      )
    dolfin_error("Used Matrix Market type is not implemented.");

  // find out size of matrix
  if (MTXFile::mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
  {
    dolfin_error("Could not parse matrix size.");
  }
  // resize A
  A.init(M,N,nz);

  // read from File and store into the Matrix A
  int k,l;
  real value;
  for (i=0; i<nz; i++)
  {
    fscanf(f, "%d %d %lg\n", &k, &l, &value);
    A(k-1,l-1)=value;
  }

  if (f !=stdin) fclose(f);
}
//-----------------------------------------------------------------------------
void MTXFile::operator<<(Vector& x)
{
  FILE *f;
  int m  = x.size();

  f = fopen(filename.c_str(), "a") ;

  MM_typecode matcode;
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_array(&matcode);
  mm_set_real(&matcode);

  mm_write_banner(f, matcode);
  fprintf(f,"%% Vector generated with dolfin\n");
  mm_write_mtx_array_size(f, m, 1);

  // NOTE: matrix market files use 1-based indices, i.e. 
  // first element of a vector has index 1, not 0.  
  for (int i=0; i<m; i++)
  {
    fprintf(f, " %- 20.19g\n", real(x(i)));
  }
  if (f !=stdout) 
    fclose(f);

  dolfin_info("Saved vector to file %s in Matrix Market format.",
	      filename.c_str());
}
//-----------------------------------------------------------------------------
void MTXFile::operator<<(Matrix& A)
{
  FILE *f;
  int m  = A.size(0);
  int n  = A.size(1);
  int nz = 0;
  real aij=0;    
  MM_typecode matcode;

  // open file
  f = fopen(filename.c_str(), "a"); 

  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_coordinate(&matcode);
  mm_set_real(&matcode);

  // NOTE: matrix market files use 1-based indices, i.e.
  // first element of a vector has index 1, not 0. 
  nz=0;
  for (int i=0; i<m; i++)
  {
    for (int j=0; j<m; j++)
    {
      if (std::abs(A(i,j) - 0.0)> DOLFIN_EPS)
      {
	nz++;
      }
    }
  }

  // write to file
  mm_write_banner(f, matcode);
  fprintf(f,"%% Matrix generated with dolfin\n");
  mm_write_mtx_crd_size(f, m, n, nz);

  // NOTE: matrix market files use 1-based indices, i.e.
  // first element of a vector/matrix has index 1, not 0.
  for (int i=0; i<m; i++)
  {
    for (int j=0; j<m; j++)
    {
      aij = A(i,j);
      if (std::abs(aij - 0.0)> DOLFIN_EPS)
      {
	fprintf(f, "   %4d   %4d   %- 20.19g\n", i+1, j+1, aij);
      }
    }
  }

  if (f !=stdout) fclose(f);    

  dolfin_info("Saved matrix to file %s in Matrix Market format.",
	      filename.c_str());
}
//-----------------------------------------------------------------------------
int MTXFile::mm_read_banner(FILE *f, MM_typecode *matcode)
{
  char line[MM_MAX_LINE_LENGTH];
  char banner[MM_MAX_TOKEN_LENGTH];
  char mtx[MM_MAX_TOKEN_LENGTH];
  char crd[MM_MAX_TOKEN_LENGTH];
  char data_type[MM_MAX_TOKEN_LENGTH];
  char storage_scheme[MM_MAX_TOKEN_LENGTH];
  char *p;

  mm_clear_typecode(matcode);

  if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
  {
    dolfin_error("premature EOF");
    return 1;
  }

  if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type,
	     storage_scheme) != 5)
  {
    dolfin_error("premature EOF");
    return 1;
  }

  for (p=mtx; *p!='\0'; *p=tolower(*p),p++);  /* convert to lower case */
  for (p=crd; *p!='\0'; *p=tolower(*p),p++);
  for (p=data_type; *p!='\0'; *p=tolower(*p),p++);
  for (p=storage_scheme; *p!='\0'; *p=tolower(*p),p++);

  // check for banner
  if (strncmp(banner, "%%MatrixMarket", strlen("%%MatrixMarket")) != 0)
  {
    dolfin_error("no header in file"); 
    return 0;
  }

  // first field should be "mtx"
  if (strcmp(mtx, "matrix") != 0)
  {
    dolfin_error("unsupported type in first field"); 
    return 0;
  }
  mm_set_matrix(matcode);

  // second field describes whether this is a sparse matrix 
  // (in coordinate storage) or a dense array

  if (strcmp(crd, "coordinate") == 0)
    mm_set_sparse(matcode);
  else
    if (strcmp(crd, "array") == 0)
      mm_set_dense(matcode);
    else
    {
      dolfin_error("unsupported type in second field"); 
      return 0;
    }

  // third field

  if (strcmp(data_type, "real") == 0)
    mm_set_real(matcode);
  else
    if (strcmp(data_type, "complex") == 0)
      mm_set_complex(matcode);
    else
      if (strcmp(data_type, "pattern") == 0)
	mm_set_pattern(matcode);
      else
	if (strcmp(data_type, "integer") == 0)
	  mm_set_integer(matcode);
	else
	{
	  dolfin_error("unsupported type in third field"); 
	  return 0;
	}

  // fourth field

  if (strcmp(storage_scheme, "general") == 0)
    mm_set_general(matcode);
  else
    if (strcmp(storage_scheme, "symmetric") == 0)
      mm_set_symmetric(matcode);
    else
      if (strcmp(storage_scheme, "hermitian") == 0)
	mm_set_hermitian(matcode);
      else
	if (strcmp(storage_scheme, "skew-symmetric") == 0)
	  mm_set_skew(matcode);
	else
	{
	  dolfin_error("unsupported type in fourth field"); 
	  return 0;
	}

  return 0;
}
//-----------------------------------------------------------------------------
int MTXFile::mm_write_mtx_crd_size(FILE *f, int M, int N, int nz)
{
  if (fprintf(f, "%d %d %d\n", M, N, nz) < 3)
  {
    dolfin_error("Unable to write file");
    return 0;
  }
  else
    return 0;
}
//-----------------------------------------------------------------------------
int MTXFile::mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz )
{
  char line[MM_MAX_LINE_LENGTH];
  int num_items_read;

  // set return null parameter values, in case we exit with errors
  *M = *N = *nz = 0;

  // now continue scanning until you reach the end-of-comments
  do
  {
    if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL)
    {
      dolfin_error("premature EOF");
      return 1;
    }
    
  }
  while (line[0] == '%');

  // line[] is either blank or has M,N, nz
  if (sscanf(line, "%d %d %d", M, N, nz) == 3)
    return 0;

  else
    do
    {
      num_items_read = fscanf(f, "%d %d %d", M, N, nz);
      if (num_items_read == EOF)
      {
        dolfin_error("premature EOF");
        return 1;
      }
    }
    while (num_items_read != 3);

  return 0;
}
//-----------------------------------------------------------------------------
int MTXFile::mm_read_mtx_array_size(FILE *f, int *M, int *N)
{
  char line[MM_MAX_LINE_LENGTH];
  int num_items_read;
  // set return null parameter values, in case we exit with errors
  *M = *N = 0;

  // now continue scanning until you reach the end-of-comments
  do
  {
    if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL)
    {
      dolfin_error("premature EOF");
      return 1;
    }
  }
  while (line[0] == '%');

  // line[] is either blank or has M,N, nz
  if (sscanf(line, "%d %d", M, N) == 2)
    return 0;

  else // we have a blank line
    do
    {
      num_items_read = fscanf(f, "%d %d", M, N);
      if (num_items_read == EOF) 
      {
        dolfin_error("premature EOF");
        return 1;
      }
      
    }
    while (num_items_read != 2);

  return 0;
}
//-----------------------------------------------------------------------------
int MTXFile::mm_write_mtx_array_size(FILE *f, int M, int N)
{
  if (fprintf(f, "%d %d\n", M, N) < 0)
  {
    dolfin_error("Unable to write file");
    return 0;
  }
  else
    return 0;
}
//-----------------------------------------------------------------------------
int MTXFile::mm_write_banner(FILE *f, MM_typecode matcode)
{
  char* str=new char[MM_MAX_LINE_LENGTH];
  MTXFile::mm_typecode_to_str(matcode, str);
  int ret_code;

  ret_code = fprintf(f, "%%MatrixMarket %s\n", str);
  delete[] str;
  if (ret_code < 0 )
  {
    dolfin_error("Unable to write file");
    return 0;
  }
  else
    return 0;
}
//-----------------------------------------------------------------------------
int  MTXFile::mm_typecode_to_str(MM_typecode matcode, char* buffer)
{
  char *types[4] = {0, 0, 0, 0};

  // check for MTX type
  if (mm_is_matrix(matcode))
    types[0] = "matrix";
  else
  {
    dolfin_error("invalid matrix type");
    return 0;
  }

  // check for CRD or ARR matrix
  if (mm_is_sparse(matcode))
    types[1] = "coordinate";
  else
    if (mm_is_dense(matcode))
      types[1] = "array";
    else
    {
      dolfin_error("invalid spares/dense specification");
      return 0;
    }

  // check for element data type
  if (mm_is_real(matcode))
    types[2] = "real";
  else
    if (mm_is_complex(matcode))
      types[2] = "complex";
    else
      if (mm_is_pattern(matcode))
	types[2] = "pattern";
      else
	if (mm_is_integer(matcode))
	  types[2] = "integer";
	else
	{
	  dolfin_error("invalid data type");
	  return 0;
	}

  // check for symmetry type
  if (mm_is_general(matcode))
    types[3] = "general";
  else
    if (mm_is_symmetric(matcode))
      types[3] = "symmetric";
    else
      if (mm_is_hermitian(matcode))
	types[3] = "hermitian";
      else
	if (mm_is_skew(matcode))
	  types[3] = "skew-symmetric";
	else
	{
	  dolfin_error("invalid symmetry type");
	  return 0;
	}

  strcat (buffer,types[0]);
  strcat (buffer," ");
  strcat (buffer,types[1]);
  strcat (buffer," ");
  strcat (buffer,types[2]);
  strcat (buffer," ");
  strcat (buffer,types[3]);
  return 0;
}
//-----------------------------------------------------------------------------
void MTXFile::mm_clear_typecode(MM_typecode *typecode)
{
  (*typecode)[0]=(*typecode)[1]=(*typecode)[2]=' ';
  (*typecode)[3]='G';
}
//-----------------------------------------------------------------------------
void MTXFile::mm_initialize_typecode(MM_typecode *typecode)
{
  MTXFile::mm_clear_typecode(typecode);
}
//-----------------------------------------------------------------------------
inline bool MTXFile::mm_is_matrix(MM_typecode typecode)
{
  return typecode[0]==MM_MATRIX;
}
//-----------------------------------------------------------------------------
inline bool MTXFile::mm_is_sparse(MM_typecode typecode)
{
  return typecode[1]==MM_COORDINATE;
}
//-----------------------------------------------------------------------------
inline bool MTXFile::mm_is_coordinate(MM_typecode typecode)
{
  return typecode[1]==MM_COORDINATE;
}
//-----------------------------------------------------------------------------
inline bool MTXFile::mm_is_dense(MM_typecode typecode)
{
  return typecode[1]==MM_ARRAY;
}
//-----------------------------------------------------------------------------
inline bool MTXFile::mm_is_array(MM_typecode typecode)
{
  return typecode[1]==MM_ARRAY;
}
//-----------------------------------------------------------------------------
inline bool MTXFile::mm_is_complex(MM_typecode typecode)
{
  return typecode[2]==MM_COMPLEX;
}
//-----------------------------------------------------------------------------
inline bool MTXFile::mm_is_real(MM_typecode typecode)
{
  return typecode[2]==MM_REAL;
}
//-----------------------------------------------------------------------------
inline bool MTXFile::mm_is_pattern(MM_typecode typecode)
{
  return typecode[2]==MM_PATTERN;
}
//-----------------------------------------------------------------------------
inline bool MTXFile::mm_is_integer(MM_typecode typecode)
{
  return typecode[2]==MM_INTEGER;
}
//-----------------------------------------------------------------------------
inline bool MTXFile::mm_is_symmetric(MM_typecode typecode)
{
  return typecode[3]==MM_SYMMETRIC;
}
//-----------------------------------------------------------------------------
inline bool MTXFile::mm_is_general(MM_typecode typecode)
{
  return typecode[3]==MM_GENERAL;
}
//-----------------------------------------------------------------------------
inline bool MTXFile::mm_is_skew(MM_typecode typecode)
{
  return typecode[3]==MM_SKEW;
}
//-----------------------------------------------------------------------------
inline bool MTXFile::mm_is_hermitian(MM_typecode typecode)
{
  return typecode[3]==MM_HERMITIAN;
}
//-----------------------------------------------------------------------------
inline void MTXFile::mm_set_matrix(MM_typecode* typecode)
{
  (*typecode)[0]=MM_MATRIX;
}
//-----------------------------------------------------------------------------
inline void MTXFile::mm_set_coordinate(MM_typecode* typecode)
{
  (*typecode)[1]=MM_COORDINATE;
}
//-----------------------------------------------------------------------------
inline void MTXFile::mm_set_array(MM_typecode* typecode)
{
  (*typecode)[1]=MM_ARRAY;
}
//-----------------------------------------------------------------------------
inline void MTXFile::mm_set_dense(MM_typecode* typecode)
{
  mm_set_array(typecode);
}
//-----------------------------------------------------------------------------
inline void MTXFile::mm_set_sparse(MM_typecode* typecode)
{
  mm_set_coordinate(typecode);
}
//-----------------------------------------------------------------------------
inline void MTXFile::mm_set_complex(MM_typecode* typecode)
{
  (*typecode)[2]=MM_COMPLEX;
}
//-----------------------------------------------------------------------------
inline void MTXFile::mm_set_real(MM_typecode* typecode)
{
  (*typecode)[2]=MM_REAL;
}
//-----------------------------------------------------------------------------
inline void MTXFile::mm_set_pattern(MM_typecode* typecode)
{
  (*typecode)[2]=MM_PATTERN;
}
//-----------------------------------------------------------------------------
inline void MTXFile::mm_set_integer(MM_typecode* typecode)
{
  (*typecode)[2]=MM_INTEGER;
}
//-----------------------------------------------------------------------------
inline void MTXFile::mm_set_symmetric(MM_typecode* typecode)
{
  (*typecode)[3]=MM_SYMMETRIC;
}
//-----------------------------------------------------------------------------
inline void MTXFile::mm_set_general(MM_typecode* typecode)
{
  (*typecode)[3]=MM_GENERAL;
}
//-----------------------------------------------------------------------------
inline void MTXFile::mm_set_skew(MM_typecode* typecode)
{
  (*typecode)[3]=MM_SKEW;
}
//-----------------------------------------------------------------------------
inline void MTXFile::mm_set_hermitian(MM_typecode* typecode)
{
  (*typecode)[3]=MM_HERMITIAN;
}
//-----------------------------------------------------------------------------
