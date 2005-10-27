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

#ifndef __MTX_FILE_H
#define __MTX_FILE_H

#include <dolfin/constants.h>
#include <dolfin/GenericFile.h>

namespace dolfin
{
  
  // FIXME: remove all these defines, and use private static consts
  // FIXME: instead if necessary
  

  // Defines for Matrix Market constants
  typedef char MM_typecode[4];
  
#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64
  
#define MM_MTX_STR         "matrix"
#define MM_ARRAY_STR       "array"
#define MM_DENSE_STR       "array"
#define MM_COORDINATE_STR  "coordinate"
#define MM_SPARSE_STR      "coordinate"
#define MM_COMPLEX_STR     "complex"
#define MM_REAL_STR        "real"
#define MM_INT_STR         "integer"
#define MM_GENERAL_STR     "general"
#define MM_SYMM_STR        "symmetric"
#define MM_HERM_STR        "hermitian"
#define MM_SKEW_STR        "skew-symmetric"
#define MM_PATTERN_STR     "pattern"
  
  // FIXME: Remove all these macros and implement as private static
  // FIXME: functions if they are needed

#define mm_is_matrix(typecode)       ((typecode)[0]=='M')
  
#define mm_is_sparse(typecode)       ((typecode)[1]=='C')
#define mm_is_coordinate(typecode)   ((typecode)[1]=='C')
#define mm_is_dense(typecode)        ((typecode)[1]=='A')
#define mm_is_array(typecode)        ((typecode)[1]=='A')
  
#define mm_is_complex(typecode)      ((typecode)[2]=='C')
#define mm_is_real(typecode)         ((typecode)[2]=='R')
#define mm_is_pattern(typecode)      ((typecode)[2]=='P')
#define mm_is_integer(typecode)      ((typecode)[2]=='I')
  
#define mm_is_symmetric(typecode)    ((typecode)[3]=='S')
#define mm_is_general(typecode)      ((typecode)[3]=='G')
#define mm_is_skew(typecode)         ((typecode)[3]=='K')
#define mm_is_hermitian(typecode)    ((typecode)[3]=='H')
  
#define mm_set_matrix(typecode)      ((*typecode)[0]='M')
  
#define mm_set_coordinate(typecode)  ((*typecode)[1]='C')
#define mm_set_array(typecode)       ((*typecode)[1]='A')
#define mm_set_dense(typecode)       mm_set_array(typecode)
#define mm_set_sparse(typecode)      mm_set_coordinate(typecode)
  
#define mm_set_complex(typecode)     ((*typecode)[2]='C')
#define mm_set_real(typecode)        ((*typecode)[2]='R')
#define mm_set_pattern(typecode)     ((*typecode)[2]='P')
#define mm_set_integer(typecode)     ((*typecode)[2]='I')
  
#define mm_set_symmetric(typecode)   ((*typecode)[3]='S')
#define mm_set_general(typecode)     ((*typecode)[3]='G')
#define mm_set_skew(typecode)        ((*typecode)[3]='K')
#define mm_set_hermitian(typecode)   ((*typecode)[3]='H')
  
  // FIXME: Remove

#define MM_COULD_NOT_READ_FILE  11
#define MM_PREMATURE_EOF        12
#define MM_NOT_MTX              13
#define MM_NO_HEADER            14
#define MM_UNSUPPORTED_TYPE     15
#define MM_LINE_TOO_LONG        16
#define MM_COULD_NOT_WRITE_FILE 17

  class Vector;
  class Matrix;
  class Mesh;
  class ParameterList;
  
  class MTXFile : public GenericFile 
  {
  public:
    
    MTXFile(const std::string filename);
    ~MTXFile();
    
    // Input
    
    void operator>> (Vector& x);
    void operator>> (Matrix& A);
    
    // Output
    
    void operator<< (Vector& x);
    void operator<< (Matrix& A);
    
  private:
    
    // Functions to do the MM operations
    int  mm_read_banner(FILE *f, MM_typecode *matcode);
    int  mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz);
    int  mm_read_mtx_array_size(FILE *f, int *M, int *N);
    
    int  mm_write_banner(FILE *f, MM_typecode matcode);
    int  mm_write_mtx_crd_size(FILE *f, int M, int N, int nz);
    int  mm_write_mtx_array_size(FILE *f, int M, int N);
    
    void mm_clear_typecode(MM_typecode *typecode);
    void mm_initialize_typecode(MM_typecode *typecode);

    // FIXME: remove
    char *strdup(const char *s);

    // FIXME: rewrite
    char *mm_typecode_to_str(MM_typecode matcode);
    
  };
}

#endif
