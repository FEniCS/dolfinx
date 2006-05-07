// Copyright (C) 2005 Haiko Etzel.
// Licensed under the GNU GPL Version 2.
//
// Uses code from http://math.nist.gov/MatrixMarket/mmio/c/mmio.c
// which is in the public domain.
//
// Modified by Anders Logg 2005-2006.
//
// First added:  2005-10-18
// Last changed: 2006-05-07

#ifndef __MTX_FILE_H
#define __MTX_FILE_H

#ifdef HAVE_PETSC_H

#include <dolfin/constants.h>
#include <dolfin/GenericFile.h>

namespace dolfin
{
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
    
    // Defines for Matrix Market constants
    typedef char MM_typecode[4];
    
    static const int  MM_MAX_LINE_LENGTH =1025;
    static const int  MM_MAX_TOKEN_LENGTH=64;
    static const char MM_MATRIX     ='M';
    static const char MM_COORDINATE ='C';
    static const char MM_ARRAY      ='A';
    static const char MM_COMPLEX    ='C';
    static const char MM_REAL       ='R';
    static const char MM_PATTERN    ='P';
    static const char MM_INTEGER    ='I';
    static const char MM_SYMMETRIC  ='S';
    static const char MM_GENERAL    ='G';
    static const char MM_SKEW       ='K';
    static const char MM_HERMITIAN  ='H';
  
    // Functions to do the MM operations
    int  mm_read_banner(FILE *f, MM_typecode *matcode);
    int  mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz);
    int  mm_read_mtx_array_size(FILE *f, int *M, int *N);
    
    int  mm_write_banner(FILE *f, MM_typecode matcode);
    int  mm_write_mtx_crd_size(FILE *f, int M, int N, int nz);
    int  mm_write_mtx_array_size(FILE *f, int M, int N);
    
    void mm_clear_typecode(MM_typecode *typecode);
    void mm_initialize_typecode(MM_typecode *typecode);
    int  mm_typecode_to_str(MM_typecode matcode, char* buffer);

    inline bool mm_is_matrix    (MM_typecode typecode);

    inline bool mm_is_sparse    (MM_typecode typecode);
    inline bool mm_is_coordinate(MM_typecode typecode);
    inline bool mm_is_dense     (MM_typecode typecode);
    inline bool mm_is_array     (MM_typecode typecode);

    inline bool mm_is_complex   (MM_typecode typecode);
    inline bool mm_is_real      (MM_typecode typecode);
    inline bool mm_is_pattern   (MM_typecode typecode);
    inline bool mm_is_integer   (MM_typecode typecode);

    inline bool mm_is_symmetric (MM_typecode typecode);
    inline bool mm_is_general   (MM_typecode typecode);
    inline bool mm_is_skew      (MM_typecode typecode);
    inline bool mm_is_hermitian (MM_typecode typecode);

    inline void mm_set_matrix(MM_typecode* typecode);

    inline void mm_set_coordinate(MM_typecode* typecode);
    inline void mm_set_array(MM_typecode* typecode);
    inline void mm_set_dense(MM_typecode* typecode);
    inline void mm_set_sparse(MM_typecode* typecode);

    inline void mm_set_complex(MM_typecode* typecode);
    inline void mm_set_real(MM_typecode* typecode);
    inline void mm_set_pattern(MM_typecode* typecode);
    inline void mm_set_integer(MM_typecode* typecode);

    inline void mm_set_symmetric(MM_typecode* typecode);
    inline void mm_set_general(MM_typecode* typecode);
    inline void mm_set_skew(MM_typecode* typecode);
    inline void mm_set_hermitian(MM_typecode* typecode);

  };
}

#endif

#endif
