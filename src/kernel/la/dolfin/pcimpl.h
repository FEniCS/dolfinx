/* $Id: pcimpl.h,v 1.1 2005/02/11 08:06:58 johanjan Exp $ */

#ifndef _PCIMPL
#define _PCIMPL

#include <petsc/petscksp.h>
#include <petsc/petscpc.h>

typedef struct _PCOps *PCOps;
struct _PCOps {
  int          (*setup)(PC);
  int          (*apply)(PC,Vec,Vec);
  int          (*applyrichardson)(PC,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,int);
  int          (*applyBA)(PC,int,Vec,Vec,Vec);
  int          (*applytranspose)(PC,Vec,Vec);
  int          (*applyBAtranspose)(PC,int,Vec,Vec,Vec);
  int          (*setfromoptions)(PC);
  int          (*presolve)(PC,KSP,Vec,Vec);
  int          (*postsolve)(PC,KSP,Vec,Vec);  
  int          (*getfactoredmatrix)(PC,Mat*);
  int          (*applysymmetricleft)(PC,Vec,Vec);
  int          (*applysymmetricright)(PC,Vec,Vec);
  int          (*setuponblocks)(PC);
  int          (*destroy)(PC);
  int          (*view)(PC,PetscViewer);
};

/*
   Preconditioner context
*/
struct _p_PC {
  PETSCHEADER(struct _PCOps)
  int           setupcalled;
  MatStructure  flag;
  Mat           mat,pmat;
  Vec           vec;
  Vec           diagonalscaleright,diagonalscaleleft; /* used for time integration scaling */
  PetscTruth    diagonalscale;
  MatNullSpace  nullsp;
  int           (*modifysubmatrices)(PC,int,const IS[],const IS[],Mat[],void*); /* user provided routine */
  void          *modifysubmatricesP; /* context for user routine */
  void          *data;
};



#endif
