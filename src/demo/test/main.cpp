
/* Program usage:  mpirun ex1 [-help] [all PETSc options] */

static char help[] = "Basic vector routines.\n\n";

/*T
   Concepts: vectors^basic routines;
   Processors: n
T*/

/* 
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petsc.h       - base PETSc routines   petscis.h     - index sets
     petscsys.h    - system routines       petscviewer.h - viewers
*/

#include "petscvec.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec            x,y,w;               /* vectors */
  Vec            *z;                    /* array of vectors */
  PetscReal      norm,v,v1,v2;
  PetscInt       n = 20;
  PetscErrorCode ierr;
  PetscTruth     flg;
  PetscScalar    one = 1.0,two = 2.0,three = 3.0,dots[3],dot;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  /* 
     Create a vector, specifying only its global dimension.
     When using VecCreate(), VecSetSizes() and VecSetFromOptions(), the vector format 
     (currently parallel, shared, or sequential) is determined at runtime.  Also, the 
     parallel partitioning of the vector is determined by PETSc at runtime.

     Routines for creating particular vector types directly are:
        VecCreateSeq() - uniprocessor vector
        VecCreateMPI() - distributed vector, where the user can
                         determine the parallel partitioning
        VecCreateShared() - parallel vector that uses shared memory
                            (available only on the SGI); otherwise,
                            is the same as VecCreateMPI()

     With VecCreate(), VecSetSizes() and VecSetFromOptions() the option -vec_type mpi or 
     -vec_type shared causes the particular type of vector to be formed.

  */

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  /*
     Duplicate some work vectors (of the same format and
     partitioning as the initial vector).
  */
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&w);CHKERRQ(ierr);
  ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);


  /*
     Duplicate more work vectors (of the same format and
     partitioning as the initial vector).  Here we duplicate
     an array of vectors, which is often more convenient than
     duplicating individual ones.
  */
  ierr = VecDuplicateVecs(x,3,&z);CHKERRQ(ierr); 

  /*
     Set the vectors to entries to a constant value.
  */
  ierr = VecSet(&one,x);CHKERRQ(ierr);
  ierr = VecSet(&two,y);CHKERRQ(ierr);
  ierr = VecSet(&one,z[0]);CHKERRQ(ierr);
  ierr = VecSet(&two,z[1]);CHKERRQ(ierr);
  ierr = VecSet(&three,z[2]);CHKERRQ(ierr);

  /*
     Demonstrate various basic vector routines.
  */
  ierr = VecDot(x,x,&dot);CHKERRQ(ierr);
  ierr = VecMDot(3,x,z,dots);CHKERRQ(ierr);

  /* 
     Note: If using a complex numbers version of PETSc, then
     PETSC_USE_COMPLEX is defined in the makefiles; otherwise,
     (when using real numbers) it is undefined.
  */
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Vector length %D\n",(PetscInt) (PetscRealPart(dot)));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Vector length %D %D %D\n",(PetscInt)PetscRealPart(dots[0]),
                             (PetscInt)PetscRealPart(dots[1]),(PetscInt)PetscRealPart(dots[2]));CHKERRQ(ierr);
#else
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Vector length %D\n",(PetscInt)dot);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Vector length %D %D %D\n",(PetscInt)dots[0],
                             (PetscInt)dots[1],(PetscInt)dots[2]);CHKERRQ(ierr);
#endif

  ierr = PetscPrintf(PETSC_COMM_WORLD,"All other values should be near zero\n");CHKERRQ(ierr);

  ierr = VecScale(&two,x);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  v = norm-2.0*sqrt((double)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0; 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecScale %g\n",v);CHKERRQ(ierr);


  ierr = VecCopy(x,w);CHKERRQ(ierr);
  ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);
  v = norm-2.0*sqrt((double)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0; 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecCopy  %g\n",v);CHKERRQ(ierr);

  ierr = VecAXPY(&three,x,y);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  v = norm-8.0*sqrt((double)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0; 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecAXPY %g\n",v);CHKERRQ(ierr);

  ierr = VecAYPX(&two,x,y);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  v = norm-18.0*sqrt((double)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0; 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecAYPX %g\n",v);CHKERRQ(ierr);

  ierr = VecSwap(x,y);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  v = norm-2.0*sqrt((double)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0; 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecSwap  %g\n",v);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  v = norm-18.0*sqrt((double)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0; 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecSwap  %g\n",v);CHKERRQ(ierr);

  ierr = VecWAXPY(&two,x,y,w);CHKERRQ(ierr);
  ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);
  v = norm-38.0*sqrt((double)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0; 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecWAXPY %g\n",v);CHKERRQ(ierr);

  ierr = VecPointwiseMult(y,x,w);CHKERRQ(ierr);
  ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr); 
  v = norm-36.0*sqrt((double)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0; 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecPointwiseMult %g\n",v);CHKERRQ(ierr);

  ierr = VecPointwiseDivide(x,y,w);CHKERRQ(ierr);
  ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);
  v = norm-9.0*sqrt((double)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0; 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecPointwiseDivide %g\n",v);CHKERRQ(ierr);

  dots[0] = one;
  dots[1] = three;
  dots[2] = two;
  ierr = VecSet(&one,x);CHKERRQ(ierr);
  ierr = VecMAXPY(3,dots,x,z);CHKERRQ(ierr);
  ierr = VecNorm(z[0],NORM_2,&norm);CHKERRQ(ierr);
  v = norm-sqrt((double)n); if (v > -PETSC_SMALL && v < PETSC_SMALL) v = 0.0; 
  ierr = VecNorm(z[1],NORM_2,&norm);CHKERRQ(ierr);
  v1 = norm-2.0*sqrt((double)n); if (v1 > -PETSC_SMALL && v1 < PETSC_SMALL) v1 = 0.0; 
  ierr = VecNorm(z[2],NORM_2,&norm);CHKERRQ(ierr);
  v2 = norm-3.0*sqrt((double)n); if (v2 > -PETSC_SMALL && v2 < PETSC_SMALL) v2 = 0.0; 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"VecMAXPY %g %g %g \n",v,v1,v2);CHKERRQ(ierr);

  /* 
     Test whether vector has been corrupted (just to demonstrate this
     routine) not needed in most application codes.
  */
  ierr = VecValid(x,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(1,"Corrupted vector.");

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  ierr = VecDestroy(w);CHKERRQ(ierr);
  ierr = VecDestroyVecs(z,3);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
