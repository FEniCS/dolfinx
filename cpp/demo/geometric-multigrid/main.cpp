#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mpi.h>
#include <numeric>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <sys/types.h>
#include <vector>

int main(int argc, char** argv)
{
    int n_coarse = 8;
    int n_fine = 16;
    
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    
    KSP ksp;
    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetType(ksp, "preonly");

    PC pc;
    KSPGetPC(ksp, &pc);
    KSPSetFromOptions(ksp);
    PCSetType(pc, "mg");

    PCMGSetLevels(pc, 2, NULL);
    PCMGSetType(pc, PC_MG_MULTIPLICATIVE);
    PCMGSetCycleType(pc, PC_MG_CYCLE_V);
    PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH);
    PCMGSetNumberSmooth(pc, 2);
    PCSetFromOptions(pc);

    int64_t nz = n_fine + 2*n_fine; // fine vertices + averaging of coarse ones
    std::vector<PetscInt> i; // row indices
    i.reserve(nz);
    std::vector<PetscInt> j; // col indices
    j.reserve(nz);
    std::vector<PetscScalar> a;
    a.reserve(nz);
    for (int64_t idx = 0; idx < n_fine+1; idx ++)
    {
        if (idx % 2 == 0)
        {
            i.emplace_back(idx);
            j.emplace_back(PetscInt(idx/2.));
            a.emplace_back(1);
        } else {
            i.emplace_back(idx);
            j.emplace_back(floor(idx/2.));
            a.emplace_back(.5);            

            i.emplace_back(idx);
            j.emplace_back(ceil(idx/2.));
            a.emplace_back(.5);
        }
    }
    Mat interpolation;
    MatCreateSeqAIJFromTriple(MPI_COMM_SELF, n_fine+1, n_coarse+1, i.data(), j.data(), a.data(), &interpolation, a.size(), PETSC_FALSE);
    MatView(interpolation, PETSC_VIEWER_STDOUT_SELF);

    Mat restriction;
    MatTranspose(interpolation, MAT_INITIAL_MATRIX, &restriction);
    MatView(restriction, PETSC_VIEWER_STDOUT_SELF);

    PCMGSetInterpolation(pc, 1, interpolation);
    PCMGSetRestriction(pc, 1, restriction);

    Mat A;
    {
        std::vector<PetscInt> i(n_fine+1), j(n_fine+1);
        std::iota(i.begin(), i.end(), 0.0);
        std::iota(j.begin(), j.end(), 0.0);
        std::vector<PetscScalar> a(n_fine+1, 1.0);
        MatCreateSeqAIJFromTriple(MPI_COMM_SELF, n_fine+1, n_fine+1, i.data(), j.data(), a.data(), &A, a.size(), PETSC_FALSE);
    }
    // MatCreateConstantDiagonal(MPI_COMM_SELF, n_fine+1, n_fine+1, n_fine+1, n_fine+1, 1.0, &A);

    MatView(A, PETSC_VIEWER_STDOUT_SELF);

    KSPSetOperators(ksp, A, A);

    KSPSetUp(ksp);

    Vec x, b;
    MatCreateVecs(A, &x, &b);
    VecSet(b, 1.0);
    VecView(b, PETSC_VIEWER_STDOUT_SELF);

    KSPSolve(ksp, x, b);
    VecView(x, PETSC_VIEWER_STDOUT_SELF);

    KSPDestroy(&ksp);
}