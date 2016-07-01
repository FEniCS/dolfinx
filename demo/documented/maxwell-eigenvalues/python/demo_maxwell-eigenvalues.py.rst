Stable and unstable finite elements for the Maxwell eigenvalue problem
======================================================================

The Maxwell eigenvalue problem seeks eigenvalues :math:`\lambda` and
the the corresponding nonzero vector-valued
eigenfunctions :math:`u` satisfying
the partial differential equation

.. math::
    \operatorname{curl}\operatorname{curl} u = \lambda u \text{ in $\Omega$}

(we have simplified slightly by setting the material parameters equal to 1).
The PDE is to be supplemented with boundary conditions, which we take here
to be the essential boundary condition

.. math::
    u \times n = 0 \text{ on $\partial\Omega$}.

The eigenvalues :math:`\lambda` are all real and non-negative, but only the positive
ones are of interest, since, if :math:`\lambda >0`,
then it follows from the PDE that :math:`\operatorname{div} u = 0`, which is
also a requirement of Maxwell's equations.  There exist, in addition,
an infinite-dimensional family of eigenfunctions with eigenvalue :math:`\lambda=0`,
since for any smooth function :math:`\phi` vanishing to second order on the boundary,
:math:`u=\operatorname{grad}\phi` is such an eigenfunction.  But these functions are not
divergence-free and should not be considered Maxwell eigenfunctions.

Model problem
-------------

In this demo we shall consider the Maxwell eigenvalue problem in two dimensions with the
domain :math:`\Omega` taken to be the square :math:`(0,\pi)\times(0,\pi)`, since it that
case the exact eigenpairs have a simple analytic expression.  They are

.. math::
    u(x,y) = (\sin m x, \sin n y), \quad \lambda = m^2 + n^2,

for any non-negative integers :math:`m` and :math:`n`, not both zero.  Thus the eigenvalues
are

.. math::
    \lambda = 1, 1, 2, 4, 4, 5, 5, 8, 9, 9, 10, 10, 13, 13, \dots
    
In the demo program we compute the 12 eigenvalues nearest 5.5, and so should obtain
the first 12 numbers on this list, ranging from 1 to 10.

The weak formulation and the finite element method
--------------------------------------------------

A weak formulation of the eigenvalue problem seeks :math:`0\ne u\in H_0(\operatorname{curl})`
and :math:`\lambda>0` such that

.. math::
    \int_{\Omega} \operatorname{curl} u\, \operatorname{curl}v\, {\rm d} x = \lambda
    \int_{\Omega} u v\, {\rm d} x\quad\forall \ v\in H_0(\operatorname{curl}),

where :math:`H_0(\operatorname{curl})` is the space of square-integrable vector
fields with square-integrable curl and satisfying the essential boundary condition.
If we replace :math:`H_0(\operatorname{curl})` in this formulation by a finite element
subspace, we obtain a finite element method.

Stable and unstable finite elements
-----------------------------------

We consider here two possible choices of finite element spaces.  The first,
the Nedelec edge elements, which are obtained in FEniCS as ``FunctionSpace(mesh, 'H1curl', 1)``,
are well suited to this problem and give an
accurate discretization.  The second choice is simply the vector-valued Lagrange
piecewise linears: ``VectorFunctionSpace(mesh, 'Lagrange', 1)``.  To the uninitiated it usually
comes as a surprise that the Lagrange elements do not provide an accurate discretization of
the Maxwell eigenvalue problem:
the computed eigenvalues do not converge to the true ones as the mesh is refined!
This is a subtle matter connected to the stability theory of mixed finite element methods.
See `this paper <http://umn.edu/~arnold/papers/icm2002.pdf>`_ for details.

While the Nedelec elements behave stably for any mesh, the failure of the Lagrange elements
differs on different sorts of meshes.  Here we compute with two structured meshes, the first
obtained from a :math:`40\times 40` grid of squares by dividing each with its positively-sloped diagonal, and
the second the crossed mesh obtained by dividing each subsquare into four using both diagonals.
The output from the first case is:

.. code-block:: none

    diagonal mesh
    Nedelec:   [ 1.00  1.00  2.00  4.00  4.00  5.00  5.00  8.01  8.98  8.99  9.99  9.99]
    Lagrange:  [ 5.16  5.26  5.26  5.30  5.39  5.45  5.53  5.61  5.61  5.62  5.71  5.73]
    Exact:     [ 1.00  1.00  2.00  4.00  4.00  5.00  5.00  8.00  9.00  9.00 10.00 10.00]

Note that the eigenvalues calculated using the Nedelec elements are all correct to within a fraction
of a percent. But the 12 eigenvalues computed by the Lagrange elements are certainly all *wrong*,
since they are far from being integers!

On the crossed mesh, we obtain a different mode of failure:

.. code-block:: none

    crossed mesh
    Nedelec:   [ 1.00  1.00  2.00  4.00  4.00  5.00  5.00  7.99  9.00  9.00 10.00 10.00]
    Lagrange:  [ 1.00  1.00  2.00  4.00  4.00  5.00  5.00  6.00  8.01  9.01  9.01 10.02]
    Exact:     [ 1.00  1.00  2.00  4.00  4.00  5.00  5.00  8.00  9.00  9.00 10.00 10.00]

Again the Nedelec elements are accurate.  The Lagrange elements also approximate most
of the eigenvalues well, but they return a *totally spurious* value of 6.00 as well.
If we were to compute more eigenvalues, more spurious ones would be returned.
This mode of failure might be considered more dangerous, since it is harder to spot.

The implementation
------------------

First we import ``dolfin`` and ``numpy`` and make sure that dolfin has been configured with PETSc
and SLEPc (since we depend on the SLEPc eigenvalue solver). ::

    # demo_maxwell-eigenvalues.py, contributed by Patrick E. Farrell and Douglas N. Arnold, 2016
    from dolfin import *
    import numpy as np
    if not has_linear_algebra_backend("PETSc"):
        print("DOLFIN has not been configured with PETSc. Exiting.")
        exit()
    if not has_slepc():
        print("DOLFIN has not been configured with SLEPc. Exiting.")
        exit()

Given the finite element space ``V`` and the essential boundary conditions ``bcs`` for it,
the function ``eigenvalues(V, bcs)`` solves the Maxwell eigenvalue problem and
returns the requested eigenvalues in a sorted numpy array.  It consists of four steps:

    1. Define the bilinear forms on the right- and left-hand sides of the weak formulation.

    2. Assemble these into PETSc matrices to obtain a generalized matrix eigenvalue problem
       :math:`Ax=\lambda B x`, imposing the essential boundary conditions using
       ``assemble_system`` in order to maintain symmetry.  Since ``assemble_system``
       requires a right-hand side linear form even though irrelevant
       for the eigenvalue problem, we define a dummy form. We also zero out the rows
       of :math:`B` corresponding to the boundary DOFs, so as not to introduce
       spurious eigenpairs with nonzero boundary DOFs.

    3. Solve the symmetric generalized matrix eigenvalue problem using SLEPc's Krylov-Schur solver,
       requesting the 12 eigenvalues nearest 5.5.
    
    4. Return the computed eigenvalues in a sorted array.
    
::

    def eigenvalues(V, bcs):

        # define the bilinear forms on the right- and left-hand sides
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(curl(u), curl(v))*dx
        b = inner(u, v)*dx
        # assemble into PETSc matrices
        dummy = v[0]*dx
        A = PETScMatrix()
        assemble_system(a, dummy, bcs, A_tensor=A)
        B = PETScMatrix()
        assemble_system(b, dummy, bcs, A_tensor=B)
        [bc.zero(B) for bc in bcs]
        # solve the generalize matrix eigenvalue problem
        solver = SLEPcEigenSolver(A, B)
        solver.parameters["solver"] = "krylov-schur"
        solver.parameters["problem_type"] = "gen_hermitian"
        solver.parameters["spectrum"] = "target magnitude"
        solver.parameters["spectral_transform"] = "shift-and-invert"
        solver.parameters["spectral_shift"] = 5.5
        neigs = 12
        solver.solve(neigs)
        # return the computed eigenvalues in a sorted array
        computed_eigenvalues = []
        for i in range(min(neigs, solver.get_number_converged())):
            r, _ = solver.get_eigenvalue(i) # ignore the imaginary part
            computed_eigenvalues.append(r)
        return np.sort(np.array(computed_eigenvalues))

        
Given a mesh, the function ``print_eigenvalues(mesh)`` calls ``eigenvalues`` to solve the Maxwell eigenvalue problem for
each of the two finite element spaces, and prints the results, together with the known exact eigenvalues.
Note that, since the degrees of freedom for the Nedelec edge space are values of the tangential component,
specifying the boundary conditions for it just means setting all DOFs on the boundary to zero.  However,
for the vector Lagrange space, both components are DOFs, so we much specify which component
must vanish (the x-component on horizontal edges and the y-component on vertical edges).

::

    def print_eigenvalues(mesh):

        nedelec_V   = FunctionSpace(mesh, "N1curl", 1)
        nedelec_bcs = [DirichletBC(nedelec_V, Constant((0.0, 0.0)), DomainBoundary())]
        nedelec_eig = eigenvalues(nedelec_V, nedelec_bcs)

        lagrange_V   = VectorFunctionSpace(mesh, "Lagrange", 1)
        lagrange_bcs = [DirichletBC(lagrange_V.sub(1), 0, "near(x[0], 0) || near(x[0], pi)"),
                        DirichletBC(lagrange_V.sub(0), 0, "near(x[1], 0) || near(x[1], pi)")]
        lagrange_eig = eigenvalues(lagrange_V, lagrange_bcs)

        true_eig = np.sort(np.array([float(m**2 + n**2) for m in range(6) for n in range(6)]))[1:13]

        np.set_printoptions(formatter={'float': '{:5.2f}'.format})
        print "Nedelec:  ",
        print nedelec_eig 
        print "Lagrange: ",
        print lagrange_eig
        print "Exact:    ",
        print true_eig

Finally, we display the results for each of two different meshes. ::

    mesh = RectangleMesh(Point(0, 0), Point(pi, pi), 40, 40)
    print("\ndiagonal mesh")
    print_eigenvalues(mesh)

    mesh = RectangleMesh(Point(0, 0), Point(pi, pi), 40, 40, "crossed")
    print("\ncrossed mesh")
    print_eigenvalues(mesh)
