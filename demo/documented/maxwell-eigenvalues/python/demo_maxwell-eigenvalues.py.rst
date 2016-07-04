.. codeauthor:: Douglas N. Arnold <arnold@umn.edu>, Patrick E. Farrell <patrick.farrell@maths.ox.ac.uk>

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
domain :math:`\Omega` taken to be the square :math:`(0,\pi)\times(0,\pi)`, since in that
case the exact eigenpairs have a simple analytic expression.  They are

.. math::
    u(x,y) = (\sin m x, \sin n y), \quad \lambda = m^2 + n^2,

for any non-negative integers :math:`m` and :math:`n,` not both zero.  Thus the eigenvalues
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
the Nédélec edge elements, which are obtained in FEniCS as ``FunctionSpace(mesh, 'H1curl', 1)``,
are well suited to this problem and give an
accurate discretization.  The second choice is simply the vector-valued Lagrange
piecewise linears: ``VectorFunctionSpace(mesh, 'Lagrange', 1)``.  To the uninitiated it usually
comes as a surprise that the Lagrange elements do not provide an accurate discretization of
the Maxwell eigenvalue problem:
the computed eigenvalues do not converge to the true ones as the mesh is refined!
This is a subtle matter connected to the stability theory of mixed finite element methods.
See `this paper <http://umn.edu/~arnold/papers/icm2002.pdf>`_ for details.

While the Nédélec elements behave stably for any mesh, the failure of the Lagrange elements
differs on different sorts of meshes.  Here we compute with two structured meshes, the first
obtained from a :math:`40\times 40` grid of squares by dividing each with its positively-sloped diagonal, and
the second the crossed mesh obtained by dividing each subsquare into four using both diagonals.
The output from the first case is:

.. code-block:: none

    diagonal mesh
    Nédélec:   [ 1.00  1.00  2.00  4.00  4.00  5.00  5.00  8.01  8.98  8.99  9.99  9.99]
    Lagrange:  [ 5.16  5.26  5.26  5.30  5.39  5.45  5.53  5.61  5.61  5.62  5.71  5.73]
    Exact:     [ 1.00  1.00  2.00  4.00  4.00  5.00  5.00  8.00  9.00  9.00 10.00 10.00]

Note that the eigenvalues calculated using the Nédélec elements are all correct to within a fraction
of a percent. But the 12 eigenvalues computed by the Lagrange elements are certainly all *wrong*,
since they are far from being integers!

On the crossed mesh, we obtain a different mode of failure:

.. code-block:: none

    crossed mesh
    Nédélec:   [ 1.00  1.00  2.00  4.00  4.00  5.00  5.00  7.99  9.00  9.00 10.00 10.00]
    Lagrange:  [ 1.00  1.00  2.00  4.00  4.00  5.00  5.00  6.00  8.01  9.01  9.01 10.02]
    Exact:     [ 1.00  1.00  2.00  4.00  4.00  5.00  5.00  8.00  9.00  9.00 10.00 10.00]

Again the Nédélec elements are accurate.  The Lagrange elements also approximate most
of the eigenvalues well, but they return a *totally spurious* value of 6.00 as well.
If we were to compute more eigenvalues, more spurious ones would be returned.
This mode of failure might be considered more dangerous, since it is harder to spot.

The implementation
------------------

**Preamble.**
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

**Function eigenvalues.**
The function ``eigenvalues`` takes the finite element space ``V`` and the
essential boundary conditions ``bcs`` for it, and returns a requested
set of Maxwell eigenvalues (specified in the code below)
as a sorted numpy array::

    def eigenvalues(V, bcs):

We start by defining the bilinear forms on the right- and left-hand sides of the weak formulation.

::

    #
        # define the bilinear forms on the right- and left-hand sides
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(curl(u), curl(v))*dx
        b = inner(u, v)*dx

Next we assemble the bilinear forms ``a`` and ``b`` into PETSc matrices
``A`` and ``B``, so the eigenvalue problem is converted into a generalized
matrix eigenvalue problem :math:`Ax=\lambda B x`.  During the assembly step
the essential boundary conditions are incorporated by modifying the rows
of the matrices corresponding to constrained boundary degrees of freedom.
We use ``assemble_system`` rather than ``assemble`` to do the assembly,
since it maintains the symmetry of the matrices.  ``assemble_system``
is designed for source problems, rather than eigenvalue problems, and
requires a right-hand side linear form, so we define a dummy form to feed it.

::

    #
        # assemble into PETSc matrices
        dummy = v[0]*dx
        A = PETScMatrix()
        assemble_system(a, dummy, bcs, A_tensor=A)
        B = PETScMatrix()
        assemble_system(b, dummy, bcs, A_tensor=B)

We zero out the rows of :math:`B` corresponding to constrained
boundary degrees of freedom, so as not to introduce
spurious eigenpairs with nonzero boundary DOFs.

::

    #
        [bc.zero(B) for bc in bcs]

Now we solve the generalized matrix eigenvalue problem
using the SLEPc package.  The behavior of the ``SLEPcEigenSolver`` is
controlled by a parameter set (use ``info(solver, True)`` to
see all possible parameters).  We use parameters to set
the eigensolution method to Krylov-Schur,
which is good for computing a subset of the eigenvalues
of a sparse matrix, and to tell SLEPc that the matrices
``A`` and ``B`` in the generalized eigenvalue problem are symmetric
(Hermitian).

::

    #
        solver = SLEPcEigenSolver(A, B)
        solver.parameters["solver"] = "krylov-schur"
        solver.parameters["problem_type"] = "gen_hermitian"
        
We specify that we want 12 eigenvalues nearest in magnitude to
a target value of 5.5.  Note that when the ``spectrum`` parameter
is set to ``target magnitude``, the ``spectral_transform`` parameter
should be set to ``shift-and-invert`` and the ``spectral_shift``
parameter should be set equal to the target.

::

    #
        solver.parameters["spectrum"] = "target magnitude"
        solver.parameters["spectral_transform"] = "shift-and-invert"
        solver.parameters["spectral_shift"] = 5.5
        neigs = 12
        solver.solve(neigs)
        
Finally we collect the computed eigenvalues in list which we convert
to a numpy array and sort before returning.  Note that we are
not guaranteed to get the number of eigenvalues requested.
The function ``solver.get_number_converged()`` reports the actual number
of eigenvalues computed, which may be more or less than the
number requested.

::

    #
        # return the computed eigenvalues in a sorted array
        computed_eigenvalues = []
        for i in range(min(neigs, solver.get_number_converged())):
            r, _ = solver.get_eigenvalue(i) # ignore the imaginary part
            computed_eigenvalues.append(r)
        return np.sort(np.array(computed_eigenvalues))
   
**Function print_eigenvalues.**
Given just a mesh, the function ``print_eigenvalues`` calls the preceding
function ``eigenvalues`` to solve the Maxwell eigenvalue problem for
each of the two finite element spaces, Nédélec and Lagrange, and prints the results, together
with the known exact eigenvalues::

    def print_eigenvalues(mesh):

First we define the Nédélec edge element space and the essential boundary
conditions for it, and call ``eigenvalues`` to compute the eigenvalues.
Since the degrees of freedom for the Nédélec space
are tangential components on element edges, we simply need to constrained
all the DOFs associated to boundary points to zero.

::

    #
        nedelec_V   = FunctionSpace(mesh, "N1curl", 1)
        nedelec_bcs = [DirichletBC(nedelec_V, Constant((0.0, 0.0)), DomainBoundary())]
        nedelec_eig = eigenvalues(nedelec_V, nedelec_bcs)

Then we do the same for the vector Lagrange elements.  Since the Lagrange DOFs
are both components of the vector, we must specify which component must vanish
on which edges (the x-component on horizontal edges and the y-component on vertical
edges).

::

    #
        lagrange_V   = VectorFunctionSpace(mesh, "Lagrange", 1)
        lagrange_bcs = [DirichletBC(lagrange_V.sub(1), 0, "near(x[0], 0) || near(x[0], pi)"),
                        DirichletBC(lagrange_V.sub(0), 0, "near(x[1], 0) || near(x[1], pi)")]
        lagrange_eig = eigenvalues(lagrange_V, lagrange_bcs)

The true eigenvalues are  just the 12 smallest numbers of the form :math:`m^2 + n^2`, :math:`m,n\ge0`,
not counting 0.

::

    #
        true_eig = np.sort(np.array([float(m**2 + n**2) for m in range(6) for n in range(6)]))[1:13]

Finally we print the results::

    #
        np.set_printoptions(formatter={'float': '{:5.2f}'.format})
        print "Nedelec:  ",
        print nedelec_eig 
        print "Lagrange: ",
        print lagrange_eig
        print "Exact:    ",
        print true_eig

**Calling the functions.**
To complete the program, we call ``print_eigenvalues`` for each of two different meshes::

    mesh = RectangleMesh(Point(0, 0), Point(pi, pi), 40, 40)
    print("\ndiagonal mesh")
    print_eigenvalues(mesh)

    mesh = RectangleMesh(Point(0, 0), Point(pi, pi), 40, 40, "crossed")
    print("\ncrossed mesh")
    print_eigenvalues(mesh)

