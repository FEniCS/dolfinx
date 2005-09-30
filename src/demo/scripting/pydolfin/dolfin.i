%module dolfin
%{
#include <dolfin.h>
#include <string>

namespace dolfin {

%}

%include "std_string.i"



%typemap(python,out) real = double; 
%typemap(python,in) uint = int; 

%import "dolfin.h"

%rename(increment) dolfin::NodeIterator::operator++;
%rename(increment) dolfin::CellIterator::operator++;
%rename(increment) dolfin::EdgeIterator::operator++;


/* DOLFIN public interface */

/* main includes */

%include "dolfin/constants.h"
%include "dolfin/init.h"

/* common includes */

/* System.h seems to be obsolete? */

%include "dolfin/Array.h"
%include "dolfin/List.h"
%include "dolfin/TimeDependent.h"
%include "dolfin/Variable.h"
%include "dolfin/meminfo.h"
%include "dolfin/sysinfo.h"
%include "dolfin/timeinfo.h"
%include "dolfin/utils.h"

/* io includes */

%include "dolfin/File.h"

/* la includes */

%include "dolfin/Vector.h"
%include "dolfin/Matrix.h"
%include "dolfin/VirtualMatrix.h"
%include "dolfin/GMRES.h"
%include "dolfin/LinearSolver.h"
%include "dolfin/EigenvalueSolver.h"
%include "dolfin/Preconditioner.h"
%include "dolfin/PETScManager.h"

/* function includes */

%include "dolfin/Function.h"

/* fem includes */

%include "dolfin/FEM.h"
%include "dolfin/FiniteElement.h"
%include "dolfin/AffineMap.h"
%include "dolfin/BoundaryValue.h"
%include "dolfin/BoundaryCondition.h"

/* form includes */

%include "dolfin/Form.h"
%include "dolfin/BilinearForm.h"
%include "dolfin/LinearForm.h"

/* mesh includes */

%include "dolfin/Mesh.h"
%include "dolfin/Boundary.h"
%include "dolfin/Point.h"
%include "dolfin/Node.h"
%include "dolfin/Edge.h"
%include "dolfin/Triangle.h"
%include "dolfin/Tetrahedron.h"
%include "dolfin/Cell.h"
%include "dolfin/Edge.h"
%include "dolfin/Face.h"
%include "dolfin/NodeIterator.h"
%include "dolfin/CellIterator.h"
%include "dolfin/EdgeIterator.h"
%include "dolfin/FaceIterator.h"
%include "dolfin/MeshIterator.h"
%include "dolfin/MeshIterator.h"
%include "dolfin/MeshIterator.h"

/* modules */

/* poisson */

%include "dolfin/PoissonSolver.h"
