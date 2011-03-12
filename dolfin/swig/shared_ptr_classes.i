/* -*- C -*- */
// Copyright (C) 2005-2006 Johan Hake
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders logg, 2009.
// Modified by Garth N. Wells, 2009.
//
// First added:  2007-11-25
// Last changed: 2011-03-11

//=============================================================================
// SWIG directives for the shared_ptr stored classes in PyDOLFIN
//
// Objects of these classes can then be passed to c++ functions
// demanding a boost::shared_ptr<type>
//=============================================================================

//-----------------------------------------------------------------------------
// Handle shared_ptr only available for swig version >= 1.3.35
//-----------------------------------------------------------------------------
#if SWIG_VERSION >= 0x010335

//-----------------------------------------------------------------------------
// Un-comment these lines to use std::tr1, only works with swig version >=1.3.37
//-----------------------------------------------------------------------------
//#define SWIG_SHARED_PTR_NAMESPACE std
//#define SWIG_SHARED_PTR_SUBNAMESPACE tr1

//-----------------------------------------------------------------------------
// Include macros for shared_ptr support
//-----------------------------------------------------------------------------
%include <boost_shared_ptr.i>

//-----------------------------------------------------------------------------
// Make DOLFIN aware of the types defined in UFC
//-----------------------------------------------------------------------------
%import "swig/ufc.i"

//-----------------------------------------------------------------------------
// Declare shared_ptr stored types in PyDOLFIN
//-----------------------------------------------------------------------------
#if SWIG_VERSION >= 0x020000

// adaptivity
%shared_ptr(dolfin::TimeSeries)

// common
%shared_ptr(dolfin::Variable)

// fem
%shared_ptr(dolfin::Hierarchical<dolfin::Form>)
%shared_ptr(dolfin::GenericDofMap)
%shared_ptr(dolfin::DofMap)
%shared_ptr(dolfin::Form)
%shared_ptr(dolfin::FiniteElement)

%shared_ptr(dolfin::Hierarchical<dolfin::VariationalProblem>)
%shared_ptr(dolfin::VariationalProblem)

%shared_ptr(dolfin::BoundaryCondition)
%shared_ptr(dolfin::Hierarchical<dolfin::DirichletBC>)
%shared_ptr(dolfin::DirichletBC)
%shared_ptr(dolfin::EqualityBC)
%shared_ptr(dolfin::PeriodicBC)

// function
%shared_ptr(dolfin::Hierarchical<dolfin::FunctionSpace>)
%shared_ptr(dolfin::FunctionSpace)
%shared_ptr(dolfin::SubSpace)

%shared_ptr(dolfin::GenericFunction)
%shared_ptr(dolfin::Hierarchical<dolfin::Function>)
%shared_ptr(dolfin::Function)
%shared_ptr(dolfin::Expression)
%shared_ptr(dolfin::FacetArea)
%shared_ptr(dolfin::Constant)
%shared_ptr(dolfin::MeshCoordinates)

// mesh
%shared_ptr(dolfin::Hierarchical<dolfin::Mesh>)
%shared_ptr(dolfin::Mesh)
%shared_ptr(dolfin::BoundaryMesh)
%shared_ptr(dolfin::SubMesh)
%shared_ptr(dolfin::UnitTetrahedron)
%shared_ptr(dolfin::UnitCube)
%shared_ptr(dolfin::UnitInterval)
%shared_ptr(dolfin::Interval)
%shared_ptr(dolfin::UnitTriangle)
%shared_ptr(dolfin::UnitSquare)
%shared_ptr(dolfin::UnitCircle)
%shared_ptr(dolfin::Box)
%shared_ptr(dolfin::Rectangle)
%shared_ptr(dolfin::UnitSphere)

%shared_ptr(dolfin::SubDomain)
%shared_ptr(dolfin::DomainBoundary)

// la
%shared_ptr(dolfin::GenericTensor)
%shared_ptr(dolfin::GenericVector)
%shared_ptr(dolfin::GenericMatrix)
%shared_ptr(dolfin::Scalar)

%shared_ptr(dolfin::Matrix)
%shared_ptr(dolfin::Vector)

%shared_ptr(dolfin::STLMatrix)
%shared_ptr(dolfin::uBLASMatrix<boost::numeric::ublas::matrix<double> >)
%shared_ptr(dolfin::uBLASMatrix<boost::numeric::ublas::compressed_matrix<double,\
	    boost::numeric::ublas::row_major> >)
%shared_ptr(dolfin::uBLASVector)

#ifdef HAS_PETSC
%shared_ptr(dolfin::PETScBaseMatrix)
%shared_ptr(dolfin::PETScKrylovMatrix)
%shared_ptr(dolfin::PETScKrylovSolver)
%shared_ptr(dolfin::PETScLUSolver)
%shared_ptr(dolfin::PETScMatrix)
%shared_ptr(dolfin::PETScObject)
%shared_ptr(dolfin::PETScPreconditioner)
%shared_ptr(dolfin::PETScVector)
%shared_ptr(dolfin::PETScUserPreconditioner)
#endif

#ifdef HAS_SLEPC
%shared_ptr(dolfin::SLEPcEigenSolver)
#endif

#ifdef HAS_MTL4
%shared_ptr(dolfin::ITLKrylovSolver)
%shared_ptr(dolfin::MTL4Matrix)
%shared_ptr(dolfin::MTL4Vector)
#endif

#ifdef HAS_TRILINOS
%shared_ptr(dolfin::EpetraKrylovSolver)
%shared_ptr(dolfin::EpetraLUSolver)
%shared_ptr(dolfin::EpetraMatrix)
%shared_ptr(dolfin::EpetraSparsityPattern)
%shared_ptr(dolfin::EpetraVector)
%shared_ptr(dolfin::TrilinosPreconditioner)
#endif

%shared_ptr(dolfin::UmfpackLUSolver)
%shared_ptr(dolfin::CholmodCholeskySolver)

%shared_ptr(dolfin::uBLASKrylovSolver)

%shared_ptr(dolfin::LinearSolver)
%shared_ptr(dolfin::GenericLinearSolver)
%shared_ptr(dolfin::GenericLUSolver)
%shared_ptr(dolfin::KrylovSolver)
%shared_ptr(dolfin::LUSolver)
%shared_ptr(dolfin::SingularSolver)

%shared_ptr(dolfin::GenericSparsityPattern)
%shared_ptr(dolfin::SparsityPattern)

// log
%shared_ptr(dolfin::Table)

// mesh
%shared_ptr(dolfin::LocalMeshData)
%shared_ptr(dolfin::MeshData)


%shared_ptr(dolfin::MeshFunction<bool>)
%shared_ptr(dolfin::MeshFunction<double>)
%shared_ptr(dolfin::MeshFunction<int>)
%shared_ptr(dolfin::MeshFunction<dolfin::uint>)
%shared_ptr(dolfin::MeshFunction<unsigned int>)

%shared_ptr(dolfin::CellFunction<bool>)
%shared_ptr(dolfin::CellFunction<double>)
%shared_ptr(dolfin::CellFunction<int>)
%shared_ptr(dolfin::CellFunction<dolfin::uint>)
%shared_ptr(dolfin::CellFunction<unsigned int>)

%shared_ptr(dolfin::EdgeFunction<bool>)
%shared_ptr(dolfin::EdgeFunction<double>)
%shared_ptr(dolfin::EdgeFunction<int>)
%shared_ptr(dolfin::EdgeFunction<dolfin::uint>)
%shared_ptr(dolfin::EdgeFunction<unsigned int>)

%shared_ptr(dolfin::FaceFunction<bool>)
%shared_ptr(dolfin::FaceFunction<double>)
%shared_ptr(dolfin::FaceFunction<int>)
%shared_ptr(dolfin::FaceFunction<dolfin::uint>)
%shared_ptr(dolfin::FaceFunction<unsigned int>)

%shared_ptr(dolfin::FacetFunction<bool>)
%shared_ptr(dolfin::FacetFunction<double>)
%shared_ptr(dolfin::FacetFunction<int>)
%shared_ptr(dolfin::FacetFunction<dolfin::uint>)
%shared_ptr(dolfin::FacetFunction<unsigned int>)

%shared_ptr(dolfin::VertexFunction<bool>)
%shared_ptr(dolfin::VertexFunction<double>)
%shared_ptr(dolfin::VertexFunction<int>)
%shared_ptr(dolfin::VertexFunction<dolfin::uint>)
%shared_ptr(dolfin::VertexFunction<unsigned int>)

// math
%shared_ptr(dolfin::Lagrange)

// nls
%shared_ptr(dolfin::NewtonSolver)

// ode
%shared_ptr(dolfin::Sample)
%shared_ptr(dolfin::ODE)
%shared_ptr(dolfin::DummyComplexODE)
%shared_ptr(dolfin::ComplexODE)
%shared_ptr(dolfin::Method)
%shared_ptr(dolfin::dGqMethod)
%shared_ptr(dolfin::cGqMethod)

// plot
%shared_ptr(dolfin::FunctionPlotData)

// quadrature
%shared_ptr(dolfin::Quadrature)
%shared_ptr(dolfin::LobattoQuadrature)
%shared_ptr(dolfin::RadauQuadrature)
%shared_ptr(dolfin::GaussQuadrature)
%shared_ptr(dolfin::GaussianQuadrature)

#else

 //SWIG_SHARED_PTR(HierarchicalForm, dolfin::Hierarchical<dolfin::Form>)
SWIG_SHARED_PTR(GenericDofMap, dolfin::GenericDofMap)
SWIG_SHARED_PTR_DERIVED(DofMap, dolfin::GenericDofMap, dolfin::DofMap)
SWIG_SHARED_PTR(FiniteElement, dolfin::FiniteElement)
SWIG_SHARED_PTR(Form, dolfin::Form)

//SWIG_SHARED_PTR(HierarchicalVariationalProblem,			\
//		dolfin::Hierarchical<dolfin::VariationalProblem>)
//SWIG_SHARED_PTR_DERIVED(VariationalProblem,				\
//			dolfin::Hierarchical<dolfin::VariationalProblem>, \
//			dolfin::VariationalProblem)
//SWIG_SHARED_PTR(HierarchicalFunctionSpace, dolfin::Hierarchical<dolfin::FunctionSpace>)
//SWIG_SHARED_PTR_DERIVED(FunctionSpace, dolfin::Hierarchical<dolfin::FunctionSpace>, \
//			dolfin::FunctionSpace)
SWIG_SHARED_PTR(FunctionSpace, dolfin::FunctionSpace)
SWIG_SHARED_PTR_DERIVED(SubSpace, dolfin::FunctionSpace, dolfin::SubSpace)

//SWIG_SHARED_PTR(HierarchicalFunction, dolfin::Hierarchical<dolfin::Function>)
SWIG_SHARED_PTR(GenericFunction, dolfin::GenericFunction)
SWIG_SHARED_PTR_DERIVED(Function, dolfin::GenericFunction, dolfin::Function)
//SWIG_SHARED_PTR_DERIVED(Function, dolfin::Hierarchical<dolfin::Function>, \
//			dolfin::Function)
SWIG_SHARED_PTR_DERIVED(Expression, dolfin::GenericFunction, dolfin::Expression)
SWIG_SHARED_PTR_DERIVED(FacetArea, dolfin::Expression, dolfin::FacetArea)
SWIG_SHARED_PTR_DERIVED(Constant, dolfin::Expression, dolfin::Constant)
SWIG_SHARED_PTR_DERIVED(MeshCoordinates, dolfin::Expression, dolfin::MeshCoordinates)


// Forward declare shared pointer declarations so typemaps will be available
SWIG_SHARED_PTR(MeshFunctionUInt, dolfin::MeshFunction<unsigned int>)
SWIG_SHARED_PTR(MeshFunctionInt, dolfin::MeshFunction<int>)
SWIG_SHARED_PTR(MeshFunctionDouble, dolfin::MeshFunction<double>)
SWIG_SHARED_PTR(MeshFunctionBool, dolfin::MeshFunction<bool>)

SWIG_SHARED_PTR(Mesh, dolfin::Mesh)
SWIG_SHARED_PTR_DERIVED(BoundaryMesh, dolfin::Mesh, dolfin::BoundaryMesh)
SWIG_SHARED_PTR_DERIVED(SubMesh, dolfin::Mesh, dolfin::SubMesh)
SWIG_SHARED_PTR_DERIVED(UnitTetrahedron, dolfin::Mesh, dolfin::UnitTetrahedron)
SWIG_SHARED_PTR_DERIVED(UnitCube, dolfin::Mesh, dolfin::UnitCube)
SWIG_SHARED_PTR_DERIVED(UnitInterval, dolfin::Mesh, dolfin::UnitInterval)
SWIG_SHARED_PTR_DERIVED(Interval, dolfin::Mesh, dolfin::Interval)
SWIG_SHARED_PTR_DERIVED(UnitTriangle, dolfin::Mesh, dolfin::UnitTriangle)
SWIG_SHARED_PTR_DERIVED(UnitSquare, dolfin::Mesh, dolfin::UnitSquare)
SWIG_SHARED_PTR_DERIVED(UnitCircle, dolfin::Mesh, dolfin::UnitCircle)
SWIG_SHARED_PTR_DERIVED(Box, dolfin::Mesh, dolfin::Box)
SWIG_SHARED_PTR_DERIVED(Rectangle, dolfin::Mesh, dolfin::Rectangle)
SWIG_SHARED_PTR_DERIVED(UnitSphere, dolfin::Mesh, dolfin::UnitSphere)

SWIG_SHARED_PTR(SubDomain, dolfin::SubDomain)
SWIG_SHARED_PTR_DERIVED(DomainBoundary, dolfin::SubDomain, dolfin::DomainBoundary)

SWIG_SHARED_PTR(GenericTensor, dolfin::GenericTensor)
SWIG_SHARED_PTR_DERIVED(GenericVector, dolfin::GenericTensor, dolfin::GenericVector)
SWIG_SHARED_PTR_DERIVED(GenericMatrix, dolfin::GenericTensor, dolfin::GenericMatrix)
SWIG_SHARED_PTR_DERIVED(Scalar, dolfin::GenericTensor, dolfin::Scalar)
SWIG_SHARED_PTR_DERIVED(Matrix, dolfin::GenericMatrix, dolfin::Matrix)
SWIG_SHARED_PTR_DERIVED(Vector, dolfin::GenericVector, dolfin::Vector)

SWIG_SHARED_PTR_DERIVED(uBLASDenseMatrix, dolfin::GenericMatrix, \
			dolfin::uBLASMatrix<dolfin::ublas_dense_matrix>)
SWIG_SHARED_PTR_DERIVED(uBLASSparseMatrix, dolfin::GenericMatrix, \
			dolfin::uBLASMatrix<dolfin::ublas_sparse_matrix>)
SWIG_SHARED_PTR_DERIVED(uBLASVector, dolfin::GenericVector, dolfin::uBLASVector)
SWIG_SHARED_PTR_DERIVED(STLMatrix, dolfin::GenericMatrix, dolfin::STLMatrix)

#ifdef HAS_TRILINOS
SWIG_SHARED_PTR_DERIVED(EpetraMatrix, dolfin::GenericMatrix, dolfin::EpetraMatrix)
SWIG_SHARED_PTR_DERIVED(EpetraVector, dolfin::GenericVector, dolfin::EpetraVector)
#endif

#ifdef HAS_PETSC
SWIG_SHARED_PTR_DERIVED(PETScMatrix, dolfin::GenericMatrix, dolfin::PETScMatrix)
SWIG_SHARED_PTR_DERIVED(PETScVector, dolfin::GenericVector, dolfin::PETScVector)
#endif

#ifdef HAS_MTL4
SWIG_SHARED_PTR_DERIVED(MTL4Matrix, dolfin::GenericMatrix, dolfin::MTL4Matrix)
SWIG_SHARED_PTR_DERIVED(MTL4Vector, dolfin::GenericVector, dolfin::MTL4Vector)
#endif

SWIG_SHARED_PTR(FunctionPlotData, dolfin::FunctionPlotData)

#endif

//-----------------------------------------------------------------------------
// Macro that exposes the Variable interface for the derived classes
// This is a hack to get around the problem that Variable is not declared
// as a shared_ptr class.
//
// Ideally we would like to make Variable a shared_ptr type, but we do not want
// to make all derived classes shared_ptr types. This means we need to implement
// the Variable interface for derived types of Variable.
//-----------------------------------------------------------------------------
%define IMPLEMENT_VARIABLE_INTERFACE(DERIVED_TYPE)
%ignore dolfin::DERIVED_TYPE::str;

%extend dolfin::DERIVED_TYPE
{
  void rename(const std::string name, const std::string label)
  {
    self->rename(name, label);
  }

  const std::string& name() const
  {
    return self->name();
  }

  const std::string& label() const
  {
    return self->label();
  }

  std::string __str__() const
  {
    return self->str(false);
  }

  std::string _str(bool verbose) const
  {
    return self->str(verbose);
  }

  dolfin::Parameters& _get_parameters()
  {
    return self->parameters;
  }

%pythoncode %{
    def str(self, verbose):
        "Return a string representation of it self"
        return self._str(verbose)

    def _get_parameters(self):
        return _cpp. ## DERIVED_TYPE ## __get_parameters(self)

    parameters = property(_get_parameters)
%}

}

%enddef

//-----------------------------------------------------------------------------
// Include knowledge of the NoDeleter template, used in macros below
//-----------------------------------------------------------------------------
%{
#include <dolfin/common/NoDeleter.h>
%}

//-----------------------------------------------------------------------------
// Run the macros on derived classes of Variable that is defined shared_ptr
//-----------------------------------------------------------------------------
IMPLEMENT_VARIABLE_INTERFACE(GenericFunction)
IMPLEMENT_VARIABLE_INTERFACE(FunctionSpace)
IMPLEMENT_VARIABLE_INTERFACE(Mesh)
IMPLEMENT_VARIABLE_INTERFACE(GenericDofMap)
IMPLEMENT_VARIABLE_INTERFACE(GenericTensor)
#if SWIG_VERSION >= 0x020000
IMPLEMENT_VARIABLE_INTERFACE(VariationalProblem)
#endif


//-----------------------------------------------------------------------------
// Macros for defining in and out typemaps for foreign types that DOLFIN
// use as in and ouput from functions. More specific Epetra_FEFoo
// FIXME: Make these const aware...
//-----------------------------------------------------------------------------
%define FOREIGN_SHARED_PTR_TYPEMAPS(TYPE)

//-----------------------------------------------------------------------------
// Make swig aware of the type and the shared_ptr version of it
//-----------------------------------------------------------------------------
%types(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<TYPE>*, TYPE*);

//-----------------------------------------------------------------------------
// Typecheck
//-----------------------------------------------------------------------------
%typecheck(SWIG_TYPECHECK_POINTER) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<TYPE> {
  int res = SWIG_ConvertPtr($input, 0, SWIGTYPE_p_ ## TYPE, 0);
  $1 = SWIG_CheckState(res);
}

//-----------------------------------------------------------------------------
// In typemap
//-----------------------------------------------------------------------------
%typemap(in) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<TYPE> {
  void *argp = 0;
  TYPE * arg = 0;
  int res = SWIG_ConvertPtr($input, &argp, $descriptor(TYPE*), 0);
  if (SWIG_IsOK(res)) {
    arg = reinterpret_cast<TYPE *>(argp);
    $1 = dolfin::reference_to_no_delete_pointer(*arg);
  }
  else
    SWIG_exception(SWIG_TypeError, "expected a TYPE");
}

//-----------------------------------------------------------------------------
// In typemap
//-----------------------------------------------------------------------------
%typemap(out) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<TYPE> {
  TYPE * out = $1.get();
  $result = SWIG_NewPointerObj(SWIG_as_voidptr(out), $descriptor(TYPE*), 0 | 0 );
}
%enddef

#if SWIG_VERSION < 0x020000
//-----------------------------------------------------------------------------
// Macro that exposes the Hierarchical interface for the derived classes
// This is a hack to get around the problem that Hierarchical is not declared
// as a shared_ptr class in SWIG version < 2.0
//-----------------------------------------------------------------------------
%define IMPLEMENT_HIERARCHICAL_INTERFACE(DERIVED_TYPE)
%extend dolfin::DERIVED_TYPE
{
    unsigned int depth() const
    { return self->depth(); }

    bool has_parent() const
    { return self->has_parent();}

    bool has_child() const
    { return self->has_child(); }

    boost::shared_ptr<DERIVED_TYPE> parent()
    { return self->parent_shared_ptr(); }

    boost::shared_ptr<DERIVED_TYPE> child()
    { return self->child_shared_ptr(); }

    boost::shared_ptr<DERIVED_TYPE> coarse()
    { return self->coarse_shared_ptr(); }

    boost::shared_ptr<DERIVED_TYPE> fine()
    { return self->fine_shared_ptr(); }

    void set_parent(boost::shared_ptr<DERIVED_TYPE> parent)
    { self->set_parent(parent); }

    void set_child(boost::shared_ptr<DERIVED_TYPE> child)
    { self->set_child(child); }

    void _debug() const
    { self->_debug(); }

    //const Hierarchical& assignoperator= (const Hierarchical& hierarchical)
    //{
    //}
}

%enddef

IMPLEMENT_HIERARCHICAL_INTERFACE(Mesh)
IMPLEMENT_HIERARCHICAL_INTERFACE(Form)
IMPLEMENT_HIERARCHICAL_INTERFACE(FunctionSpace)
IMPLEMENT_HIERARCHICAL_INTERFACE(Function)
#endif

#endif
