// Define classes in dolfin that are stored using shared_ptr
// Objects of these classes can then be passed to c++ functions
// demanding a boost::shared_ptr<type>

# if defined(SWIG_SHARED_PTR_QNAMESPACE)
%import "swig/ufc.i"

SWIG_SHARED_PTR(DofMap,dolfin::DofMap)
SWIG_SHARED_PTR(FiniteElement,dolfin::FiniteElement)

SWIG_SHARED_PTR(FunctionSpace,dolfin::FunctionSpace)
SWIG_SHARED_PTR_DERIVED(SubSpace,dolfin::FunctionSpace,dolfin::SubSpace)

SWIG_SHARED_PTR(Function,dolfin::Function)
SWIG_SHARED_PTR_DERIVED(Constant,dolfin::Function,dolfin::Constant)
SWIG_SHARED_PTR_DERIVED(MeshCoordinates,dolfin::Function,dolfin::MeshCoordinates)
SWIG_SHARED_PTR_DERIVED(CellSize,dolfin::Function,dolfin::CellSize)
SWIG_SHARED_PTR_DERIVED(FacetNormal,dolfin::Function,dolfin::FacetNormal)
SWIG_SHARED_PTR_DERIVED(FacetArea,dolfin::Function,dolfin::FacetArea)
SWIG_SHARED_PTR_DERIVED(SUPGStabilizer,dolfin::Function,dolfin::SUPGStabilizer)
SWIG_SHARED_PTR_DERIVED(DiscreteFunction,dolfin::Function,dolfin::DiscreteFunction)

SWIG_SHARED_PTR(Mesh,dolfin::Mesh)
SWIG_SHARED_PTR_DERIVED(BoundaryMesh,dolfin::Mesh,dolfin::BoundaryMesh)
SWIG_SHARED_PTR_DERIVED(SubMesh,dolfin::Mesh,dolfin::SubMesh)
SWIG_SHARED_PTR_DERIVED(UnitCube,dolfin::Mesh,dolfin::UnitCube)
SWIG_SHARED_PTR_DERIVED(UnitInterval,dolfin::Mesh,dolfin::UnitInterval)
SWIG_SHARED_PTR_DERIVED(Interval,dolfin::Mesh,dolfin::Interval)
SWIG_SHARED_PTR_DERIVED(UnitSquare,dolfin::Mesh,dolfin::UnitSquare)
SWIG_SHARED_PTR_DERIVED(UnitCircle,dolfin::Mesh,dolfin::UnitCircle)
SWIG_SHARED_PTR_DERIVED(Box,dolfin::Mesh,dolfin::Box)
SWIG_SHARED_PTR_DERIVED(Rectangle,dolfin::Mesh,dolfin::Rectangle)
SWIG_SHARED_PTR_DERIVED(UnitSphere,dolfin::Mesh,dolfin::UnitSphere)

SWIG_SHARED_PTR(SubDomain,dolfin::SubDomain)
SWIG_SHARED_PTR_DERIVED(DomainBoundary,dolfin::SubDomain,dolfin::DomainBoundary)

// This macro exposes the Variable interface for the derived classes
// This is a hack to get around the problem that Variable is not declared
// as a shared_ptr class.
%define IMPLEMENT_VARIABLE_INTERFACE(DERIVED_TYPE)
%extend dolfin::DERIVED_TYPE
{
  void rename(const std::string name, const std::string label)
  {
    self->rename(name,label);
  }

  const std::string& name()  const
  {
    return self->name();
  }

  const std::string& label() const
  {
    return self->label();
  }

  const std::string __str__(bool verbose=false) const
  {
    return self->str(verbose);
  }

}
%enddef

IMPLEMENT_VARIABLE_INTERFACE(Function)
IMPLEMENT_VARIABLE_INTERFACE(Mesh)

// FIXME: Make these const aware...
%define FOREIGN_SHARED_PTR_TYPEMAPS(TYPE)
// Define some dummy classes so SWIG becomes aware of these types
%inline %{
  class TYPE
  {
  };
%}

%typedef SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<TYPE> Shared ## TYPE;

%typecheck(0) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<TYPE> {
  int res = SWIG_ConvertPtr($input, 0, SWIGTYPE_p_ ## TYPE,0);
  $1 = SWIG_CheckState(res);
}

%typemap(in) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<TYPE> {
  void *argp = 0;
  TYPE * arg = 0;
  int res = SWIG_ConvertPtr($input, &argp, SWIGTYPE_p_ ## TYPE,0);
  if (SWIG_IsOK(res)) {
    arg = reinterpret_cast<TYPE *>(argp);
    $1 = dolfin::reference_to_no_delete_pointer(*arg);
  }
  else
    SWIG_exception(SWIG_TypeError, "expected an  ## TYPE");
}

%typemap(out) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr<TYPE> {
  TYPE * out = $1.get();
  $result = SWIG_NewPointerObj(SWIG_as_voidptr(out), SWIGTYPE_p_ ## TYPE, 0 |  0 );
}
%enddef


#endif
