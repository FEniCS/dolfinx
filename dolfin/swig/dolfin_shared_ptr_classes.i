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
SWIG_SHARED_PTR_DERIVED(MeshSize,dolfin::Function,dolfin::MeshSize)
SWIG_SHARED_PTR_DERIVED(AvgMeshSize,dolfin::Function,dolfin::AvgMeshSize)
SWIG_SHARED_PTR_DERIVED(InvMeshSize,dolfin::Function,dolfin::InvMeshSize)
SWIG_SHARED_PTR_DERIVED(FacetNormal,dolfin::Function,dolfin::FacetNormal)
SWIG_SHARED_PTR_DERIVED(FacetArea,dolfin::Function,dolfin::FacetArea)
SWIG_SHARED_PTR_DERIVED(InvFacetArea,dolfin::Function,dolfin::InvFacetArea)
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
  
  const std::string __str__() const
  {
    return self->str();
  }

}
%enddef

IMPLEMENT_VARIABLE_INTERFACE(Function)
IMPLEMENT_VARIABLE_INTERFACE(Mesh)

#endif
