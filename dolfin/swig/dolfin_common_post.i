%template(STLVectorBoundaryConditionPtr) std::vector<dolfin::BoundaryCondition *>;
%template(STLVectorFunctionPtr) std::vector<dolfin::Function *>;
%template(STLVectorFunctionSpacePtr) std::vector<dolfin::FunctionSpace *>;
%template(STLVectorUInt) std::vector<dolfin::uint>;
%template(STLVectorDouble) std::vector<double>;

%extend dolfin::Variable
{
  const std::string __str__() const
  {
    return self->str();
  }
   
}
