%template(STLVectorDirichletBCPtr) std::vector<dolfin::DirichletBC *>;
%template(STLVectorBoundaryConditionPtr) std::vector<dolfin::BoundaryCondition *>;
%template(STLVectorFunctionPtr) std::vector<dolfin::Function *>;
%template(STLVectorFunctionSpacePtr) std::vector<dolfin::FunctionSpace *>;
%template(STLVectorUInt) std::vector<dolfin::uint>;
%template(STLVectorDouble) std::vector<double>;
%template(STLVectorString) std::vector<std::string>;

%extend dolfin::Variable
{
  const std::string __str__(bool verbose=false) const
  {
    return self->str(verbose);
  }

}
