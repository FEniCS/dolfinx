%template(STLVectorBoundaryConditionPtr) std::vector<dolfin::BoundaryCondition *>;

%template(STLVectorFunctionPtr) std::vector<dolfin::Function *>;
%template(ArrayFunctionPtr) dolfin::Array<dolfin::Function *>;

%template(STLVectorFunctionSpacePtr) std::vector<dolfin::FunctionSpace *>;
%template(ArrayFunctionSpacePtr) dolfin::Array<dolfin::FunctionSpace *>;

%template(STLVectorUInt) std::vector<dolfin::uint>;
%template(ArrayUInt) dolfin::Array<dolfin::uint>;

%template(STLVectorDouble) std::vector<double>;
%template(ArrayDouble) dolfin::Array<double>;

