//%template(STLVectorDirichletBCPtr) std::vector<dolfin::DirichletBC *>;
//%template(STLVectorBoundaryConditionPtr) std::vector<dolfin::BoundaryCondition *>;
//%template(STLVectorFunctionPtr) std::vector<dolfin::Function *>;
//%template(STLVectorFunctionSpacePtr) std::vector<dolfin::FunctionSpace *>;
//%template(STLVectorUInt) std::vector<dolfin::uint>;
//%template(STLVectorDouble) std::vector<double>;
//%template(STLVectorString) std::vector<std::string>;
//%template(STLPairUInt) std::pair<dolfin::uint,dolfin::uint>;

/*%template(BOOSTUnorderSetUInt) boost::unordered_set<dolfin::uint>;*/

//%template(ConstDoubleArray) dolfin::Array<const double>;


%extend dolfin::Array {
  T sub(unsigned int i) const { return (*self)[i]; }

  T __getitem__(unsigned int i) const { return (*self)[i]; }
  void __setitem__(unsigned int i, const T& val) { (*self)[i] = val; }
}

%ignore dolfin::Array<const double>::__setitem__;

%template(DoubleArray) dolfin::Array<double>;
%template(ConstDoubleArray) dolfin::Array<const double>;


%extend dolfin::Variable
{
  std::string __str__() const
  {
    return self->str(false);
  }
}
