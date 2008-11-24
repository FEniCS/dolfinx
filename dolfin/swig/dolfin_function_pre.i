// Rename misc function classes from Foo --> cpp_Foo (gets mapped in function.py)
%rename(cpp_Function) dolfin::Function;
%rename(cpp_FunctionSpace) dolfin::FunctionSpace;
%rename(cpp_FacetNormal) dolfin::FacetNormal;
%rename(cpp_MeshSize) dolfin::MeshSize;
%rename(cpp_AvgMeshSize) dolfin::AvgMeshSize;
%rename(cpp_FacetArea) dolfin::FacetArea;
%rename(cpp_InvFacetArea) dolfin::InvFacetArea;

// Modifying the interface of Function
%rename(sub) dolfin::Function::operator[];
%rename(assign) dolfin::Function::operator=;
%rename(_in) dolfin::Function::in;

// Modifying the interface of DiscreteFunction
%rename(sub)    dolfin::cpp_DiscreteFunction::operator[];
%rename(assign) dolfin::cpp_DiscreteFunction::operator=;
%rename(_in)    dolfin::cpp_DiscreteFunction::in;

// Ignore eval(val,data) function
%ignore dolfin::Function::eval(double* values, const Data& data) const;
