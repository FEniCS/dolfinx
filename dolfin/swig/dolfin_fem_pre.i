// Rename of swig proxy classes
// This is done so they can be distinguished from python versions
// of the same classes defined in site-packages/dolfin/*

%rename(cpp_assemble) dolfin::assemble;
%rename(cpp_assemble_system) dolfin::assemble_system;
