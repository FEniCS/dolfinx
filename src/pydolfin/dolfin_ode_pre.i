# FIXME: These renames don't seem to work.
%rename(fmono) dolfin::ODE::f(const dolfin::uBlasVector&, dolfin::real, dolfin::uBlasVector&);
%rename(fmulti) dolfin::ODE::f(const dolfin::uBlasVector&, dolfin::real, dolfin::uint);
