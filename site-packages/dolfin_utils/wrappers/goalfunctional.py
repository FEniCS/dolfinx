# Copyright (C) 2010 Marie E. Rognes
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Last changed: 2011-07-06

__all__ = ["generate_update_ec"]

#-------------------------------------------------------------------------------
attach_coefficient_template = """
    // Attach coefficients from %(from)s to %(to)s
    for (std::size_t i = 0; i < %(from)s.num_coefficients(); i++)
    {
      name = %(from)s.coefficient_name(i);
      // Don't attach discrete primal solution here (not computed).
      if (name == "__discrete_primal_solution")
        continue;

      // Test whether %(to)s has coefficient named 'name'
      try {
        %(to)s->coefficient_number(name);
      } catch (...) {
        std::cout << "Attaching coefficient named: " << name << " to %(to)s";
        std::cout << " failed! But this might be expected." << std::endl;
        continue;
      }
      %(to)s->set_coefficient(name, %(from)s.coefficient(i));
    }
    """
#-------------------------------------------------------------------------------
attach_domains_template = """
    // Attach subdomains from %(from)s to %(to)s
    %(to)s->dx = %(from)s.cell_domains();
    %(to)s->ds = %(from)s.exterior_facet_domains();
    %(to)s->dS = %(from)s.interior_facet_domains();
"""
#-------------------------------------------------------------------------------
update_ec_template = """
  /// Initialize all error control forms, attach coefficients and
  /// (re-)set error control
  virtual void update_ec(const dolfin::Form& a, const dolfin::Form& L)
  {
    // This stuff is created here and shipped elsewhere
    boost::shared_ptr<dolfin::Form> a_star;           // Dual lhs
    boost::shared_ptr<dolfin::Form> L_star;           // Dual rhs
    boost::shared_ptr<dolfin::FunctionSpace> V_Ez_h;  // Extrapolation space
    boost::shared_ptr<dolfin::Function> Ez_h;         // Extrapolated dual
    boost::shared_ptr<dolfin::Form> residual;         // Residual (as functional)
    boost::shared_ptr<dolfin::FunctionSpace> V_R_T;   // Trial space for cell residual
    boost::shared_ptr<dolfin::Form> a_R_T;            // Cell residual lhs
    boost::shared_ptr<dolfin::Form> L_R_T;            // Cell residual rhs
    boost::shared_ptr<dolfin::FunctionSpace> V_b_T;   // Function space for cell bubble
    boost::shared_ptr<dolfin::Function> b_T;          // Cell bubble
    boost::shared_ptr<dolfin::FunctionSpace> V_R_dT;  // Trial space for facet residual
    boost::shared_ptr<dolfin::Form> a_R_dT;           // Facet residual lhs
    boost::shared_ptr<dolfin::Form> L_R_dT;           // Facet residual rhs
    boost::shared_ptr<dolfin::FunctionSpace> V_b_e;   // Function space for cell cone
    boost::shared_ptr<dolfin::Function> b_e;          // Cell cone
    boost::shared_ptr<dolfin::FunctionSpace> V_eta_T; // Function space for indicators
    boost::shared_ptr<dolfin::Form> eta_T;            // Indicator form

    // Some handy views
    const dolfin::FunctionSpace& Vhat(*(a.function_space(0))); // Primal test
    const dolfin::FunctionSpace& V(*(a.function_space(1)));    // Primal trial
    assert(V.mesh());
    const dolfin::Mesh& mesh(*V.mesh());
    std::string name;

    // Initialize dual forms
    a_star.reset(new %(a_star)s(V, Vhat));
    L_star.reset(new %(L_star)s(V));

    %(attach_a_star)s
    %(attach_L_star)s

    // Initialize residual
    residual.reset(new %(residual)s(mesh));
    %(attach_residual)s

    // Initialize extrapolation space and (fake) extrapolation
    V_Ez_h.reset(new %(V_Ez_h)s(mesh));
    Ez_h.reset(new dolfin::Function(V_Ez_h));
    residual->set_coefficient("__improved_dual", Ez_h);

    // Create bilinear and linear form for computing cell residual R_T
    V_R_T.reset(new %(V_R_T)s(mesh));
    a_R_T.reset(new %(a_R_T)s(V_R_T, V_R_T));
    L_R_T.reset(new %(L_R_T)s(V_R_T));

    // Initialize bubble and attach to a_R_T and L_R_T
    V_b_T.reset(new %(V_b_T)s(mesh));
    b_T.reset(new dolfin::Function(V_b_T));
    *b_T->vector() = 1.0;
    %(attach_L_R_T)s

    // Attach bubble function to _a_R_T and _L_R_T
    a_R_T->set_coefficient("__cell_bubble", b_T);
    L_R_T->set_coefficient("__cell_bubble", b_T);

    // Create bilinear and linear form for computing facet residual R_dT
    V_R_dT.reset(new %(V_R_dT)s(mesh));
    a_R_dT.reset(new %(a_R_dT)s(V_R_dT, V_R_dT));
    L_R_dT.reset(new %(L_R_dT)s(V_R_dT));
    %(attach_L_R_dT)s

    // Initialize (fake) cone and attach to a_R_dT and L_R_dT
    V_b_e.reset(new %(V_b_e)s(mesh));
    b_e.reset(new dolfin::Function(V_b_e));
    a_R_dT->set_coefficient("__cell_cone", b_e);
    L_R_dT->set_coefficient("__cell_cone", b_e);

    // Create error indicator form
    V_eta_T.reset(new %(V_eta_T)s(mesh));
    eta_T.reset(new %(eta_T)s(V_eta_T));

    // Update error control
    _ec.reset(new dolfin::ErrorControl(a_star, L_star, residual,
                                       a_R_T, L_R_T, a_R_dT, L_R_dT, eta_T,
                                       %(linear)s));

  }
"""
#-------------------------------------------------------------------------------
def _attach(tos, froms):

    if not isinstance(froms, tuple):
        return attach_coefficient_template % {"to": tos, "from": froms} \
            + attach_domains_template % {"to": tos, "from": froms}

    # NB: If multiple forms involved, attach domains from last form.
    coeffs = "\n".join([attach_coefficient_template % {"to": to, "from": fro}
                        for (to, fro) in zip(tos, froms)])
    domains = attach_domains_template % {"to": tos[-1], "from": froms[-1]}
    return coeffs + domains

#-------------------------------------------------------------------------------
def generate_maps(linear):
    """
    NB: This depends on the ordering of the forms
    """
    maps = {"a_star":           "Form_%d" % 0,
            "L_star":           "Form_%d" % 1,
            "residual":         "Form_%d" % 2,
            "a_R_T":            "Form_%d" % 3,
            "L_R_T":            "Form_%d" % 4,
            "a_R_dT":           "Form_%d" % 5,
            "L_R_dT":           "Form_%d" % 6,
            "eta_T":            "Form_%d" % 7,
            "V_Ez_h":           "CoefficientSpace_%s" % "__improved_dual",
            "V_R_T":            "Form_%d::TestSpace" % 4,
            "V_b_T":            "CoefficientSpace_%s" % "__cell_bubble",
            "V_R_dT":           "Form_%d::TestSpace" % 6,
            "V_b_e":            "CoefficientSpace_%s" % "__cell_cone",
            "V_eta_T":          "Form_%d::TestSpace" % 7,
            "attach_a_star":    _attach("a_star", "a"),
            "attach_L_star":    _attach("L_star", "(*this)"),
            "attach_residual":  _attach(("residual",)*2, ("a", "L")),
            "attach_L_R_T":     _attach(("L_R_T",)*2, ("a", "L")),
            "attach_L_R_dT":    _attach(("L_R_dT",)*2, ("a", "L")),
            "linear":           "true" if linear else "false"
            }
    return maps
#-------------------------------------------------------------------------------
def generate_update_ec(form):

    linear = "__discrete_primal_solution" in form.coefficient_names
    maps = generate_maps(linear)
    code = update_ec_template % maps
    return code
