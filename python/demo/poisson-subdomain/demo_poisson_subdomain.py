#
# .. _demo_pde_subdomains_poisson_documentation:
# 
# Poisson equation with multiple subdomains
# =========================================
# 
# This demo is implemented in a single Python file,
# :download:`demo_subdomains-poisson.py`, which contains both the
# variational forms and the solver. We suggest that you familiarize
# yourself with the :ref:`Poisson demo
# <demo_pde_poisson_python_documentation>` before studying this example,
# as some of the more standard steps will be described in less detail.
# 
# 
# The main purpose of this demo is to demonstrate how to create and
# integrate variational forms over distinct regions of a domain and/or
# its boundaries.
# 
# Equation and problem definition
# -------------------------------
# 
# For illustration purposes, we consider a weighted Poisson equation over
# a unit square: :math:`\Omega = [0,1] \times [0,1]` with mixed boundary
# conditions: find :math:`u` satisfying
# 
# .. math::
#    - \mathrm{div} (a \nabla u) &= 1.0 \quad {\rm in} \ \Omega, \\
#                   u &= 5.0 \quad {\rm on} \ \Gamma_{T}, \\
#                   u &= 0.0 \quad {\rm on} \ \Gamma_{B}, \\
#                   \nabla u \cdot n &= - 10.0 \, e^{-(y - 0.5)^2} \quad {\rm on} \ \Gamma_{L}. \\
#                   \nabla u \cdot n &= 1.0 \quad {\rm on} \ \Gamma_{R}, \\
# 
# where :math:`\Gamma_{T}`, :math:`\Gamma_{B}`, :math:`\Gamma_{L}`,
# :math:`\Gamma_{R}` denote the top, bottom, left and right sides of the
# unit square, respectively. The coefficient :math:`a` may vary over the
# domain: here, we let :math:`a = a_1 = 0.01` for :math:`(x, y) \in
# \Omega_1 = [0.5, 0.7] \times [0.2, 1.0]` and :math:`a = a_0 = 1.0` in
# :math:`\Omega_0 = \Omega \backslash \Omega_1`. We can think of
# :math:`\Omega_1` as an obstacle with differing material properties
# from the rest of the domain.
# 
# 
# Variational form
# -------------------------------
# 
# We can write the above boundary value problem in the standard linear
# variational form: find :math:`u \in V` such that
# 
# .. math::
# 
#    a(u, v) = L(v) \quad \forall \ v \in \hat{V},
# 
# where :math:`V` and :math:`\hat{V}` are suitable function spaces
# incorporating the Dirichlet boundary conditions on :math:`\Gamma_{T}`
# and :math:`\Gamma_{B}`, and
# 
# .. math::
# 
#    a(u, v) &= \int_{\Omega_0} a_0 \nabla u \cdot \nabla v \, {\rm d} x
#             + \int_{\Omega_1} a_1 \nabla u \cdot \nabla v \, {\rm d} x, \\
#    L(v)      &=  \int_{\Gamma_{L}} g_L v \, {\rm d} s
#             + \int_{\Gamma_{R}} g_R v \, {\rm d} s
#             + \int_{\Omega} f \, v \, {\rm d} x.
# 
# where :math:`f = 1.0`, :math:`g_L = - 10.0 e^{-(y - 0.5)^2}` and
# :math:`g_R = 1.0`.
