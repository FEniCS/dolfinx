"""Central parameters for the DOLFINx PML demo.

Defining constants so that main script reads it easily.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PhysicalConstants:
    epsilon_0: float = 8.8541878128e-12
    mu_0: float = 4 * np.pi * 1e-7


@dataclass(frozen=True)
class Tags:
    au: int = 1
    bkg: int = 2
    scatt: int = 3
    pml: int = 4


@dataclass(frozen=True)
class MeshParameters:
    # Radius of the wire and of the boundary of the domain
    radius_wire: float = 0.05
    l_dom: float = 0.8
    l_pml: float = 1.0
    mesh_factor: float = 1.0

    @property
    def radius_scatt(self) -> float:
        return 0.8 * self.l_dom / 2

    @property
    def in_wire_size(self) -> float: # Mesh size inside the wire
        return self.mesh_factor * 6e-3

    @property
    def on_wire_size(self) -> float: # Mesh size at the boundary of the wire
        return self.mesh_factor * 3.0e-3

    @property
    def scatt_size(self) -> float: # Mesh size in the background
        return self.mesh_factor * 15.0e-3

    @property
    def pml_size(self) -> float: # Mesh size at the boundary
        return self.mesh_factor * 15.0e-3


@dataclass(frozen=True)
class ProblemParameters:
    wl0: float = 0.4 # Wavelength of the background field
    n_bkg: float = 1.0 # Background refractive index
    theta: float = 0.0 # Angle of incidence of the background field
    degree: int = 3
    alpha: float = 1.0
    eps_au: complex = -1.0782 + 1j * 5.8089 # Definition of relative permittivity for Au @400nm

    @property
    def eps_bkg(self) -> float:
        return self.n_bkg**2

    @property
    def k0(self) -> float:
        return 2 * np.pi / self.wl0
