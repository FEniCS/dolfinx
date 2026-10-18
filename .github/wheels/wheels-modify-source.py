"""Stamp dev versions, pin the stack and inject the MPI check for the wheels."""

import argparse
import pathlib

import tomlkit
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dev-stamp", required=True, help="replaces 0 in each .dev0 version")
parser.add_argument("--build-stack", action="store_true", help="also prepare UFL, FFCx and Basix")
args = parser.parse_args()

paths = {"fenics-dolfinx": pathlib.Path("dolfinx/python/pyproject.toml")}
if args.build_stack:
    paths |= {
        name: pathlib.Path(name.removeprefix("fenics-"), "pyproject.toml")
        for name in ("fenics-ufl", "fenics-ffcx", "fenics-basix")
    }
projects = {name: tomlkit.parse(path.read_text(encoding="utf-8")) for name, path in paths.items()}

versions = {}
for name, pyproject in projects.items():
    version = Version(pyproject["project"]["version"])
    if version.dev != 0 or version.local:
        raise RuntimeError(f"{paths[name]} has version {version}, not a .dev0 version")
    versions[name] = Version(f"{str(version).removesuffix('.dev0')}.dev{args.dev_stamp}")
    pyproject["project"]["version"] = str(versions[name])
    print(f"{name} {versions[name]}")


# DOLFINx links the Basix it was built with, and PyPI rejects direct URL
# dependencies, so stack dependencies are pinned exactly.
def _pin(requirements, project):
    for i, requirement in enumerate(requirements):
        requirement = Requirement(requirement)
        name = canonicalize_name(requirement.name)
        if name not in versions or name == project:
            continue
        if requirement.url is None and not requirement.specifier.contains(
            versions[name], prereleases=True
        ):
            raise RuntimeError(f"{name} {versions[name]} does not satisfy {requirement}")
        requirement.url = None
        requirement.specifier = SpecifierSet(f"=={versions[name]}")
        requirements[i] = str(requirement)


for name, pyproject in projects.items():
    _pin(pyproject["project"].get("dependencies", []), name)
    for requirements in pyproject["project"].get("optional-dependencies", {}).values():
        _pin(requirements, name)
    paths[name].write_text(tomlkit.dumps(pyproject), encoding="utf-8")

init = pathlib.Path("dolfinx/python/dolfinx/__init__.py")
check = pathlib.Path("dolfinx/.github/wheels/wheels_mpi_check.py.in").read_text(encoding="utf-8")
text = init.read_text(encoding="utf-8")
if text.count("# WHEELMPICHECK\n") != 1:
    raise RuntimeError(f"{init} has no WHEELMPICHECK placeholder")
init.write_text(text.replace("# WHEELMPICHECK\n", check), encoding="utf-8")
