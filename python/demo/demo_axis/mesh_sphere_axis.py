import sys

try:
    import gmsh
except ModuleNotFoundError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)

from numpy import pi


def generate_mesh_sphere_axis(
        radius_sph: float, radius_scatt: float, radius_dom: float,
        radius_pml: float, in_sph_size: float, on_sph_size: float,
        scatt_size: float, pml_size: float, au_tag: int, bkg_tag: int,
        pml_tag: int, scatt_tag: int):

    gmsh.model.add("geometry")

    gmsh.model.occ.addCircle(
        0, 0, 0, radius_sph * 0.5, angle1=-pi / 2, angle2=pi / 2, tag=1)
    gmsh.model.occ.addCircle(
        0, 0, 0, radius_sph, angle1=-pi / 2, angle2=pi / 2, tag=2)
    gmsh.model.occ.addCircle(
        0, 0, 0, radius_scatt, angle1=-pi / 2, angle2=pi / 2, tag=3)
    gmsh.model.occ.addCircle(
        0, 0, 0, radius_dom, angle1=-pi / 2, angle2=pi / 2, tag=4)
    gmsh.model.occ.addCircle(
        0, 0, 0, radius_dom + radius_pml, angle1=-pi / 2, angle2=pi / 2,
        tag=5)

    gmsh.model.occ.addLine(10, 8, tag=6)
    gmsh.model.occ.addLine(8, 6, tag=7)
    gmsh.model.occ.addLine(6, 4, tag=8)
    gmsh.model.occ.addLine(4, 2, tag=9)
    gmsh.model.occ.addLine(2, 1, tag=10)
    gmsh.model.occ.addLine(1, 3, tag=11)
    gmsh.model.occ.addLine(3, 5, tag=12)
    gmsh.model.occ.addLine(5, 7, tag=13)
    gmsh.model.occ.addLine(7, 9, tag=14)

    gmsh.model.occ.addCurveLoop([10, 1], tag=1)
    gmsh.model.occ.addPlaneSurface([1], tag=1)
    gmsh.model.occ.addCurveLoop([11, 2, 9, -1], tag=2)
    gmsh.model.occ.addPlaneSurface([2], tag=2)
    gmsh.model.occ.addCurveLoop([8, -2, 12, 3], tag=3)
    gmsh.model.occ.addPlaneSurface([3], tag=3)
    gmsh.model.occ.addCurveLoop([13, 4, 7, -3], tag=4)
    gmsh.model.occ.addPlaneSurface([4], tag=4)
    gmsh.model.occ.addCurveLoop([4, -6, -5, -14], tag=5)
    gmsh.model.occ.addPlaneSurface([5], tag=5)

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [1, 2], tag=au_tag)
    gmsh.model.addPhysicalGroup(2, [3, 4], tag=bkg_tag)
    gmsh.model.addPhysicalGroup(2, [5], tag=pml_tag)
    gmsh.model.addPhysicalGroup(1, [3], tag=scatt_tag)

    gmsh.model.mesh.setSize([(0, 1)], size=in_sph_size)
    gmsh.model.mesh.setSize([(0, 2)], size=in_sph_size)
    gmsh.model.mesh.setSize([(0, 3)], size=on_sph_size)
    gmsh.model.mesh.setSize([(0, 4)], size=on_sph_size)
    gmsh.model.mesh.setSize([(0, 5)], size=scatt_size)
    gmsh.model.mesh.setSize([(0, 6)], size=scatt_size)
    gmsh.model.mesh.setSize([(0, 7)], size=pml_size)
    gmsh.model.mesh.setSize([(0, 8)], size=pml_size)
    gmsh.model.mesh.setSize([(0, 9)], size=pml_size)
    gmsh.model.mesh.setSize([(0, 10)], size=pml_size)

    gmsh.model.mesh.generate(2)

    return gmsh.model
