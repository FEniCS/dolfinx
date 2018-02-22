// Gmsh input file for unit radius circle with mesh symmetry line on y-axis
cl__1 = 0.05;
Point(1) = {0, 1, 0, cl__1};
Point(2) = {1, 0, 0, cl__1};
Point(3) = {0, -1, 0, cl__1};
Point(4) = {-1, 0, 0, cl__1};
Point(6) = {0, 0, 0, cl__1};
Circle(1) = {1, 6, 4};
Circle(2) = {4, 6, 3};
Circle(3) = {3, 6, 2};
Circle(4) = {2, 6, 1};
Line Loop(7) = {1, 2, 3, 4};
Plane Surface(7) = {7};
Line(8) = {1, 3};
Line{8} In Surface {7};