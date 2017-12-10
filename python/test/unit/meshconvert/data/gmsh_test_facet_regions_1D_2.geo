Point(1) = {0, 0, 0, 0.1};
Extrude {0.5, 0, 0} {
  Point{1}; Layers{5};
}
Extrude {0.5, 0, 0} {
  Point{2}; Layers{5};
}
Physical Point(999) = {1,3};
Physical Line(1) = {1};
Physical Line(2) = {2};
