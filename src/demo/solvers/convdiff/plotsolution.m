% Plot solution using Octave

% Load solution
convdiff

% Plot the mesh
figure(1)
pdemesh(points, edges, cells)
title('Mesh')

% Plot with pdesurf
figure(2)
pdesurf(points, cells, u{6})
title('Convection around a hot dolphin')

% Plot contour lines
%figure(3); clf
%pdeplot(points,edges,cells,'contour', 'on', 'xydata', u{size(u,2)})
%title('Convection around a hot dolphin')
