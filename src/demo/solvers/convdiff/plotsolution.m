% Load solution
convdiff

% Plot contour lines
figure(1); clf
pdeplot(points,edges,cells,'contour', 'on', 'xydata', u{size(u,2)})
title('Convection around a hot dolphin')

% Plot with pdesurf
figure(2); clf
pdesurf(points, cells, u{size(u,2)})
shading faceted
title('Convection around a hot dolphin')
