% Load solution
convdiff

% Plot solution at final time
pdeplot(points,edges,cells,'contour', 'on', 'xydata', u{size(u,2)})
title('Convection around a hot dolphin')
