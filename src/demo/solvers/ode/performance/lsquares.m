function [km] = lsquares(x, y)

 A = [x' ones(size(x'))];
 v = A \ y';
 km = v;

end
