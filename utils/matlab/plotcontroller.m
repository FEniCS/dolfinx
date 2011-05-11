% Copyright (C) 2005 Anders Logg
%
% This file is part of DOLFIN.
%
% DOLFIN is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% DOLFIN is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Lesser General Public License for more details.
%
% You should have received a copy of the GNU Lesser General Public License
% along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
% 
% Simple script for evaluating the performance of
% different time step controllers.

clf

subplot(2,1,1)
plot(t, k)

subplot(2,1,2)
e = k.*abs(r);
semilogy(t, e);
hold on
plot(t, 2.0*tol*ones(size(t)))
plot(t, 1.0*tol*ones(size(t)))
plot(t, 0.5*tol*ones(size(t)))
grid on
xlabel('t')
ylabel('e')
