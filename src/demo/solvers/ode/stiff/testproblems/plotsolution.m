% Copyright (C) 2004 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% Plot solution and time steps for the stiff test problems.

%--- Test problem 1 ---

disp('Plotting solution and time steps for test problem 1')

primal_1
figure(1); clf

subplot(2,2,1)
plot(t',u')
axis([-0.5 10 -0.05 1.05])
xlabel('t')
ylabel('u')

subplot(2,2,2)
plot(t',u')
axis([-5e-4 1e-2 -0.05 1.05])
xlabel('t')
ylabel('u')

subplot(2,2,3)
semilogy(t',k')
axis([-0.5 10 1e-5 2])
grid on
xlabel('t')
ylabel('k')

subplot(2,2,4)
semilogy(t',k')
axis([-5e-4 1e-2 1e-5 2])
grid on
xlabel('t')
ylabel('k')

print -depsc solution_1.eps

disp('Press any key to continue')
pause

%--- Test problem 2 ---

disp('Plotting solution and time steps for test problem 2')

primal_2
figure(1); clf

subplot(2,2,1)
plot(t',u')
axis([-0.5 10 -0.05 1.05])
xlabel('t')
ylabel('u')

subplot(2,2,2)
plot(t',u')
axis([-5e-3 0.1 -0.05 1.05])
xlabel('t')
ylabel('u')

subplot(2,2,3)
semilogy(t',k')
axis([-0.5 10 1e-5 2])
grid on
xlabel('t')
ylabel('k')

subplot(2,2,4)
semilogy(t',k')
axis([-5e-3 0.1 1e-5 2])
grid on
xlabel('t')
ylabel('k')

print -depsc solution_2.eps

disp('Press any key to continue')
pause

%--- Test problem 3 ---

disp('Plotting solution and time steps for test problem 3')

primal_3
figure(1); clf

subplot(2,2,1)
plot(t',u')
axis([-0.05 1 -37 3])
xlabel('t')
ylabel('u')

subplot(2,2,2)
plot(t',u')
axis([-5e-3 0.1 -37 3])
xlabel('t')
ylabel('u')

subplot(2,2,3)
semilogy(t',k')
axis([-0.02 1 1e-6 2])
grid on
xlabel('t')
ylabel('k')

subplot(2,2,4)
semilogy(t',k')
axis([-5e-3 0.1 1e-6 2])
grid on
xlabel('t')
ylabel('k')

print -depsc solution_3.eps

disp('Press any key to continue')
pause

%--- Test problem 4 ---

disp('Plotting solution and time steps for test problem 4')

primal_4
figure(1); clf

subplot(2,2,1)
plot(t',u')
axis([-10 321.8122 -0.05 1.05])
xlabel('t')
ylabel('u')

subplot(2,2,2)
plot(t',u')
axis([-2e-1 5 -0.05 1.05])
xlabel('t')
ylabel('u')

subplot(2,2,3)
semilogy(t',k')
axis([-10 321.8122 1e-4 2])
grid on
xlabel('t')
ylabel('k')

subplot(2,2,4)
semilogy(t',k')
axis([-2e-1 5 1e-4 2])
grid on
xlabel('t')
ylabel('k')

print -depsc solution_4.eps

disp('Press any key to continue')
pause

%--- Test problem 5 ---

disp('Plotting solution and time steps for test problem 5')

primal_5
figure(1); clf

subplot(2,2,1)
plot(t',u')
axis([-5 180 -0.02 0.5])
xlabel('t')
ylabel('u')

subplot(2,2,2)
plot(t',u')
axis([-2e-1 5 -5e-5 1e-3])
xlabel('t')
ylabel('u')

subplot(2,2,3)
semilogy(t',k','b')
axis([-5 180 5e-3 2])
grid on
xlabel('t')
ylabel('k')

subplot(2,2,4)
semilogy(t',k','b')
axis([-2e-1 5 5e-3 2])
grid on
xlabel('t')
ylabel('k')

print -depsc solution_5.eps

disp('Press any key to continue')
pause

%--- Test problem 6 ---

disp('Plotting solution and time steps for test problem 6')

primal_6
figure(1); clf

subplot(2,2,1)
plot(t',u')
axis([0 100 -15 15])
xlabel('t')
ylabel('u')

subplot(2,2,2)
plot(t',u')
axis([38 48 -15 15])
xlabel('t')
ylabel('u')

subplot(2,2,3)
semilogy(t',k')
axis([0 100 1e-10 2])
grid on
xlabel('t')
ylabel('k')

subplot(2,2,4)
plot(t',k')
axis([38 48 -0.05 1.05])
grid on
xlabel('t')
ylabel('k')

print -depsc solution_6.eps

disp('Press any key to continue')
pause

%--- Test problem 7 ---

disp('Plotting solution and time steps for test problem 7')

primal_7
figure(1); clf

subplot(2,1,1)
plot(t',u')
axis([-0.01 1 -0.01 0.3])
xlabel('t')
ylabel('u')

subplot(2,1,2)
semilogy(t',k')
axis([-0.01 1 5e-6 2])
grid on
xlabel('t')
ylabel('k')

print -depsc solution_7.eps

disp('Press any key to continue')
pause

%--- Test problem 8 ---

disp('Plotting solution and time steps for test problem 8')

primal_8
figure(1); clf

subplot(4,2,1)
plot(t,u(1,:))
axis([-0.01 0.3 0.988 1.001])
xlabel('t')
ylabel('u1')

subplot(4,2,3)
plot(t,u(3,:))
axis([-0.01 0.3 -0.001 0.012])
xlabel('t')
ylabel('u3')

subplot(2,2,2)
plot(t,u(2,:))
axis([-0.01 0.3 -1e-6 4e-5])
xlabel('t')
ylabel('u2')

subplot(2,1,2)
semilogy(t',k')
axis([-0.005 0.3 1e-3 1])
grid on
xlabel('t')
ylabel('k')

print -depsc solution_8.eps

disp('Press any key to continue')
pause

%--- Test problem 9 ---

disp('Plotting solution and time steps for test problem 9')

primal_9
figure(1); clf

subplot(2,2,1)
plot(t',u')
axis([-0.5 30 -1.05 1.05])
xlabel('t')
ylabel('u')

subplot(2,2,2)
plot(t',u')
axis([-0.005 0.3 -1.05 1.05])
xlabel('t')
ylabel('u')

subplot(2,1,2)
plot(t',k')
axis([-0.25 30 -0.05 1.05])
grid on
xlabel('t')
ylabel('k')

print -depsc solution_9.eps
