%--- Test problem 1 ---

primal_1
figure(1); clf

subplot(2,2,1)
plot(t',u')
axis([-0.5 10 -0.1 1.1])
xlabel('t')
ylabel('u')

subplot(2,2,2)
plot(t',u')
axis([-5e-4 1e-2 -0.1 1.1])
xlabel('t')
ylabel('u')

subplot(2,1,2)
semilogy(t',k')
axis([-0.2 10 1e-5 2])
grid on
xlabel('t')
ylabel('k')

disp('Press any key to continue')
pause

%--- Test problem 2 ---

primal_2
figure(1); clf

subplot(2,2,1)
plot(t',u')
axis([-0.5 10 -0.1 1.1])
xlabel('t')
ylabel('u')

subplot(2,2,2)
plot(t',u')
axis([-5e-3 0.1 -0.1 1.1])
xlabel('t')
ylabel('u')

subplot(2,1,2)
semilogy(t',k')
axis([-0.2 10 1e-5 2])
grid on
xlabel('t')
ylabel('k')

disp('Press any key to continue')
pause

%--- Test problem 3 ---

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

subplot(2,1,2)
semilogy(t',k')
axis([-0.02 1 1e-6 2])
grid on
xlabel('t')
ylabel('k')

disp('Press any key to continue')
pause

%--- Test problem 4 ---

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
