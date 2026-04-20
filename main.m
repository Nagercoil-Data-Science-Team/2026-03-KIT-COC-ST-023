clc;
clear all;
close all;

fprintf('==========================================\n');
fprintf('   RANDOMIZED SKETCHED GMRES SOLVER\n');
fprintf('==========================================\n\n');

%% ================= STEP 2: Initialize Parameters =================
[n, l, k_max, tol] = init_parameters();

fprintf('--- STEP 2: Parameters Initialized ---\n');
fprintf('  Problem size        n     = %d\n', n);
fprintf('  Sketch dimension    l     = %d\n', l);
fprintf('  Max iterations      k_max = %d\n', k_max);
fprintf('  Tolerance           tol   = %e\n\n', tol);

%% ================= STEP 3: Generate System =================
[A, b, x] = generate_system(n);

fprintf('--- STEP 3: System Generated ---\n');
fprintf('  Matrix A size       : %d x %d\n', size(A,1), size(A,2));
fprintf('  Nonzeros in A       : %d\n', nnz(A));
fprintf('  Density of A        : %.4f%%\n', nnz(A)/(n*n)*100);
fprintf('  Norm of b           : %e\n', norm(b));
fprintf('  Initial x           : all zeros\n\n');

%% ================= STEP 4: Initial Residual =================
r = b - A*x;
beta = norm(r);

fprintf('--- STEP 4: Initial Residual ---\n');
fprintf('  Initial Residual Norm (beta) : %e\n', beta);
fprintf('  Min residual value           : %e\n', min(r));
fprintf('  Max residual value           : %e\n', max(r));
fprintf('  Mean residual value          : %e\n\n', mean(r));

figure;
plot(r, 'b');
xlabel('Index');
ylabel('Residual Value');
title('STEP 4: Initial Residual Vector r = b - Ax_0');
grid on;

%% ================= STEP 5: Krylov Subspace (Arnoldi Process) =================
fprintf('--- STEP 5: Arnoldi Process (Initial) ---\n');
V = zeros(n, k_max+1);
H = zeros(k_max+1, k_max);
V(:,1) = r / beta;

for j = 1:k_max
    w = A * V(:,j);
    for i = 1:j
        H(i,j) = V(:,i)' * w;
        w = w - H(i,j) * V(:,i);
    end
    H(j+1,j) = norm(w);
    if H(j+1,j) ~= 0
        V(:,j+1) = w / H(j+1,j);
    end
end

fprintf('  Krylov basis V built : %d x %d\n', size(V,1), size(V,2));
fprintf('  Hessenberg H built   : %d x %d\n', size(H,1), size(H,2));
fprintf('  Norm of V(:,1)       : %e\n', norm(V(:,1)));
fprintf('  Norm of V(:,2)       : %e\n', norm(V(:,2)));
fprintf('  H(1,1)               : %e\n', H(1,1));
fprintf('  H(2,1)               : %e\n\n', H(2,1));

figure;
plot(V(:,1));
xlabel('Index');
ylabel('Value');
title('STEP 5: First Krylov Basis Vector v_1');
grid on;

%% ================= STEP 6: Apply Randomized Sketching =================
fprintf('--- STEP 6: Randomized Sketching ---\n');
S = randn(l, k_max+1);

fprintf('  Sketch matrix S size : %d x %d\n', size(S,1), size(S,2));
fprintf('  Mean of S            : %e\n', mean(S(:)));
fprintf('  Std  of S            : %e\n', std(S(:)));
fprintf('  Min  of S            : %e\n', min(S(:)));
fprintf('  Max  of S            : %e\n\n', max(S(:)));

figure;
histogram(S(:), 30);
xlabel('Value');
ylabel('Frequency');
title('STEP 6: Sketch Matrix Distribution (S)');
grid on;

%% ================= STEP 7: Solve Reduced System =================
fprintf('--- STEP 7: Solving Reduced Sketched System ---\n');
e1 = zeros(k_max+1, 1);
e1(1) = beta;
SH   = S * H;
Sbe1 = S * e1;
y    = SH \ Sbe1;

fprintf('  Size of SH           : %d x %d\n', size(SH,1), size(SH,2));
fprintf('  Size of Sbe1         : %d x %d\n', size(Sbe1,1), size(Sbe1,2));
fprintf('  Size of y            : %d x 1\n', length(y));
fprintf('  Norm of y            : %e\n', norm(y));
fprintf('  Min  of y            : %e\n', min(y));
fprintf('  Max  of y            : %e\n\n', max(y));

figure;
plot(y, '-o');
xlabel('Index');
ylabel('Value');
title('STEP 7: Solution of Reduced System y');
grid on;

%% ================= STEP 8: Update Solution =================
x = x + V(:,1:k_max) * y;

fprintf('--- STEP 8: Solution Updated ---\n');
fprintf('  Norm of updated x    : %e\n', norm(x));
fprintf('  Min  of x            : %e\n', min(x));
fprintf('  Max  of x            : %e\n', max(x));
fprintf('  Mean of x            : %e\n\n', mean(x));

figure;
plot(x);
xlabel('Index');
ylabel('Value');
title('STEP 8: Updated Solution Vector x');
grid on;

%% ================= STEP 9-10: Iterative Loop =================
fprintf('--- STEP 9-10: Iterative Refinement Loop ---\n');
fprintf('----------------------------------------------------------\n');
fprintf('  %-10s %-20s %-20s\n', 'Iteration', 'Residual Norm', 'Reduction Ratio');
fprintf('----------------------------------------------------------\n');

residuals = [];
prev_res  = beta;
tic;

for iter = 1:k_max
    % Arnoldi Process
    r    = b - A*x;
    beta = norm(r);

    V = zeros(n, k_max+1);
    H = zeros(k_max+1, k_max);
    V(:,1) = r / beta;

    for j = 1:k_max
        w = A * V(:,j);
        for i = 1:j
            H(i,j) = V(:,i)' * w;
            w = w - H(i,j) * V(:,i);
        end
        H(j+1,j) = norm(w);
        if H(j+1,j) ~= 0
            V(:,j+1) = w / H(j+1,j);
        end
    end

    % Randomized Sketching
    S    = randn(l, k_max+1);
    SH   = S * H;
    e1   = zeros(k_max+1, 1);
    e1(1) = beta;
    Sbe1 = S * e1;

    % Solve sketched least squares
    y = SH \ Sbe1;

    % Update solution
    x = x + V(:,1:k_max) * y;

    % Compute residual
    res       = norm(b - A*x);
    ratio     = res / prev_res;
    prev_res  = res;
    residuals = [residuals res];

    fprintf('  %-10d %-20e %-20e\n', iter, res, ratio);

    % Check convergence
    if res < tol
        fprintf('\n  >>> Converged at iteration %d with residual %e <<<\n', iter, res);
        break;
    end
end

exec_time = toc;

fprintf('----------------------------------------------------------\n');
fprintf('\n--- STEP 10: Final Summary ---\n');
fprintf('  Total iterations run        : %d\n',   length(residuals));
fprintf('  Final residual norm         : %e\n',   residuals(end));
fprintf('  Initial residual norm       : %e\n',   residuals(1));
fprintf('  Total reduction             : %e\n',   residuals(end)/residuals(1));
fprintf('  Converged (res < tol)       : %s\n',   string(residuals(end) < tol));
fprintf('  Total Execution Time        : %.4f seconds\n\n', exec_time);

%% ================= STEP 11: Plot Convergence =================
fprintf('--- STEP 11: Convergence Plot Generated ---\n\n');
figure;
semilogy(1:length(residuals), residuals, '-o', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Residual Norm ||r_k||');
title('STEP 11: Residual Convergence Plot');
grid on;

fprintf('==========================================\n');
fprintf('         SOLVER COMPLETED\n');
fprintf('==========================================\n');

%% ==================== FUNCTIONS ====================
function [n, l, k_max, tol] = init_parameters()
    n     = 2500;
    l     = round(n/4);
    k_max = 50;
    tol   = 1e-6;
end

function [A, b, x] = generate_system(n)
    % Generate a more realistic sparse SPD-like system
    A = sprandn(n, n, 0.01);      % random sparse normal
    A = (A + A')/2;               % symmetrize
    A = A + 5*speye(n);           % moderate diagonal (less dominant than 50*I)
    b = randn(n,1);
    x = zeros(n,1);
end