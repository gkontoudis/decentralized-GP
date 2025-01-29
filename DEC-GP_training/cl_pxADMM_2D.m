function [t, theta_ADMM, k] = cl_pxADMM_2D(rho, opts, models, hyplength, Lip, ADMM_TOL)

% Distributed GP via classical ADMM, using gradient descent to solve
% sub-problem
%
% rho: ADMM penalty parameter
% alpha: step size for gradient descent
%
% M: number of local machines (data subsets)
% K: number of training data points on each local machine
% inihyp = [ log(ell_1)
%            log(ell_2)
%             .
%            log(ell_D)
%            log(sf) ]
%            log(sn) ]         , initial guess of hyperparameters,
% where ell_1^2,...,ell_D^2 are ARD parameters, sf^2 is the signal
% variance, and sn^2 is the noise variance.

t_start = tic;
% ADMM parameters
MAX_ITER = 1e6;

% initialization
M = opts.Ms;
for i=1:M
    theta_ADMM_iter(:,i,1) = [log(opts.ell);log(opts.el2);log(sqrt(opts.sf2));log(sqrt(opts.sn2))];
    beta_ADMM(:,i,1) = 1.5*ones(hyplength,1);
end
z_ADMM(:,1) = 1.5*ones(hyplength,1);
for k = 1:MAX_ITER
    k;
    % z-update
    for i=1:M
        theta_beta_local(:,i) = theta_ADMM_iter(:,i,k) + (1/rho).* beta_ADMM(:,i,k);
    end
    z_ADMM(:,k+1) = (1/M) .* sum(theta_beta_local')';
    z_ADMM_new = z_ADMM(:,k+1);
    for i=1:M
        beta_ADMM_old = beta_ADMM(:,i,k);
        model = models{1,i};
        % theta-update
        [nll, grad] = getNlmlGrad(z_ADMM_new, @mySEard, model);
        theta_ADMM_iter(:,i,k+1) = z_ADMM_new - ...
            (1/(rho+Lip))*(grad + beta_ADMM_old);
        % beta-update
        beta_ADMM(:,i,k+1) = beta_ADMM(:,i,k) + rho*(theta_ADMM_iter(:,i,k+1) - z_ADMM(:,k+1));
    end
    ck = 0;
    for i=1:M
        constraint_norm(i) = norm (theta_ADMM_iter(:,i,k+1) - z_ADMM(:,k+1));
        if constraint_norm(i) < ADMM_TOL
            ck = ck+1;
        end
    end
    if ck == M
        theta_ADMM = theta_ADMM_iter(:,:,k+1);
        break;
    end
end
t = toc(t_start);
end