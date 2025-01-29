function [t, theta_ADMM, k, kk_all_ADMM] = cADMM_2D(rho, alpha, opts, models, hyplength, ADMM_TOL)

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
INT_TOL  = 1e-3; % x-minimization tolerance
% initialization
M = opts.Ms;
for i=1:M
    theta_ADMM_iter(:,i,1) = [log(opts.ell);log(opts.el2);log(sqrt(opts.sf2));log(sqrt(opts.sn2))];
    beta_ADMM(:,i,1) = 1.5*ones(hyplength,1);
end
z_ADMM(:,1) = 1.5*ones(hyplength,1);
kk_all = 0;
for k = 1:MAX_ITER
    % z-update
    for i=1:M
        theta_beta_local(:,i) = theta_ADMM_iter(:,i,k) + (1/rho).* beta_ADMM(:,i,k);
    end
    z_ADMM(:,k+1) = (1/M) .* sum(theta_beta_local')';
    z_ADMM_new = z_ADMM(:,k+1);
    for i=1:M
        theta_local_old = theta_ADMM_iter(:,i,k);
        beta_ADMM_old = beta_ADMM(:,i,k);
        model = models{1,i};
        n_local = size(models{1,i}.Y, 1);
        % theta-update using gradient descent
        for kk = 1:MAX_ITER
            [nll, grad] = getNlmlGrad(theta_local_old, @mySEard, model);
            objgrad = grad + beta_ADMM_old + rho * (theta_local_old - z_ADMM_new);
            theta(:,kk) = theta_local_old - alpha * objgrad;
            theta_norm = norm(theta(:,kk) - theta_local_old);
            
            if and(theta_norm < INT_TOL, kk>1)
                theta_ADMM_iter(:,i,k+1) = theta(:,kk);
                theta = [];
                break;
            end
            theta_local_old = theta(:,kk);
        end
        % beta-update
        beta_ADMM(:,i,k+1) = beta_ADMM(:,i,k) + rho*(theta_ADMM_iter(:,i,k+1) - z_ADMM(:,k+1));
    end
    kk_all = kk_all + kk;
    ck = 0;
    for i=1:M
        constraint_norm(i) = norm (theta_ADMM_iter(:,i,k+1) - z_ADMM(:,k+1));
        if constraint_norm(i) < ADMM_TOL
            ck = ck+1;
        end
    end
    if ck == M
        theta_ADMM = theta_ADMM_iter(:,:,k+1);
        kk_all_ADMM = round(kk_all/M);
        break;
    end
end
t = toc(t_start);
end