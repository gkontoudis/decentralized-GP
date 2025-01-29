function [t, theta_ADMM, k, kk_all_ADMM] = ...
    dec_cADMM_2D(rho, alpha, opts, models, hyplength, max_iter_dec_cADMM)

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
    p_ADMM(:,i,1) = zeros(hyplength,1);
end
kk_all = 0;
for k = 1:max_iter_dec_cADMM
    % sum_xi_xj for a path graph topology
    sum_xi_xj(:,1,k) = theta_ADMM_iter(:,1,k) - theta_ADMM_iter(:,2,k);
    for j=2:M-1
        sum_xi_xj(:,j,k) = 2*theta_ADMM_iter(:,j,k) - theta_ADMM_iter(:,j-1,k)...
            - theta_ADMM_iter(:,j+1,k);
    end
    sum_xi_xj(:,M,k) = theta_ADMM_iter(:,M,k) - theta_ADMM_iter(:,M-1,k);
    
    for i=1:M
        theta_local_old = theta_ADMM_iter(:,i,k);
        model = models{1,i};
        n_local = size(models{1,i}.Y, 1);
        % p-update
        p_ADMM(:,i,k+1) = p_ADMM(:,i,k) + rho*sum_xi_xj(:,i,k);
        for kk = 1:MAX_ITER
            % sum_x_xi_xj
            if i == 1
                sum_x_xi_xj(:,1,k) = theta_local_old ...
                    - .5*(theta_ADMM_iter(:,1,k) + theta_ADMM_iter(:,2,k) );
            elseif and(i>1, i<M)
                for j=2:M-1
                    sum_x_xi_xj(:,j,k) = 2*theta_local_old ...
                        - theta_ADMM_iter(:,j,k) ...
                        -.5 * ( theta_ADMM_iter(:,j-1,k) + theta_ADMM_iter(:,j+1,k) );
                end
            else
                sum_x_xi_xj(:,M,k) = theta_local_old ...
                    - .5*(theta_ADMM_iter(:,M,k) + theta_ADMM_iter(:,M-1,k) );
            end
            % theta-update using gradient descent
            [nll, grad] = getNlmlGrad(theta_local_old, @mySEard, model);
            objgrad = grad + p_ADMM(:,i,k+1) + 2*rho*sum_x_xi_xj(:,i,k);
            theta(:,kk) = theta_local_old - alpha * objgrad;
            theta_norm = norm(theta(:,kk) - theta_local_old);
            if and(theta_norm < INT_TOL, kk>1)
                theta_ADMM_iter(:,i,k+1) = theta(:,kk);
                theta = [];
                break;
            end
            theta_local_old = theta(:,kk);
        end
    end
    kk_all = kk_all + kk;
end
theta_ADMM = theta_ADMM_iter(:,:,k+1);
kk_all_ADMM = round(kk_all/M);
t = toc(t_start);
end