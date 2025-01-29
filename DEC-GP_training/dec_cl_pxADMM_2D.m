function [t, theta_ADMM, k] = ...
    dec_cl_pxADMM_2D(rho, opts, models, hyplength, Lip, max_iter_dec_cl_pxADMM)

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
% initialization
M = opts.Ms;
for i=1:M
    theta_ADMM_iter(:,i,1) = [log(opts.ell);log(opts.el2);log(sqrt(opts.sf2));log(sqrt(opts.sn2))];
    p_ADMM(:,i,1) = zeros(hyplength,1);
end
for k = 1:max_iter_dec_cl_pxADMM
    % sum_xi_xj for a path graph topology
    sum_xi_xj(:,1,k) = theta_ADMM_iter(:,1,k) - theta_ADMM_iter(:,2,k);
    for j=2:M-1
        sum_xi_xj(:,j,k) = 2*theta_ADMM_iter(:,j,k) - theta_ADMM_iter(:,j-1,k)...
            - theta_ADMM_iter(:,j+1,k);
    end
    sum_xi_xj(:,M,k) = theta_ADMM_iter(:,M,k) - theta_ADMM_iter(:,M-1,k);
    
    for i=1:M
        model = models{1,i};
        % p-update
        p_ADMM(:,i,k+1) = p_ADMM(:,i,k) + rho*sum_xi_xj(:,i,k);
        % NLL gradient computation
        [nll, grad] = getNlmlGrad(theta_ADMM_iter(:,i,k), @mySEard, model);
        % sum_x_xi_xj
        if i == 1
            sum_xj(:,1,k) = theta_ADMM_iter(:,2,k);
            card_n(1) = 1;
        elseif and(i>1, i<M)
            for j=2:M-1
                sum_xj(:,j,k) = theta_ADMM_iter(:,j-1,k) + theta_ADMM_iter(:,j+1,k);
                card_n(j) = 2;
            end
        else
            sum_xj(:,M,k) = theta_ADMM_iter(:,M-1,k);
            card_n(M) = 1;
        end
        % theta-update
        theta_ADMM_iter(:,i,k+1) = -(1/(Lip+2*rho*card_n(i)))*( grad ...
            - ( Lip + rho*card_n(i) )*theta_ADMM_iter(:,i,k)  ...
            + p_ADMM(:,i,k+1) - rho*sum_xj(:,i,k) );
    end
end
theta_ADMM = theta_ADMM_iter(:,:,k+1);
t = toc(t_start);
end