function [t, theta,sn2] = clasADMM(rho, opts, models, tol_theta, max_iter, tol_ADMM,delta_ADMM )

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
n_all = opts.Ms;
l_b = 0.01*ones(3,1); % eps*ones(3,1); % .05*ones(3,1);

for i=1:n_all
    for k=1:max_iter
        theta_ADMM_iter(:,i,k) = [opts.ell; opts.ell; opts.sn2];
        beta_ADMM(:,i,k) = zeros(length(theta_ADMM_iter(:,1,1)),1);
    end
    sn2(i,1) = 1;
end
z_ADMM(:,1) = zeros(length(theta_ADMM_iter(:,1,1)),1);


for k = 1:max_iter
    % z-update
    %     zold = z_ADMM(:,k);
    for i=1:n_all
        theta_beta_local(:,i) = theta_ADMM_iter(:,i,k) + (1/rho).* beta_ADMM(:,i,k);
    end
    z_ADMM(:,k+1) = (1/n_all) .* sum(theta_beta_local');
    k
    
    theta_local = theta_ADMM_iter(:,:,k);
    for i=1:n_all
        % theta-update
        u_b = 2*var(models{1,i}.Y)*ones(3,1);
        n_local = size(models{1,i}.Y, 1);
        theta_local_old = theta_local(:,i)
        for kk = 1:max_iter
            kk
            
            % Limited memory BFGS bound
            %             [theta_local_lbfgsb,xhist,sn2_local] = ...
            %                 LBFGSB(theta_local_old,l_b,u_b,[],models{1,i},n,z_ADMM(:,k+1),beta_ADMM(:,i,k), rho)
            
            % Standard constraint minimization
            model = models{1,i};
            z_ADMM_new = z_ADMM(:,k+1);
            beta_ADMM_old = beta_ADMM(:,k);
            global z_ADMM_new beta_ADMM_old n_local model rho
            options = optimoptions('fmincon','SpecifyObjectiveGradient',true);
            theta_local_lbfgsb = fmincon('fun_cADMM',theta_local_old,[],[],[],[],l_b,u_b,[],options)
            sn2_local = scale(theta_local_lbfgsb)
            
            theta_norm = norm(theta_local_lbfgsb - theta_local_old)
            
            if theta_norm < tol_theta
                theta_ADMM_iter(:,i,k+1) = theta_local_lbfgsb;
                sn2(i,k+1) = sn2_local
                break;
            else
                theta_local_old = theta_local_lbfgsb
            end
            
        end
        
        % beta-update
        beta_ADMM(:,i,k+1) = beta_ADMM(:,i,k) + rho*(theta_ADMM_iter(:,i,k+1) - z_ADMM(:,k+1));
        
    end
    
    sn2(:,k+1) = (1/n_all)*sum(sn2(:,k+1))
    
    % diagnostics, reporting, termination checks
    ck = 1
    for i=1:n_all
        constraint_norm(i) = norm (theta_ADMM_iter(:,i,k+1) - z_ADMM(:,k+1))
        if constraint_norm(i) < tol_ADMM
            ck = ck+1
        end
    end
    if ck == n_all
        theta = theta_ADMM_iter(:,:,k+1)
        break;
    end
    
end

t = toc(t_start);

end

