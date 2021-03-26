function [mu,s2,t_predict,k_M_x_all,K_M_x_all,mu_all,s2_all,kss] = aggregation_predict(Xt,models)
% Source code: https://github.com/LiuHaiTao01/GRBCM
% H.T. Liu 2018/06/01 (htliu@ntu.edu.sg)
%
% Aggregation GP for prediction
% Inputs:
%        Xt: a nt*d matrix containing nt d-dimensional test points
%        models: a cell structure that contains the sub-models built on subsets, where models{i} is the i-th model
%                 fitted to {Xi and Yi} for 1 <= i <= M
%        criterion: 'NPAE': nested pointwise aggregation of experts
% Outputs:
%         mu: a nt*1 vector that represents the prediction mean at nt test points
%         s2: a nt*1 vector that represents the prediction variance at nt test points
%         t_predict: computing time for predictions
%%

nt = size(Xt,1) ;  % number of test points
M = models{1}.Ms ; % number of experts

% normalization of test points Xt
if strcmp(models{1}.optSet.Xnorm,'Y')
    Xt = (Xt - repmat(models{1}.X_mean,nt,1)) ./ (repmat(models{1}.X_std,nt,1)) ;
end

% predictions of each submodel
t1 = clock ;
for i = 1:M
    [mu_experts{i},s2_experts{i}] = gp(models{i}.hyp,models{i}.inffunc,models{i}.meanfunc, ...
        models{i}.covfunc,models{i}.likfunc,models{i}.X_norm,models{i}.Y_norm,Xt);
end


% use an aggregation criterion to combine predictions from submodels
mu = zeros(nt,1) ; s2 = zeros(nt,1) ;
kss = feval(models{1}.covfunc,models{1}.hyp.cov,Xt,'diag'); % no noise
K_invs = inverseKernelMarix_submodels(models) ;
K_cross = crossKernelMatrix_nestedKG(models) ;
hyp_lik = models{1}.hyp.lik ; % noise parameter

mu_all = zeros(nt,M) ;
for i = 1:M, mu_all(:,i) = mu_experts{i}; s2_all(:,i) = s2_experts{i}; end
k_M_x_all = [];
K_M_x_all = [];
for i = 1:nt
    M_x = mu_all(i,:)' ;
    
    x = Xt(i,:) ;
    k_M_x = kernelVector_nestedKG(x,models,K_invs) ;
    [K_M_x,K_M_x_inv] = kernelMatrix_nestedKG(x,models,K_invs,K_cross) ;
    
    mu(i) = k_M_x'*K_M_x_inv*M_x ;
    s2(i) = kss(i) - k_M_x'*K_M_x_inv*k_M_x + exp(2*hyp_lik) ;
    k_M_x_all = [k_M_x_all k_M_x];
    K_M_x_all(:,:,i) = K_M_x;
end


muf = mu; s2f = s2 - exp(2*models{1}.hyp.lik);

% restore predictions if needed
if strcmp(models{1}.optSet.Ynorm,'Y')
    mu = mu*models{1}.Y_std + models{1}.Y_mean ;
    s2 = s2*(models{1}.Y_std)^2 ; % squared because it's std
    muf = muf*models{1}.Y_std + models{1}.Y_mean ;
    s2f = s2f*(models{1}.Y_std)^2 ;
end

t2 = clock ;
t_predict = etime(t2,t1) ;

end


%%%%%%%%%%%%%%%
function [K_invs] = inverseKernelMarix_submodels(models)
% calculate the covariance matrics Ks, the inverse matrics K_invs and the det of matrics K_dets of submodels
% used for the nestedKG criterion
M = length(models) ;

covfunc = models{1}.covfunc ; 
hyp_cov = models{1}.hyp.cov ; 
hyp_lik = models{1}.hyp.lik ;
for i = 1:M
    K_Xi = feval(covfunc,hyp_cov,models{i}.X_norm) + exp(2*hyp_lik)*eye(size(models{i}.X_norm,1)) ;

    K_invs{i} = eye(size(models{i}.X_norm,1))/K_Xi ;
end

end % end function


function K_cross = crossKernelMatrix_nestedKG(models)
% construct the covariance of training points
% used for the nestedKG criterion 
M = length(models) ;

covfunc = models{1}.covfunc ; hyp_cov = models{1}.hyp.cov ; hyp_lik = models{1}.hyp.lik ;
for i = 1:M 
    for j = 1:M 
        if i == j % self-covariance, should consider noise term
            K_cross{i}{j} = feval(covfunc,hyp_cov,models{i}.X_norm,models{j}.X_norm) + exp(2*hyp_lik)*eye(size(models{i}.X_norm,1)) ;
        else % cross-covariance
            K_cross{i}{j} = feval(covfunc,hyp_cov,models{i}.X_norm,models{j}.X_norm) ;
        end
    end
end

end % end function


function k_M = kernelVector_nestedKG(x,models,K_invs)
% construct the covariance between test points and training points
% used for the nestedKG criterion 
M = length(models) ;

covfunc = models{1}.covfunc ; hyp_cov = models{1}.hyp.cov ;
k_M = zeros(M,1) ;
for i = 1:M 
    k_x_Xi = feval(covfunc,hyp_cov,x,models{i}.X_norm);
    k_M(i) = k_x_Xi*K_invs{i}*k_x_Xi' ;
end

end % end function


function [K_M,K_M_inv] = kernelMatrix_nestedKG(x,models,K_invs,K_cross)
% construct the covariance of training points
% used for the nestedKG criterion 
M = length(models) ;

covfunc = models{1}.covfunc ; hyp_cov = models{1}.hyp.cov ; hyp_lik = models{1}.hyp.lik ;
K_M = zeros(M,M) ;
% obtain an upper triangular matrix to save compting time
for i = 1:M 
    for j = i:M 
        k_x_Xi = feval(covfunc,hyp_cov,x,models{i}.X_norm);
        k_Xj_x = feval(covfunc,hyp_cov,models{j}.X_norm,x);
        K_Xi_Xj = K_cross{i}{j} ;
        if i == j
            K_M(i,j) = 0.5*k_x_Xi*K_invs{i}*K_Xi_Xj*K_invs{j}*k_Xj_x ; % the coef 0.5 is used to ensure K_M = K_M + K_M' along diagonal line
        else
            K_M(i,j) = k_x_Xi*K_invs{i}*K_Xi_Xj*K_invs{j}*k_Xj_x ;
        end
    end
end
% obtain whole K_M
K_M = K_M + K_M' ;

jitter = 1e-10 ;
K_M_inv = eye(size(K_M,1))/(K_M + jitter*eye(M)) ;

end % end function


