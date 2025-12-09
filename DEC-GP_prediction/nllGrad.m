function [f, df, sn2] = nllGrad(hyp_local, model, n, z_ADMM, beta_ADMM, rho)

% Covariance matrix
% Separable 2D cavarinca function
for i=1:n 
    for j=1:n
      C(i,j) = exp(- ( norm(model.X_norm(i,1)-model.X_norm(j,1))^2 )/(hyp_local(1) ) ...
          - ( norm(model.X_norm(i,2)-model.X_norm(j,2))^2 )/(hyp_local(2)) );  
    end
end
K = C  + (hyp_local(end) .* eye(n)) ; 
% C = covfunc(hyp_local(1:end-1), model.X_norm);
% K = C + exp(2 * hyp_local(end)) * eye(n);
% Cholesky decomposition
L = chol(K, 'lower');
invL = inv(L);
invK = invL'*invL;
invK_y = L' \ (L \ model.Y_norm);
% Inverse covariance
% invK = inv(K);
% detK = det(K);
% invK_y = K\model.Y;
yt_invK_y = model.Y_norm' * invK_y;

% Negative log-likelihood
c =  -n/2 * ( log(2*pi) - log(n) + 1 );
nll = -n/2 * log( yt_invK_y ) - trace(log(L)) + c ; %- .5*log(detK) + c; %  0.5*2*trace(log(L))
nll = -nll;

% x-ADMM function
f = ( nll + beta_ADMM'*(hyp_local - z_ADMM) + (rho/2)*(norm(hyp_local - z_ADMM)^2));

% Closed-form solution for scale or signal variance hyperparameter
sn2 = (1/n)*yt_invK_y;
% sn2 = 1;

% Derivative of noise variance hyperparameter
dnl_sf2 = ( (n/2)*invK_y'*invK_y )/( yt_invK_y ) - 0.5* trace( invK );

% Derivative of lengthscale hyperparameter
for i=1:n
    for j=1:n
        Omega_x(i,j) = C(i,j)* norm( model.X_norm(i,1) - model.X_norm(j,1) )^2/(hyp_local(1)^2);
        Omega_y(i,j) = C(i,j)* norm( model.X_norm(i,2) - model.X_norm(j,2) )^2/(hyp_local(2)^2);
    end
end
dnl_l1 = ( (n/2)*invK_y'*Omega_x*invK_y )/( yt_invK_y ) ...
            - 0.5* trace( invK*Omega_x );       
dnl_l2 = ( (n/2)*invK_y'*Omega_y*invK_y )/( yt_invK_y ) ...
            - 0.5* trace( invK*Omega_y );
        
% Gradient of log-likelihood for separable covariance
dnll = -[dnl_l1;dnl_l2;dnl_sf2];

df = (dnll + beta_ADMM + rho*(hyp_local - z_ADMM));

end

