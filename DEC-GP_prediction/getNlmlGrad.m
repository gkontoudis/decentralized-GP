function [nlZ, dnlZ] = getNlmlGrad(loghyp, covfunc, model)

% nlZ is the negative log marginal likelihood and dnlZ its partial
% derivatives wrt the hyperparameters.

hyps = exp(loghyp);
n = size(model.X, 1);
K = covfunc(loghyp(1:end-1), model.X);
if hyps(end)<0.005
    hyps(end) = 0.005; %ensure that sn does not fall bellow 0.005 to avoid ill-conditioned matrix
end
K = K + (hyps(end)^2) * eye(n);
L = chol(K, 'lower');
alpha = L' \ (L \ model.Y);
nlZ = -0.5 * model.Y' * alpha - trace(log(L)) - n/2 * log(2 * pi);
% negative log marginal likelihood
nlZ = -nlZ;

invL = inv(L);
alphaalphainvK = alpha * alpha' - invL' * invL;
dnlZ = zeros(size(hyps));

for i = 1:numel(hyps(1:end-1))
    dKdtheta = covfunc(loghyp(1:end-1), model.X, [], i);
    dnlZ(i) = 0.5 * sum(sum(alphaalphainvK .* dKdtheta')); % trace(A*B) = sum(sum(A.*B'))
end

dnlZ(end) = 0.5 * sum(sum(alphaalphainvK .* (2 * (hyps(end)^2) * eye(n))')); % changed (hyps(end)^2) to (hyps(end))
% gradient of nlml
dnlZ = -dnlZ;

end

