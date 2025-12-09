function [mu,s2] = bcm(nt,mu_all,s2_all,models,kss)
s2_all_inv = 1./s2_all;
kss_e = kss + exp(2*models{1}.hyp.lik);
for j=1:nt
    s2_inv(j) = sum(s2_all_inv(j,:)) + (1-models{1}.Ms)/kss_e(j);
    s2(j) = 1/s2_inv(j);
    mu(j) = s2(j)*sum(s2_all_inv(j,:)*mu_all(j,:)');
end
% restore predictions if needed
if strcmp(models{1}.optSet.Ynorm,'Y')
    mu = mu*models{1}.Y_std + models{1}.Y_mean ;
    s2 = s2*(models{1}.Y_std)^2 ;
end
mu = mu';
s2 = s2';            
end