function [mu,mu_all,mu_comm,s2,s2_all,s2_comm,models_cross] = ...
    grbcm(nt,models_grbcm,Xt,n_per)
M = models_grbcm{1}.Ms ; % number of experts
for i = 1:M+1
    if i == 1 % Communication expert model for prediction
        models_cross{i} = models_grbcm{1} ;
        models_cross{i}.X = [];
        models_cross{i}.X = models_grbcm{1}.X(n_per+1:end,:); 
        models_cross{i}.Y = [];
        models_cross{i}.Y = models_grbcm{1}.Y(n_per+1:end,:); % X1 + Xi % Y1 + Yi
        models_cross{i}.X_norm = []; 
        models_cross{i}.X_norm = models_grbcm{1}.X_norm(n_per+1:end,:); 
        models_cross{i}.Y_norm = [];
        models_cross{i}.Y_norm = models_grbcm{1}.Y_norm(n_per+1:end,:);
    else % Augmented glocal models
        models_cross{i} = models_grbcm{i-1};
    end
end
for i = 1:M+1 % local prediction
    [mu_crossExperts(:,i),s2_crossExperts(:,i)] = gp(models_cross{i}.hyp,...
        models_cross{i}.inffunc,models_cross{i}.meanfunc, ...
        models_cross{i}.covfunc,models_cross{i}.likfunc,...
        models_cross{i}.X_norm,models_cross{i}.Y_norm,Xt);
end
s2_comm = s2_crossExperts(:,1);
s2_comm_inv = 1./s2_comm;
s2_all = s2_crossExperts(:,2:end);
s2_all_inv = 1./s2_all;
mu_comm = mu_crossExperts(:,1);
mu_all = mu_crossExperts(:,2:end);
for j=1:nt % aggregation
    beta(j,:) = [1 0.5*(log(s2_comm(j).*ones(1,M-1)) - log(s2_all(j,2:end)))];  
    s2_inv(j) = sum(beta(j,:).*s2_all_inv(j,:)) + (1-sum(beta(j,:)))*s2_comm_inv(j);
    s2(j) = 1/s2_inv(j);
    mu(j) = s2(j)*( sum(beta(j,:).*s2_all_inv(j,:).*mu_all(j,:))  +  ...
        (1 - sum(beta(j,:)))*s2_comm_inv(j)*mu_comm(j) );
end
if strcmp(models_grbcm{1}.optSet.Ynorm,'Y') % restore predictions if needed
    mu = mu*models_grbcm{1}.Y_std + models_grbcm{1}.Y_mean ;
    s2 = s2*(models_grbcm{1}.Y_std)^2 ; % squared because it's std
end
mu = mu';
s2 = s2'; 
end