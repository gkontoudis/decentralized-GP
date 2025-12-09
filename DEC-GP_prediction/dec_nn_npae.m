function [mu_dist_comp,s2_dist_comp,mu_dist_1_hop,s2_dist_1_hop,mu_dist_2_hop,s2_dist_2_hop,...
    iter_jor_convrg,iter_dac_convrg_mu_comp,iter_dac_convrg_mu_1,iter_dac_convrg_mu_2,...
    iter_dac_convrg_sigma_comp,iter_dac_convrg_sigma_1,iter_dac_convrg_sigma_2,...
    nearest_neighbors]...
    = dec_nn_npae(nt,K_M_x,k_M_x,iter_jor,iter_dac,opts,kss,k_cent,y_cent,...
    z_cent,l_cent,mu_all,s2_all,models,...
    hyp_lik,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,thres_jor_star,...
    thres_dac_mu,thres_dac_sigma,thres_cbnn)

n = opts.Ms;
n_all = n;
thres = thres_cbnn;
b_M = mu_all';
A = K_M_x;
b_k = k_M_x;
s2_all = s2_all';

for j=1:nt
    L_1_nt(:,:,j) = L_1;
    L_2_nt(:,:,j) = L_2;
    L_comp_nt(:,:,j) = L_comp;
end

for j=1:nt
    non_zero.index = find(k_M_x(:,j)>thres);
    
%     for z_cbnn=1:n_all
%         if length(non_zero.index) == z_cbnn
%             n = z_cbnn;
%         end
%     end
%     if n_all>4 && n==1
%         non_zero.index = find(b_k(:,cov)>(thres-.00955));
%         for z_cbnn=1:n_all
%             if length(non_zero.index) == z_cbnn
%                 n = z_cbnn;
%             end
%         end
%     end
    nearest_neighbors(j) = length(non_zero.index); % count nearest neighbors
    
    if   nearest_neighbors(j)  > 1
    A_real = A(non_zero.index(1):non_zero.index(end),...
        non_zero.index(1):non_zero.index(end),j);
    b_M_real = b_M(non_zero.index(1):non_zero.index(end),j);
    b_k_real = b_k(non_zero.index(1):non_zero.index(end),j);
    
    if length(non_zero.index) == n_all
        epsilon_comp = 1/Delta_comp - .01;
        epsilon_1 = 1/Delta_1 - .01;
        epsilon_2 = 1/Delta_2 - .01;
        P_1 = eye(n) - epsilon_1*L_1_nt(:,:,j);
        P_2 = eye(n) - epsilon_2*L_2_nt(:,:,j);
        P_comp = eye(n) - epsilon_comp*L_comp_nt(:,:,j);
        
    else
        L_1_nt_reduced = L_1_nt(non_zero.index(1):non_zero.index(end),...
            non_zero.index(1):non_zero.index(end),j);
        L_1_cbnn = triu(L_1_nt_reduced,1) + tril(L_1_nt_reduced,-1)...
            + diag(abs (sum(triu(L_1_nt_reduced,1))+sum(tril(L_1_nt_reduced,-1)) ) );
        epsilon_1 = 1/(max(diag(L_1_cbnn))) - .01;
        P_1 = eye(n) - epsilon_1*L_1_cbnn;
        
        L_2_nt_reduced = L_2_nt(non_zero.index(1):non_zero.index(end),...
            non_zero.index(1):non_zero.index(end),j);
        L_2_cbnn = triu(L_2_nt_reduced,1) + tril(L_2_nt_reduced,-1)...
            + diag(abs (sum(triu(L_2_nt_reduced,1))+sum(tril(L_2_nt_reduced,-1)) ) );
        epsilon_2 = 1/(max(diag(L_2_cbnn))) - .01;
        P_2 = eye(n) - epsilon_2*L_2_cbnn;
        
        L_comp_nt_reduced = L_comp_nt(non_zero.index(1):non_zero.index(end),...
            non_zero.index(1):non_zero.index(end),j);
        L_comp_cbnn = triu(L_comp_nt_reduced,1) + tril(L_comp_nt_reduced,-1)...
            + diag(abs (sum(triu(L_comp_nt_reduced,1))+sum(tril(L_comp_nt_reduced,-1)) ) );
        epsilon_comp = 1/(max(diag(L_comp_cbnn))) - .01;
        P_comp = eye(n) - epsilon_comp*L_comp_cbnn;
    end
    
    
    % Decompose cross-covariance
    K_M_x_U = triu(A_real(:,:),1);
    K_M_x_L = tril(A_real(:,:),-1);
    K_M_x_D = diag(diag(A_real(:,:)));
    
    
    
    % Relaxation facotr
%     A_prime = inv(K_M_x_D)*A_real;
%     omega = 2/(max(eig(A_prime))+min(eig(A_prime)));
%     
%     [eig_A_max,eig_A_min,iterations_pm_end] = power_method_inverse(A_prime,thres_pm,iter_pm);
%     omega = 2/(eig_A_max+eig_A_min);
    omega = 2/n;
    % omega = .1
    % omega = .15
    % omega = .2
    % omega = .25
    % omega = .5

    % Jacobi overrelaxation (JOR) in matrix form
    y_dist(:,1) = ones(n,1);
    k_dist(:,1) = ones(n,1);
    k = 1;
    k_dac_mu_comp = 1;
    k_dac_sigma_comp = 1;
    k_dac_mu_1 = 1;
    k_dac_sigma_1 = 1;
    k_dac_mu_2 = 1;
    k_dac_sigma_2 = 1;
    for i=1:iter_jor
        y_dist(:,i+1) = inv(K_M_x_D+omega.*K_M_x_L)*...
            (omega*b_M_real(:) - (omega.*K_M_x_U+(omega-1).*K_M_x_D)*y_dist(:,i));
        
        k_dist(:,i+1) = inv(K_M_x_D+omega.*K_M_x_L)*...
            (omega*b_k_real(:) - (omega.*K_M_x_U+(omega-1).*K_M_x_D)*k_dist(:,i));
        
        mu_dist_temp(j,i) = b_k_real(:)'*y_dist(:,i+1)*models{1}.Y_std + models{1}.Y_mean;
        
        if abs(k_dist(1,i+1) - k_cent(non_zero.index(1),j)) < thres_jor_star && ...
                abs(y_dist(1,i+1) - y_cent(non_zero.index(1),j)) < thres_jor_star && ...
                abs(k_dist(end,i+1) - k_cent(non_zero.index(end),j)) < thres_jor_star && ...
                abs(y_dist(end,i+1) - y_cent(non_zero.index(end),j)) < thres_jor_star
            iter_jor_convrg(k,j) = i+1;
            k = k+1;
        end
    end
    
%     y_error(:,j) = y_dist(:,end,j)-y_cent(:,j);
    
    % Jacobi overrelaxation (JOR) in alternative matrix form
%     y_dist_alt(:,1) = ones(n,1);
%     k_dist_alt(:,1) = ones(n,1);
%     for i=1:iter_jor
%         y_dist_alt(:,i+1) = y_dist_alt(:,i) + omega.*inv(K_M_x_D)*b_M_real(:)...
%             -omega.*inv(K_M_x_D)*A_real(:,:)*y_dist_alt(:,i);
%         
%         k_dist_alt(:,i+1) =k_dist_alt(:,i) + omega.*inv(K_M_x_D)*b_k_real(:)...
%             -omega.*inv(K_M_x_D)*A_real(:,:)*k_dist_alt(:,i);
%         
%         mu_dist_temp_alt(j,i) = b_k_real(:)'*y_dist_alt(:,i+1)*models{1}.Y_std + models{1}.Y_mean;
%         
%     end
%     y_error(:,j) = y_dist(:,end,j)-y_cent(:,j);
%     y_error_alt(:) = y_dist(:,end) - y_dist_alt(:,end);
    
    % Discrete average consensus (DAC) in matrix form
    for i=1:n
        z_dist_comp(i,1) =  b_k_real(i)*y_dist(i,end);
        l_dist_comp(i,1) =  b_k_real(i)*k_dist(i,end);
    end
    for i=1:n
        z_dist_1(i,1) =  b_k_real(i)*y_dist(i,end);
        l_dist_1(i,1) =  b_k_real(i)*k_dist(i,end);
    end
    for i=1:n
        z_dist_2(i,1) =  b_k_real(i)*y_dist(i,end);
        l_dist_2(i,1) =  b_k_real(i)*k_dist(i,end);
    end
    
%     mu_dist_avg(j,1) = k_M_x(:,j)'*y_dist(:,end,j);
    for i=1:iter_dac
        z_dist_comp(:,i+1) = P_comp*z_dist_comp(:,i);
        l_dist_comp(:,i+1) =  P_comp*l_dist_comp(:,i);
        
        if abs(sum(z_dist_comp(1,i+1))*n - z_cent(1,j)) < thres_dac_mu
            iter_dac_convrg_mu_comp(k_dac_mu_comp,j) = i+1;
            k_dac_mu_comp = k_dac_mu_comp+1;
        end
        
        if abs(sum(l_dist_comp(1,i+1))*n - l_cent(1,j)) < thres_dac_sigma
            iter_dac_convrg_sigma_comp(k_dac_sigma_comp,j) = i+1;
            k_dac_sigma_comp = k_dac_sigma_comp+1;
        end
        
    end


    for i=1:iter_dac
        z_dist_1(:,i+1) = P_1*z_dist_1(:,i);
        l_dist_1(:,i+1) =  P_1*(l_dist_1(:,i));
        
        if abs(sum(z_dist_1(1,i+1))*n - z_cent(1,j)) < thres_dac_mu
            iter_dac_convrg_mu_1(k_dac_mu_1,j) = i+1;
            k_dac_mu_1 = k_dac_mu_1+1;
        end
        
        if abs(sum(l_dist_1(1,i+1))*n - l_cent(1,j)) < thres_dac_sigma
            iter_dac_convrg_sigma_1(k_dac_sigma_1,j) = i+1;
            k_dac_sigma_1 = k_dac_sigma_1+1;
        end
        
    end
    
    for i=1:iter_dac
        z_dist_2(:,i+1) = P_2*z_dist_2(:,i);
        l_dist_2(:,i+1) =  P_2*(l_dist_2(:,i));
        
        if abs(sum(z_dist_2(1,i+1))*n - z_cent(1,j)) < thres_dac_mu
            iter_dac_convrg_mu_2(k_dac_mu_2,j) = i+1;
            k_dac_mu_2 = k_dac_mu_2+1;
        end
        
        if abs(sum(l_dist_2(1,i+1))*n - l_cent(1,j)) < thres_dac_sigma
            iter_dac_convrg_sigma_2(k_dac_sigma_2,j) = i+1;
            k_dac_sigma_2 = k_dac_sigma_2+1;
        end
    end
    
    mu_dist_comp(j) = n*z_dist_comp(1,iter_dac_convrg_mu_comp(1,j))*models{1}.Y_std + models{1}.Y_mean;
    s2_dist_comp(j) = (kss(j) - n*l_dist_comp(1,iter_dac_convrg_sigma_comp(1,j)) + exp(2*hyp_lik))*(models{1}.Y_std)^2;

    mu_dist_1_hop(j) = n*z_dist_1(1,iter_dac_convrg_mu_1(1,j))*models{1}.Y_std + models{1}.Y_mean;
    s2_dist_1_hop(j) = (kss(j) - n*l_dist_1(1,iter_dac_convrg_sigma_1(1,j)) + exp(2*hyp_lik))*(models{1}.Y_std)^2;
    
    mu_dist_2_hop(j) = n*z_dist_2(1,iter_dac_convrg_mu_2(1,j))*models{1}.Y_std + models{1}.Y_mean;
    s2_dist_2_hop(j) = (kss(j) - n*l_dist_2(1,iter_dac_convrg_sigma_2(1,j)) + exp(2*hyp_lik))*(models{1}.Y_std)^2;
    
    else
        
        mu_dist_comp(j) = b_M(non_zero.index,j)*models{1}.Y_std + models{1}.Y_mean;
        s2_dist_comp(j) = (kss(j) - s2_all(non_zero.index,j) + exp(2*hyp_lik))*(models{1}.Y_std)^2;
        
        mu_dist_1_hop(j) = b_M(non_zero.index,j)*models{1}.Y_std + models{1}.Y_mean;
        s2_dist_1_hop(j) = (kss(j) - s2_all(non_zero.index,j) + exp(2*hyp_lik))*(models{1}.Y_std)^2;
        
        mu_dist_2_hop(j) = b_M(non_zero.index,j)*models{1}.Y_std + models{1}.Y_mean;
        s2_dist_2_hop(j) = (kss(j) - s2_all(non_zero.index,j) + exp(2*hyp_lik))*(models{1}.Y_std)^2;
        
        iter_jor_convrg(1,j) = 0;
        iter_dac_convrg_mu_comp(1,j) = 0;
        iter_dac_convrg_mu_1(1,j) = 0;
        iter_dac_convrg_mu_2(1,j) = 0;
        iter_dac_convrg_sigma_comp(1,j) = 0;
        iter_dac_convrg_sigma_1(1,j) = 0;
        iter_dac_convrg_sigma_2(1,j) = 0;
    end

    L_1_nt_reduced = [];
    L_1_cbnn = [];
    epsilon_1 = [];
    P_1 = [];
    L_2_nt_reduced = [];
    L_2_cbnn = [];
    epsilon_2 = [];
    P_2 = [];
    L_comp_nt_reduced = [];
    L_comp_cbnn = [];
    epsilon_comp = [];
    P_comp = [];
    y_dist_alt = [];
    k_dist_alt = [];
    z_dist_comp = [];
    l_dist_comp = [];
    z_dist_1 =  [];
    l_dist_1 =  [];
    z_dist_2 =  [];
    l_dist_2 = [];
    y_dist = [];
    k_dist = [];
    b_M_real = [];
    b_k_real = [];
    P_1 = [];
    P_2 = [];
    P_comp = [];
    A_real = [];
    K_M_x_U = [];
    K_M_x_L = [];
    K_M_x_D = [];
    n=n_all;
end



end