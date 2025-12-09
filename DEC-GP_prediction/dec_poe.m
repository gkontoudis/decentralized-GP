function [mu_dist_comp,s2_dist_comp,mu_dist_1_hop,...
    s2_dist_1_hop,mu_dist_2_hop,s2_dist_2_hop,...
    iter_dac_convrg_max_min,iter_dac_convrg_max_min_1,...
    iter_dac_convrg_max_min_2,nearest_neighbors] = ...
    dec_poe(mu_all,s2_all,s2_all_inv,models,...
        nt,k_M_x,iter_dac,opts,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,...
        thres_dac_max_min,thres_cbnn)


n = opts.Ms;
n_all = n;
thres = thres_cbnn;
mu_all = mu_all';
s2_all = s2_all';
s2_all_inv = s2_all_inv';
for j=1:nt
    L_1_nt(:,:,j) = L_1;
    L_2_nt(:,:,j) = L_2;
    L_comp_nt(:,:,j) = L_comp;
end

for j=1:nt
    non_zero.index = find(k_M_x(:,j)>thres);
    nearest_neighbors(j) = length(non_zero.index);
    if   nearest_neighbors(j)  > 1
        for z_cbnn=1:n_all
            if length(non_zero.index) == z_cbnn
                n = z_cbnn;
            end
        end
        mu_all_real = mu_all(non_zero.index(1):non_zero.index(end),j);
        s2_all_real = s2_all(non_zero.index(1):non_zero.index(end),j);
        s2_all_inv_real = s2_all_inv(non_zero.index(1):non_zero.index(end),j);
        
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
        k_dac_max_min = 1;
        k_dac_max_min_1 = 1;
        k_dac_max_min_2 = 1;
        
        for i=1:n
            z_dist_comp(i,1) =  s2_all_inv_real(i)*mu_all_real(i);
            l_dist_comp(i,1) =  s2_all_inv_real(i);
        end
            z_dist_1(:,1) =  z_dist_comp(:,1);
            z_dist_2(:,1) =  z_dist_comp(:,1);
            l_dist_1(:,1) =  l_dist_comp(:,1);
            l_dist_2(:,1) =  l_dist_comp(:,1);
            
        for i=1:iter_dac
            z_dist_comp(:,i+1) = P_comp*z_dist_comp(:,i);
            z_dist_max(i+1) = max(z_dist_comp(:,i+1));
            z_dist_min(i+1) = min(z_dist_comp(:,i+1));
            z_dist_max_min(i+1) = z_dist_max(i+1) - z_dist_min(i+1);
            
            l_dist_comp(:,i+1) =  P_comp*l_dist_comp(:,i);
            l_dist_max(i+1) = max(l_dist_comp(:,i+1));
            l_dist_min(i+1) = min(l_dist_comp(:,i+1));
            l_dist_max_min(i+1) = l_dist_max(i+1) - l_dist_min(i+1);
            
            if z_dist_max_min(i+1) < thres_dac_max_min && l_dist_max_min(i+1) < thres_dac_max_min
                iter_dac_convrg_max_min(k_dac_max_min,j) = i+1;
                k_dac_max_min = k_dac_max_min + 1;
            end
        end
        
        for i=1:iter_dac
            z_dist_1(:,i+1) = P_comp*z_dist_1(:,i);
            z_dist_max_1(i+1) = max(z_dist_1(:,i+1));
            z_dist_min_1(i+1) = min(z_dist_1(:,i+1));
            z_dist_max_min_1(i+1) = z_dist_max_1(i+1) - z_dist_min_1(i+1);
            
            l_dist_1(:,i+1) =  P_comp*l_dist_1(:,i);
            l_dist_max_1(i+1) = max(l_dist_1(:,i+1));
            l_dist_min_1(i+1) = min(l_dist_1(:,i+1));
            l_dist_max_min_1(i+1) = l_dist_max_1(i+1) - l_dist_min_1(i+1);
            
            if z_dist_max_min_1(i+1) < thres_dac_max_min && l_dist_max_min_1(i+1) < thres_dac_max_min
                iter_dac_convrg_max_min_1(k_dac_max_min_1,j) = i+1;
                k_dac_max_min_1 = k_dac_max_min_1 + 1;
            end
        end
        
        for i=1:iter_dac
            z_dist_2(:,i+1) = P_comp*z_dist_2(:,i);
            z_dist_max_2(i+1) = max(z_dist_2(:,i+1));
            z_dist_min_2(i+1) = min(z_dist_2(:,i+1));
            z_dist_max_min_2(i+1) = z_dist_max_2(i+1) - z_dist_min_2(i+1);
            
            l_dist_2(:,i+1) =  P_comp*l_dist_2(:,i);
            l_dist_max_2(i+1) = max(l_dist_2(:,i+1));
            l_dist_min_2(i+1) = min(l_dist_2(:,i+1));
            l_dist_max_min_2(i+1) = l_dist_max_2(i+1) - l_dist_min_2(i+1);
            
            if z_dist_max_min_2(i+1) < thres_dac_max_min && l_dist_max_min_2(i+1) < thres_dac_max_min
                iter_dac_convrg_max_min_2(k_dac_max_min_2,j) = i+1;
                k_dac_max_min_2 = k_dac_max_min_2 + 1;
            end
        end
        
        if iter_dac_convrg_max_min(1,j) > n_all && n_all <= 5 % flooding
            iter_dac_convrg_max_min(1,j) =  n_all;
            s2_inv_dist_comp(j) = sum(l_dist_comp(:,1));
            s2_dist_comp(j) = (1/s2_inv_dist_comp(j));
            mu_dist_comp(j) = (s2_dist_comp(j)*sum(z_dist_comp(:,1))*models{1}.Y_std) + models{1}.Y_mean;
            s2_dist_comp(j) = s2_dist_comp(j)*(models{1}.Y_std)^2; % Normalize after multiplying to mean value
              
        else % DAC
            s2_inv_dist_comp(j) = n*l_dist_comp(1,iter_dac_convrg_max_min(1,j));
            s2_dist_comp(j) = (1/s2_inv_dist_comp(j));
            mu_dist_comp(j) = (s2_dist_comp(j)*n*z_dist_comp(1,iter_dac_convrg_max_min(1,j))*models{1}.Y_std) + models{1}.Y_mean;
            s2_dist_comp(j) = (1/s2_inv_dist_comp(j))*(models{1}.Y_std)^2;
        end
         
        if iter_dac_convrg_max_min_1(1,j) > n_all && n_all <= 5 % flooding
            iter_dac_convrg_max_min_1(1,j) =  n_all;
            s2_inv_dist_1_hop(j) = sum(l_dist_1(:,1));
            s2_dist_1_hop(j) = (1/s2_inv_dist_1_hop(j));
            mu_dist_1_hop(j) = s2_dist_1_hop(j)*sum(z_dist_1(:,1))*models{1}.Y_std + models{1}.Y_mean;
            s2_dist_1_hop(j) = (1/s2_inv_dist_1_hop(j))*(models{1}.Y_std)^2;
        else % DAC
            s2_inv_dist_1_hop(j) = n*l_dist_1(1,iter_dac_convrg_max_min_1(1,j));
            s2_dist_1_hop(j) = (1/s2_inv_dist_1_hop(j));
            mu_dist_1_hop(j) = s2_dist_1_hop(j)*n*z_dist_1(1,iter_dac_convrg_max_min_1(1,j))*models{1}.Y_std + models{1}.Y_mean;
            s2_dist_1_hop(j) = (1/s2_inv_dist_1_hop(j))*(models{1}.Y_std)^2;
        end
        
        if iter_dac_convrg_max_min_2(1,j) > n_all && n_all <= 5 % flooding
            iter_dac_convrg_max_min_2(1,j) =  n_all;
            s2_inv_dist_2_hop(j) = sum(l_dist_2(:,1));
            s2_dist_2_hop(j) = (1/s2_inv_dist_2_hop(j));
            mu_dist_2_hop(j) = s2_dist_2_hop(j)*sum(z_dist_2(:,1))*models{1}.Y_std + models{1}.Y_mean;
            s2_dist_2_hop(j) = (1/s2_inv_dist_2_hop(j))*(models{1}.Y_std)^2;
        else % DAC
            s2_inv_dist_2_hop(j) = n*l_dist_2(1,iter_dac_convrg_max_min_2(1,j));
            s2_dist_2_hop(j) = (1/s2_inv_dist_2_hop(j));
            mu_dist_2_hop(j) = s2_dist_2_hop(j)*n*z_dist_2(1,iter_dac_convrg_max_min_2(1,j))*models{1}.Y_std + models{1}.Y_mean;
            s2_dist_2_hop(j) = (1/s2_inv_dist_2_hop(j))*(models{1}.Y_std)^2;
        end
    
    else
        s2_dist_comp(j) = (s2_all(non_zero.index,j));
        mu_dist_comp(j) = mu_all(non_zero.index,j)*models{1}.Y_std + models{1}.Y_mean; % for (g)poe and (r)bcm when we have 1 agent the variance vanishes
        s2_dist_comp(j) = (s2_all(non_zero.index,j))*(models{1}.Y_std)^2;
        
        s2_dist_1_hop(j) = s2_dist_comp(j);
        mu_dist_1_hop(j) = mu_dist_comp(j);
        
        s2_dist_2_hop(j) = s2_dist_comp(j);
        mu_dist_2_hop(j) = mu_dist_comp(j); 
    end
    mu_all_real = [];
    s2_all_real = [];
    s2_all_inv_real = [];
    non_zero.index = [];
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
    z_dist_comp = [];
    l_dist_comp = [];
    z_dist_1 =  [];
    l_dist_1 =  [];
    z_dist_2 =  [];
    l_dist_2 = [];
    P_1 = [];
    P_2 = [];
    P_comp = [];
    n=n_all;       
end
end