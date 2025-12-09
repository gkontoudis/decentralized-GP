function [mu_dist_comp,s2_dist_comp,mu_dist_1_hop,...
    s2_dist_1_hop,mu_dist_2_hop,s2_dist_2_hop,...
    iter_jor_convrg_max_min,iter_dac_convrg_max_min,iter_dac_convrg_max_min_1,...
    iter_dac_convrg_max_min_2,nearest_neighbors,...
    iterations_pm_end,omega_all]...
    = dec_npae_star(nt,K_M_x,k_M_x,iter_jor,iter_dac,opts,kss,mu_all,models,...
    hyp_lik,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,thres_jor_max_min,...
    thres_dac_max_min,thres_cbnn,thres_pm,iter_pm)

n = opts.Ms;
n_all = n;
thres = thres_cbnn;
b_M = mu_all';
A = K_M_x;
b_k = k_M_x;
for j=1:nt
    L_1_nt(:,:,j) = L_1;
    L_2_nt(:,:,j) = L_2;
    L_comp_nt(:,:,j) = L_comp;
end
iterations_pm_end = zeros(1,nt);

for j=1:nt
    non_zero.index = find(k_M_x(:,j)>thres);
    nearest_neighbors(j) = length(non_zero.index); % count nearest neighbors
    
    if   nearest_neighbors(j)  > 1
        for z_cbnn=1:n_all
            if length(non_zero.index) == z_cbnn
                n = z_cbnn;
            end
        end
        
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
        A_prime = inv(K_M_x_D)*A_real;
        %     omega = 2/(max(eig(A_prime))+min(eig(A_prime)));
        [eig_A_max,eig_A_min,iterations_pm] = power_method_inverse(A_prime,thres_pm,iter_pm);
        iterations_pm_end(j) = iterations_pm;
        omega = 2/(eig_A_max+eig_A_min);
        omega_all(j) = omega;

        % Jacobi overrelaxation (JOR) in matrix form
        y_dist(:,1) = K_M_x_D*b_M_real; % ones(n,1);
        k_dist(:,1) = K_M_x_D*b_k_real; % ones(n,1);
        k_jor_max_min = 1;
        k_dac_max_min = 1;
        k_dac_max_min_1 = 1;
        k_dac_max_min_2 = 1;
        for i=1:iter_jor
            y_dist(:,i+1) = inv(K_M_x_D+omega.*K_M_x_L)*...
                (omega*b_M_real(:) - (omega.*K_M_x_U+(omega-1).*K_M_x_D)*y_dist(:,i));
            y_dist_max(i+1) = max(y_dist(:,i+1));
            y_dist_min(i+1) = min(y_dist(:,i+1));
            y_dist_max_min(i+1) = y_dist_max(i+1) - y_dist_min(i+1);
            y_dist_max_min_rel(i) = abs(y_dist_max_min(i+1)-y_dist_max_min(i))/y_dist_max_min(i);
            
            k_dist(:,i+1) = inv(K_M_x_D+omega.*K_M_x_L)*...
                (omega*b_k_real(:) - (omega.*K_M_x_U+(omega-1).*K_M_x_D)*k_dist(:,i));
            k_dist_max(i+1) = max(k_dist(:,i+1));
            k_dist_min(i+1) = min(k_dist(:,i+1));
            k_dist_max_min(i+1) = k_dist_max(i) - k_dist_min(i);
            k_dist_max_min_rel(i) = abs(k_dist_max_min(i+1)-k_dist_max_min(i))/k_dist_max_min(i);
            
            if y_dist_max_min_rel(i) < thres_jor_max_min && k_dist_max_min_rel(i) < thres_jor_max_min
                iter_jor_convrg_max_min(k_jor_max_min,j) = i+1;
                k_jor_max_min = k_jor_max_min + 1;
                if k_jor_max_min>3
                    break;
                end
            end
        end

        % Discrete average consensus (DAC) in matrix form
        for i=1:n
            z_dist_comp(i,1) =  b_k_real(i)*y_dist(i,iter_jor_convrg_max_min(1,j));
            l_dist_comp(i,1) =  b_k_real(i)*k_dist(i,iter_jor_convrg_max_min(1,j));
        end
        for i=1:n
            z_dist_1(i,1) =  b_k_real(i)*y_dist(i,iter_jor_convrg_max_min(1,j));
            l_dist_1(i,1) =  b_k_real(i)*k_dist(i,iter_jor_convrg_max_min(1,j));
        end
        for i=1:n
            z_dist_2(i,1) =  b_k_real(i)*y_dist(i,iter_jor_convrg_max_min(1,j));
            l_dist_2(i,1) =  b_k_real(i)*k_dist(i,iter_jor_convrg_max_min(1,j));
        end        
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
                if k_dac_max_min > 3
                    break;
                end
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
                if k_dac_max_min_1 > 3
                    break;
                end
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
                if k_dac_max_min_2 > 3
                    break;
                end
            end
        end
        
        if iter_dac_convrg_max_min(1,j) > n_all && n_all <= 5 % JOR+flooding
            iter_dac_convrg_max_min(1,j) =  n_all;
            mu_dist_comp(j) = sum(z_dist_comp(:,1))*models{1}.Y_std + models{1}.Y_mean;
            s2_dist_comp(j) = (kss(j) - sum(l_dist_comp(:,1)) + exp(2*hyp_lik))*(models{1}.Y_std)^2;
        else % JOR+DAC
            mu_dist_comp(j) = n*z_dist_comp(1,iter_dac_convrg_max_min(1,j))*models{1}.Y_std + models{1}.Y_mean;
            s2_dist_comp(j) = (kss(j) - n*l_dist_comp(1,iter_dac_convrg_max_min(1,j)) + exp(2*hyp_lik))*(models{1}.Y_std)^2;
        end
        
        if iter_dac_convrg_max_min_1(1,j) > n_all && n_all <= 5
            iter_dac_convrg_max_min_1(1,j) =  n_all;
            mu_dist_1_hop(j) = sum(z_dist_1(:,1))*models{1}.Y_std + models{1}.Y_mean;
            s2_dist_1_hop(j) = (kss(j) - sum(l_dist_1(:,1)) + exp(2*hyp_lik))*(models{1}.Y_std)^2;
        else
            mu_dist_1_hop(j) = n*z_dist_1(1,iter_dac_convrg_max_min_1(1,j))*models{1}.Y_std + models{1}.Y_mean;
            s2_dist_1_hop(j) = (kss(j) - n*l_dist_1(1,iter_dac_convrg_max_min_1(1,j)) + exp(2*hyp_lik))*(models{1}.Y_std)^2;
        end
        
        if iter_dac_convrg_max_min_2(1,j) > n_all && n_all <= 5
            iter_dac_convrg_max_min_2(1,j) =  n_all;
            mu_dist_2_hop(j) = sum(z_dist_2(:,1))*models{1}.Y_std + models{1}.Y_mean;
            s2_dist_2_hop(j) = (kss(j) - sum(l_dist_2(:,1)) + exp(2*hyp_lik))*(models{1}.Y_std)^2;
        else
            mu_dist_2_hop(j) = n*z_dist_2(1,iter_dac_convrg_max_min_2(1,j))*models{1}.Y_std + models{1}.Y_mean;
            s2_dist_2_hop(j) = (kss(j) - n*l_dist_2(1,iter_dac_convrg_max_min_2(1,j)) + exp(2*hyp_lik))*(models{1}.Y_std)^2;
        end

    else
        mu_dist_comp(j) = b_M(non_zero.index,j)*models{1}.Y_std + models{1}.Y_mean;
        s2_dist_comp(j) = s2_all(non_zero.index,j)*(models{1}.Y_std)^2;
        
        mu_dist_1_hop(j) = mu_dist_comp(j);
        s2_dist_1_hop(j) = s2_dist_comp(j);
        
        mu_dist_2_hop(j) = mu_dist_comp(j);
        s2_dist_2_hop(j) = s2_dist_comp(j);
    end
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
    P_comp = [];
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
    A_real = [];
    K_M_x_U = [];
    K_M_x_L = [];
    K_M_x_D = [];
    n=n_all;
end
end