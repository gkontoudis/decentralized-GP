function [k_dist,mu_dist_comp,s2_dist_comp,mu_dist_1_hop,s2_dist_1_hop,mu_dist_2_hop,s2_dist_2_hop,...
    iter_jor_convrg,iter_dac_convrg_mu_comp,iter_dac_convrg_mu_1,iter_dac_convrg_mu_2,...
    iter_dac_convrg_sigma_comp,iter_dac_convrg_sigma_1,iter_dac_convrg_sigma_2,l_dist_1]...
    = jor_dac(nt,K_M_x,k_M_x,iter_jor,iter_dac,opts,kss,k_cent,y_cent,z_cent,l_cent,mu_all,models,...
    hyp_lik,epsilon_1,L_1,epsilon_2,L_2,epsilon_comp,L_comp,thres_jor,thres_dac_mu,thres_dac_sigma)

P_1 = eye(opts.Ms,opts.Ms) - epsilon_1*L_1;
P_2 = eye(opts.Ms,opts.Ms) - epsilon_2*L_2;
P_comp = eye(opts.Ms,opts.Ms) - epsilon_comp*L_comp;

for j=1:nt
    % Decompose cross-covariance
    K_M_x_U = triu(K_M_x(:,:,j),1);
    K_M_x_L = tril(K_M_x(:,:,j),-1);
    K_M_x_D = diag(diag(K_M_x(:,:,j)));
    
    
    
    % Relaxation facotr
    A_prime = inv(K_M_x_D)*K_M_x(:,:,j);
%     omega = 2/(max(eig(A_prime))+min(eig(A_prime)));
    omega = 2/opts.Ms;
    % omega = .1
    % omega = .15
    % omega = .2
    % omega = .25
    % omega = .5

    % Jacobi overrelaxation (JOR) in matrix form
    y_dist(:,1,j) = ones(opts.Ms,1);
    k_dist(:,1,j) = ones(opts.Ms,1);
    k = 1;
    k_dac_mu_comp = 1;
    k_dac_sigma_comp = 1;
    k_dac_mu_1 = 1;
    k_dac_sigma_1 = 1;
    k_dac_mu_2 = 1;
    k_dac_sigma_2 = 1;
    for i=1:iter_jor
        y_dist(:,i+1,j) = inv(K_M_x_D+omega.*K_M_x_L)*...
            (omega*mu_all(j,:)' - (omega.*K_M_x_U+(omega-1).*K_M_x_D)*y_dist(:,i,j));
        
        k_dist(:,i+1,j) = inv(K_M_x_D+omega.*K_M_x_L)*...
            (omega*k_M_x(:,j) - (omega.*K_M_x_U+(omega-1).*K_M_x_D)*k_dist(:,i,j));
        
        mu_dist_temp(j,i) = k_M_x(:,j)'*y_dist(:,i+1,j)*models{1}.Y_std + models{1}.Y_mean;
        
        if abs(sum(k_dist(:,i+1,j)) - sum(k_cent(:,j)))/opts.Ms < thres_jor && ...
                abs(sum(y_dist(:,i+1,j)) - sum(y_cent(:,j)))/opts.Ms < thres_jor
            iter_jor_convrg(k,j) = i+1;
            k = k+1;
        end
    end
    
    y_error(:,j) = y_dist(:,end,j)-y_cent(:,j);
    
    % Jacobi overrelaxation (JOR) in alternative matrix form
    y_dist_alt(:,1,j) = ones(opts.Ms,1);
    k_dist_alt(:,1,j) = ones(opts.Ms,1);
    for i=1:iter_jor
        y_dist_alt(:,i+1,j) = y_dist_alt(:,i,j) + omega.*inv(K_M_x_D)*mu_all(j,:)'...
            -omega.*inv(K_M_x_D)*K_M_x(:,:,j)*y_dist_alt(:,i,j);
        
        k_dist_alt(:,i+1,j) =k_dist_alt(:,i,j) + omega.*inv(K_M_x_D)*k_M_x(:,j)...
            -omega.*inv(K_M_x_D)*K_M_x(:,:,j)*k_dist_alt(:,i,j);
        
        mu_dist_temp_alt(j,i) = k_M_x(:,j)'*y_dist_alt(:,i+1,j)*models{1}.Y_std + models{1}.Y_mean;
        
    end
    y_error(:,j) = y_dist(:,end,j)-y_cent(:,j);
    y_error_alt(:,j) = y_dist(:,end,j) - y_dist_alt(:,end,j);
    
    % Discrete average consensus (DAC) in matrix form
    for i=1:opts.Ms
        z_dist_comp(i,1,j) =  k_M_x(i,j)*y_dist(i,end,j);
        l_dist_comp(i,1,j) =  k_M_x(i,j)*k_dist(i,end,j);
    end
    for i=1:opts.Ms
        z_dist_1(i,1,j) =  k_M_x(i,j)*y_dist(i,end,j);
        l_dist_1(i,1,j) =  k_M_x(i,j)*k_dist(i,end,j);
    end
    for i=1:opts.Ms
        z_dist_2(i,1,j) =  k_M_x(i,j)*y_dist(i,end,j);
        l_dist_2(i,1,j) =  k_M_x(i,j)*k_dist(i,end,j);
    end
    
%     mu_dist_avg(j,1) = k_M_x(:,j)'*y_dist(:,end,j);
    for i=1:iter_dac
        z_dist_comp(:,i+1,j) = P_comp*z_dist_comp(:,i,j);
        l_dist_comp(:,i+1,j) =  P_comp*(l_dist_comp(:,i,j));
        
        if abs(sum(z_dist_comp(1,i+1,j))*opts.Ms - z_cent(:,j)) < thres_dac_mu
            iter_dac_convrg_mu_comp(k_dac_mu_comp,j) = i+1;
            k_dac_mu_comp = k_dac_mu_comp+1;
        end
        
        if abs(sum(l_dist_comp(1,i+1,j))*opts.Ms - l_cent(:,j)) < thres_dac_sigma
            iter_dac_convrg_sigma_comp(k_dac_sigma_comp,j) = i+1;
            k_dac_sigma_comp = k_dac_sigma_comp+1;
        end
        
    end


    for i=1:iter_dac
        z_dist_1(:,i+1,j) = P_1*z_dist_1(:,i,j);
        l_dist_1(:,i+1,j) =  P_1*(l_dist_1(:,i,j));
        
        if abs(sum(z_dist_1(1,i+1,j))*opts.Ms - z_cent(:,j)) < thres_dac_mu
            iter_dac_convrg_mu_1(k_dac_mu_1,j) = i+1;
            k_dac_mu_1 = k_dac_mu_1+1;
        end
        
        if abs(sum(l_dist_1(1,i+1,j))*opts.Ms - l_cent(:,j)) < thres_dac_sigma
            iter_dac_convrg_sigma_1(k_dac_sigma_1,j) = i+1;
            k_dac_sigma_1 = k_dac_sigma_1+1;
        end
        
    end
    
    for i=1:iter_dac
        z_dist_2(:,i+1,j) = P_2*z_dist_2(:,i,j);
        l_dist_2(:,i+1,j) =  P_2*(l_dist_2(:,i,j));
        
        if abs(sum(z_dist_2(1,i+1,j))*opts.Ms - z_cent(:,j)) < thres_dac_mu
            iter_dac_convrg_mu_2(k_dac_mu_2,j) = i+1;
            k_dac_mu_2 = k_dac_mu_2+1;
        end
        
        if abs(sum(l_dist_2(1,i+1,j))*opts.Ms - l_cent(:,j)) < thres_dac_sigma
            iter_dac_convrg_sigma_2(k_dac_sigma_2,j) = i+1;
            k_dac_sigma_2 = k_dac_sigma_2+1;
        end
    end
    mu_dist_comp(j) = opts.Ms*z_dist_comp(1,iter_dac_convrg_mu_comp(1,j),j)*models{1}.Y_std + models{1}.Y_mean;
    s2_dist_comp(j) = (kss(j) - opts.Ms*l_dist_comp(1,iter_dac_convrg_sigma_comp(1,j),j) + exp(2*hyp_lik))*(models{1}.Y_std)^2;

    mu_dist_1_hop(j) = opts.Ms*z_dist_1(1,iter_dac_convrg_mu_1(1,j),j)*models{1}.Y_std + models{1}.Y_mean;
    s2_dist_1_hop(j) = (kss(j) - opts.Ms*l_dist_1(1,iter_dac_convrg_sigma_1(1,j),j) + exp(2*hyp_lik))*(models{1}.Y_std)^2;
    
    mu_dist_2_hop(j) = opts.Ms*z_dist_2(1,iter_dac_convrg_mu_2(1,j),j)*models{1}.Y_std + models{1}.Y_mean;
    s2_dist_2_hop(j) = (kss(j) - opts.Ms*l_dist_2(1,iter_dac_convrg_sigma_2(1,j),j) + exp(2*hyp_lik))*(models{1}.Y_std)^2;
end



end