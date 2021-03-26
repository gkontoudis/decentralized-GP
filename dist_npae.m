function [mu,s2,mu_2_hop,s2_2_hop,...
    iter_dale_convrg_max_min,iter_dale_convrg_max_min_2_hop,...
    nearest_neighbors]...
    = dist_npae (nt,K_M_x,k_M_x,iter_dale_1_hop,iter_dale_2_hop,opts,kss,...
    mu_all,s2_all,models,hyp_lik,thres_dale_max_min,thres_cbnn)

n = opts.Ms;
iter_1 = iter_dale_1_hop;
iter_2 = iter_dale_2_hop;
n_all = n;
thres = thres_cbnn;
b_M = mu_all';
A = K_M_x;
b_k = k_M_x;
s2_all = s2_all';

for cov=1:nt % nt prediction points
    % Covariance-based nearest neighbors
    non_zero.index = find(b_k(:,cov)>thres);
    nearest_neighbors(cov) = length(non_zero.index); % count nearest neighbors
    
    if   nearest_neighbors(cov)  > 1
        for z_dale=1:n_all
            if length(non_zero.index) == z_dale
                n = z_dale;
            end
        end
        
        A_real = A(non_zero.index(1):non_zero.index(end),...
            non_zero.index(1):non_zero.index(end),cov);
        b_M_real = b_M(non_zero.index(1):non_zero.index(end),cov);
        b_k_real =b_k(non_zero.index(1):non_zero.index(end),cov);
        
        % Initialization
        for i =1:n
            R_real(:,i) = A_real(i,:)'*inv( A_real(i,:)*A_real(i,:)' );
            P_real(:,:,i) = eye(n) - (R_real(:,i)*A_real(i,:) ) ;
            Q_M_real(:,i) = R_real(:,i)*b_M_real(i);
            Q_k_real(:,i) = R_real(:,i)*b_k_real(i);
            x_real(:,1,i) = ones(n,1);
            y_real(:,1,i) = ones(n,1);
            
            x_real_2_hop(:,1,i) = ones(n,1);
            y_real_2_hop(:,1,i) = ones(n,1);
            for j=1:n
                x_real(j,1,i) = b_M_real(j)/A_real(j,j);
                y_real(j,1,i) = b_k_real(j)/A_real(j,j);
                
                x_real_2_hop(j,1,i) = b_M_real(j)/A_real(j,j);
                y_real_2_hop(j,1,i) = b_k_real(j)/A_real(j,j);
            end
        end
        
        % DALE for path graph topology
        if n_all == 2
            for i=1:iter_1
                x_real(:,i+1,1) = (1/1)*P_real(:,:,1)*( x_real(:,i,2) ) + Q_M_real(:,1);
                x_real(:,i+1,2) = (1/1)*P_real(:,:,2)*( x_real(:,i,1) ) + Q_M_real(:,2);
                y_real(:,i+1,1) = (1/1)*P_real(:,:,1)*( y_real(:,i,2) ) + Q_k_real(:,1);
                y_real(:,i+1,2) = (1/1)*P_real(:,:,2)*( y_real(:,i,1) ) + Q_k_real(:,2);
            end
        else
            k_dale_max_min = 1;
            for i=1:iter_1
                x_real(:,i+1,1) = (1/1)*P_real(:,:,1)*( x_real(:,i,2) ) + Q_M_real(:,1);
                y_real(:,i+1,1) = (1/1)*P_real(:,:,1)*( y_real(:,i,2) ) + Q_k_real(:,1);
                for j=1:n-2
                    x_real(:,i+1,j+1) = (1/2)*P_real(:,:,j+1)*( x_real(:,i,j) + x_real(:,i,j+2) ) + Q_M_real(:,j+1);
                    y_real(:,i+1,j+1) = (1/2)*P_real(:,:,j+1)*( y_real(:,i,j) + y_real(:,i,j+2) ) + Q_k_real(:,j+1);
                end
                x_real(:,i+1,n) = (1/1)*P_real(:,:,n)*( x_real(:,i,n-1) ) + Q_M_real(:,n);
                y_real(:,i+1,n) = (1/1)*P_real(:,:,n)*( y_real(:,i,n-1) ) + Q_k_real(:,n);
                
                % Internal convergence check with break for large networks
                if i>2
                    for k=1:n
                        x_real_max(k,i) = max( abs(x_real(k,i,:) - x_real(k,i-2,:) ) );
                        x_real_min(k,i) = min( abs(x_real(k,i,:) - x_real(k,i-2,:) ) );
                        x_real_max_min(k,i) = x_real_max(k,i) - x_real_min(k,i);
                        y_real_max(k,i) = max( abs(y_real(k,i,:) - y_real(k,i-2,:) )  );
                        y_real_min(k,i) = min( abs(y_real(k,i,:) - y_real(k,i-2,:) )  );
                        y_real_max_min(k,i) = y_real_max(k,i) - y_real_min(k,i);
                        if x_real_max_min(k,i) < thres_dale_max_min &&...
                                y_real_max_min(k,i) < thres_dale_max_min
                            idx_dale_max_min(k) = 1;
                        else
                            idx_dale_max_min(k) = 0;
                        end
                    end
                    if sum(idx_dale_max_min) == n
                        iter_dale_convrg_max_min(k_dale_max_min,cov) = i+1;
                        k_dale_max_min = k_dale_max_min + 1;
                        if k_dale_max_min > 3
                            break;
                        end
                    end
                    idx_dale_max_min = [];
                end
            end
        end
        
        % DALE for augmented path graph topology
        if n_all == 2
            for i=1:iter_2
                x_real_2_hop(:,i+1,1) = (1/1)*P_real(:,:,1)*( x_real_2_hop(:,i,2) ) + Q_M_real(:,1);
                x_real_2_hop(:,i+1,2) = (1/1)*P_real(:,:,2)*( x_real_2_hop(:,i,1) ) + Q_M_real(:,2);
                y_real_2_hop(:,i+1,1) = (1/1)*P_real(:,:,1)*( y_real_2_hop(:,i,2) ) + Q_k_real(:,1);
                y_real_2_hop(:,i+1,2) = (1/1)*P_real(:,:,2)*( y_real_2_hop(:,i,1) ) + Q_k_real(:,2);
            end
            
        elseif n_all==4
            if n <=3
                for i=1:iter_2
                    x_real_2_hop(:,i+1,1) = (1/1)*P_real(:,:,1)*( x_real_2_hop(:,i,2) ) + Q_M_real(:,1);
                    y_real_2_hop(:,i+1,1) = (1/1)*P_real(:,:,1)*( y_real_2_hop(:,i,2) ) + Q_k_real(:,1);
                    for j=1:n-2
                        x_real_2_hop(:,i+1,j+1) = (1/2)*P_real(:,:,j+1)*( x_real_2_hop(:,i,j) + x_real_2_hop(:,i,j+2) ) + Q_M_real(:,j+1);
                        y_real_2_hop(:,i+1,j+1) = (1/2)*P_real(:,:,j+1)*( y_real_2_hop(:,i,j) + y_real_2_hop(:,i,j+2) ) + Q_k_real(:,j+1);
                    end
                    x_real_2_hop(:,i+1,n) = (1/1)*P_real(:,:,n)*( x_real_2_hop(:,i,n-1) ) + Q_M_real(:,n);
                    y_real_2_hop(:,i+1,n) = (1/1)*P_real(:,:,n)*( y_real_2_hop(:,i,n-1) ) + Q_k_real(:,n);
                end
                
            else
                for i=1:iter_2
                    x_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( x_real_2_hop(:,i,2) + x_real_2_hop(:,i,3) ) + Q_M_real(:,1);
                    y_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( y_real_2_hop(:,i,2) + y_real_2_hop(:,i,3) ) + Q_k_real(:,1);
                    
                    x_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( x_real_2_hop(:,i,1) + x_real_2_hop(:,i,3) + x_real_2_hop(:,i,4) ) + Q_M_real(:,2);
                    y_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( y_real_2_hop(:,i,1) + y_real_2_hop(:,i,3) + y_real_2_hop(:,i,4) ) + Q_k_real(:,2);
                    
                    x_real_2_hop(:,i+1,3) = (1/3)*P_real(:,:,3)*( x_real_2_hop(:,i,1) + x_real_2_hop(:,i,2) + x_real_2_hop(:,i,4) ) + Q_M_real(:,3);
                    y_real_2_hop(:,i+1,3) = (1/3)*P_real(:,:,3)*( y_real_2_hop(:,i,1) + y_real_2_hop(:,i,2) + y_real_2_hop(:,i,4) ) + Q_k_real(:,3);
                    
                    x_real_2_hop(:,i+1,4) = (1/2)*P_real(:,:,4)*( x_real_2_hop(:,i,3) + x_real_2_hop(:,i,2)) + Q_M_real(:,4);
                    y_real_2_hop(:,i+1,4) = (1/2)*P_real(:,:,4)*( y_real_2_hop(:,i,3) + y_real_2_hop(:,i,2)) + Q_k_real(:,4);
                end
            end
            
        elseif n_all ==5
            if n <=3
                for i=1:iter_2
                    x_real_2_hop(:,i+1,1) = (1/1)*P_real(:,:,1)*( x_real_2_hop(:,i,2) ) + Q_M_real(:,1);
                    y_real_2_hop(:,i+1,1) = (1/1)*P_real(:,:,1)*( y_real_2_hop(:,i,2) ) + Q_k_real(:,1);
                    for j=1:n-2
                        x_real_2_hop(:,i+1,j+1) = (1/2)*P_real(:,:,j+1)*( x_real_2_hop(:,i,j) + x_real_2_hop(:,i,j+2) ) + Q_M_real(:,j+1);
                        y_real_2_hop(:,i+1,j+1) = (1/2)*P_real(:,:,j+1)*( y_real_2_hop(:,i,j) + y_real_2_hop(:,i,j+2) ) + Q_k_real(:,j+1);
                    end
                    x_real_2_hop(:,i+1,n) = (1/1)*P_real(:,:,n)*( x_real_2_hop(:,i,n-1) ) + Q_M_real(:,n);
                    y_real_2_hop(:,i+1,n) = (1/1)*P_real(:,:,n)*( y_real_2_hop(:,i,n-1) ) + Q_k_real(:,n);
                end
                
            elseif n==4
                for i=1:iter_2
                    x_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( x_real_2_hop(:,i,2) + x_real_2_hop(:,i,3) ) + Q_M_real(:,1);
                    y_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( y_real_2_hop(:,i,2) + y_real_2_hop(:,i,3) ) + Q_k_real(:,1);
                    
                    x_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( x_real_2_hop(:,i,1) + x_real_2_hop(:,i,3) + x_real_2_hop(:,i,4) ) + Q_M_real(:,2);
                    y_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( y_real_2_hop(:,i,1) + y_real_2_hop(:,i,3) + y_real_2_hop(:,i,4) ) + Q_k_real(:,2);
                    
                    x_real_2_hop(:,i+1,3) = (1/3)*P_real(:,:,3)*( x_real_2_hop(:,i,1) + x_real_2_hop(:,i,2) + x_real_2_hop(:,i,4) ) + Q_M_real(:,3);
                    y_real_2_hop(:,i+1,3) = (1/3)*P_real(:,:,3)*( y_real_2_hop(:,i,1) + y_real_2_hop(:,i,2) + y_real_2_hop(:,i,4) ) + Q_k_real(:,3);
                    
                    x_real_2_hop(:,i+1,4) = (1/2)*P_real(:,:,4)*( x_real_2_hop(:,i,3) + x_real_2_hop(:,i,2)) + Q_M_real(:,4);
                    y_real_2_hop(:,i+1,4) = (1/2)*P_real(:,:,4)*( y_real_2_hop(:,i,3) + y_real_2_hop(:,i,2)) + Q_k_real(:,4);
                end
                
            else
                for i=1:iter_2
                    x_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( x_real_2_hop(:,i,2) + x_real_2_hop(:,i,3) ) + Q_M_real(:,1);
                    y_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( y_real_2_hop(:,i,2) + y_real_2_hop(:,i,3) ) + Q_k_real(:,1);
                    
                    x_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( x_real_2_hop(:,i,1) + x_real_2_hop(:,i,3) + x_real_2_hop(:,i,4) ) + Q_M_real(:,2);
                    y_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( y_real_2_hop(:,i,1) + y_real_2_hop(:,i,3) + y_real_2_hop(:,i,4) ) + Q_k_real(:,2);
                    
                    x_real_2_hop(:,i+1,3) = (1/4)*P_real(:,:,3)*( x_real_2_hop(:,i,1) + x_real_2_hop(:,i,2) + x_real_2_hop(:,i,4) + x_real_2_hop(:,i,5) ) + Q_M_real(:,3);
                    y_real_2_hop(:,i+1,3) = (1/4)*P_real(:,:,3)*( y_real_2_hop(:,i,1) + y_real_2_hop(:,i,2) + y_real_2_hop(:,i,4) + y_real_2_hop(:,i,5) ) + Q_k_real(:,3);
                    
                    x_real_2_hop(:,i+1,4) = (1/3)*P_real(:,:,4)*( x_real_2_hop(:,i,2) + x_real_2_hop(:,i,3) + x_real_2_hop(:,i,5)) + Q_M_real(:,4);
                    y_real_2_hop(:,i+1,4) = (1/3)*P_real(:,:,4)*( y_real_2_hop(:,i,2) + y_real_2_hop(:,i,3) + y_real_2_hop(:,i,5)) + Q_k_real(:,4);
                    
                    x_real_2_hop(:,i+1,5) = (1/2)*P_real(:,:,5)*( x_real_2_hop(:,i,3) + x_real_2_hop(:,i,5)) + Q_M_real(:,5);
                    y_real_2_hop(:,i+1,5) = (1/2)*P_real(:,:,5)*( y_real_2_hop(:,i,3) + y_real_2_hop(:,i,5)) + Q_k_real(:,5);
                end
            end
            
        else
            if n <=3
                for i=1:iter_2
                    x_real_2_hop(:,i+1,1) = (1/1)*P_real(:,:,1)*( x_real_2_hop(:,i,2) ) + Q_M_real(:,1);
                    y_real_2_hop(:,i+1,1) = (1/1)*P_real(:,:,1)*( y_real_2_hop(:,i,2) ) + Q_k_real(:,1);
                    for j=1:n-2
                        x_real_2_hop(:,i+1,j+1) = (1/2)*P_real(:,:,j+1)*( x_real_2_hop(:,i,j) + x_real_2_hop(:,i,j+2) ) + Q_M_real(:,j+1);
                        y_real_2_hop(:,i+1,j+1) = (1/2)*P_real(:,:,j+1)*( y_real_2_hop(:,i,j) + y_real_2_hop(:,i,j+2) ) + Q_k_real(:,j+1);
                    end
                    x_real_2_hop(:,i+1,n) = (1/1)*P_real(:,:,n)*( x_real_2_hop(:,i,n-1) ) + Q_M_real(:,n);
                    y_real_2_hop(:,i+1,n) = (1/1)*P_real(:,:,n)*( y_real_2_hop(:,i,n-1) ) + Q_k_real(:,n);
                end
                
            elseif n==4
                for i=1:iter_2
                    x_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( x_real_2_hop(:,i,2) + x_real_2_hop(:,i,3) ) + Q_M_real(:,1);
                    y_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( y_real_2_hop(:,i,2) + y_real_2_hop(:,i,3) ) + Q_k_real(:,1);
                    
                    x_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( x_real_2_hop(:,i,1) + x_real_2_hop(:,i,3) + x_real_2_hop(:,i,4) ) + Q_M_real(:,2);
                    y_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( y_real_2_hop(:,i,1) + y_real_2_hop(:,i,3) + y_real_2_hop(:,i,4) ) + Q_k_real(:,2);
                    
                    x_real_2_hop(:,i+1,3) = (1/3)*P_real(:,:,3)*( x_real_2_hop(:,i,1) + x_real_2_hop(:,i,2) + x_real_2_hop(:,i,4) ) + Q_M_real(:,3);
                    y_real_2_hop(:,i+1,3) = (1/3)*P_real(:,:,3)*( y_real_2_hop(:,i,1) + y_real_2_hop(:,i,2) + y_real_2_hop(:,i,4) ) + Q_k_real(:,3);
                    
                    x_real_2_hop(:,i+1,4) = (1/2)*P_real(:,:,4)*( x_real_2_hop(:,i,3) + x_real_2_hop(:,i,2)) + Q_M_real(:,4);
                    y_real_2_hop(:,i+1,4) = (1/2)*P_real(:,:,4)*( y_real_2_hop(:,i,3) + y_real_2_hop(:,i,2)) + Q_k_real(:,4);
                end
                
            elseif n==5
                for i=1:iter_2
                    x_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( x_real_2_hop(:,i,2) + x_real_2_hop(:,i,3) ) + Q_M_real(:,1);
                    y_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( y_real_2_hop(:,i,2) + y_real_2_hop(:,i,3) ) + Q_k_real(:,1);
                    
                    x_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( x_real_2_hop(:,i,1) + x_real_2_hop(:,i,3) + x_real_2_hop(:,i,4) ) + Q_M_real(:,2);
                    y_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( y_real_2_hop(:,i,1) + y_real_2_hop(:,i,3) + y_real_2_hop(:,i,4) ) + Q_k_real(:,2);
                    
                    x_real_2_hop(:,i+1,3) = (1/4)*P_real(:,:,3)*( x_real_2_hop(:,i,1) + x_real_2_hop(:,i,2) + x_real_2_hop(:,i,4) + x_real_2_hop(:,i,5) ) + Q_M_real(:,3);
                    y_real_2_hop(:,i+1,3) = (1/4)*P_real(:,:,3)*( y_real_2_hop(:,i,1) + y_real_2_hop(:,i,2) + y_real_2_hop(:,i,4) + y_real_2_hop(:,i,5) ) + Q_k_real(:,3);
                    
                    x_real_2_hop(:,i+1,4) = (1/3)*P_real(:,:,4)*( x_real_2_hop(:,i,2) + x_real_2_hop(:,i,3) + x_real_2_hop(:,i,5)) + Q_M_real(:,4);
                    y_real_2_hop(:,i+1,4) = (1/3)*P_real(:,:,4)*( y_real_2_hop(:,i,2) + y_real_2_hop(:,i,3) + y_real_2_hop(:,i,5)) + Q_k_real(:,4);
                    
                    x_real_2_hop(:,i+1,5) = (1/2)*P_real(:,:,5)*( x_real_2_hop(:,i,3) + x_real_2_hop(:,i,5)) + Q_M_real(:,5);
                    y_real_2_hop(:,i+1,5) = (1/2)*P_real(:,:,5)*( y_real_2_hop(:,i,3) + y_real_2_hop(:,i,5)) + Q_k_real(:,5);
                end
                
            else
                k_dale_max_min_2_hop = 1;
                for i=1:iter_2
                    x_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( x_real_2_hop(:,i,2) + x_real_2_hop(:,i,3) ) + Q_M_real(:,1);
                    y_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( y_real_2_hop(:,i,2) + y_real_2_hop(:,i,3) ) + Q_k_real(:,1);
                    
                    x_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( x_real_2_hop(:,i,1) + x_real_2_hop(:,i,3) + x_real_2_hop(:,i,4) ) + Q_M_real(:,2);
                    y_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( y_real_2_hop(:,i,1) + y_real_2_hop(:,i,3) + y_real_2_hop(:,i,4) ) + Q_k_real(:,2);
                    
                    for j=2:n-3
                        x_real_2_hop(:,i+1,j+1) = (1/4)*P_real(:,:,j+1)*( x_real_2_hop(:,i,j-1) + x_real_2_hop(:,i,j) + x_real_2_hop(:,i,j+2) + x_real_2_hop(:,i,j+3) ) + Q_M_real(:,j+1);
                        y_real_2_hop(:,i+1,j+1) = (1/4)*P_real(:,:,j+1)*( y_real_2_hop(:,i,j-1) + y_real_2_hop(:,i,j) + y_real_2_hop(:,i,j+2) + y_real_2_hop(:,i,j+3) ) + Q_k_real(:,j+1);
                    end
                    
                    x_real_2_hop(:,i+1,n-1) = (1/3)*P_real(:,:,n-1)*( x_real_2_hop(:,i,n) + x_real_2_hop(:,i,n-2) + x_real_2_hop(:,i,n-3) ) + Q_M_real(:,n-1);
                    y_real_2_hop(:,i+1,n-1) = (1/3)*P_real(:,:,n-1)*( y_real_2_hop(:,i,n) + y_real_2_hop(:,i,n-2) + y_real_2_hop(:,i,n-3) ) + Q_k_real(:,n-1);
                    
                    x_real_2_hop(:,i+1,n) = (1/2)*P_real(:,:,n)*( x_real_2_hop(:,i,n-1) + x_real_2_hop(:,i,n-2)) + Q_M_real(:,n);
                    y_real_2_hop(:,i+1,n) = (1/2)*P_real(:,:,n)*( y_real_2_hop(:,i,n-1) + y_real_2_hop(:,i,n-2)) + Q_k_real(:,n);
                    
                    % Internal convergence check with break for large networks
                    if i>2
                        for k=1:n
                            x_real_max_2_hop(k,i) = max( abs(x_real_2_hop(k,i,:) - x_real_2_hop(k,i-2,:) ) );
                            x_real_min_2_hop(k,i) = min( abs(x_real_2_hop(k,i,:) - x_real_2_hop(k,i-2,:) ) );
                            x_real_max_min_2_hop(k,i) = x_real_max_2_hop(k,i) - x_real_min_2_hop(k,i);
                            y_real_max_2_hop(k,i) = max( abs(y_real_2_hop(k,i,:) - y_real_2_hop(k,i-2,:) ) );
                            y_real_min_2_hop(k,i) = min( abs(y_real_2_hop(k,i,:) - y_real_2_hop(k,i-2,:) ) );
                            y_real_max_min_2_hop(k,i) = y_real_max_2_hop(k,i) - y_real_min_2_hop(k,i);
                            if x_real_max_min_2_hop(k,i) < thres_dale_max_min &&...
                                    y_real_max_min_2_hop(k,i) < thres_dale_max_min
                                idx_dale_max_min_2_hop(k) = 1;
                            else
                                idx_dale_max_min_2_hop(k) = 0;
                            end
                        end
                        if sum(idx_dale_max_min_2_hop) == n
                            iter_dale_convrg_max_min_2_hop(k_dale_max_min_2_hop,cov) = i+1;
                            k_dale_max_min_2_hop = k_dale_max_min_2_hop + 1;
                            if k_dale_max_min_2_hop > 3
                                break;
                            end
                        end
                        idx_dale_max_min_2_hop = [];
                    end
                end
            end
        end
        
        % External convergence check for small networks
        k_dale_max_min = 1;
        if n==2
            for i=3:iter_1
                for k=1:n
                    x_real_max(k,i) = max( abs(x_real(k,i,:) - x_real(k,i-2,:) ) );
                    x_real_min(k,i) = min( abs(x_real(k,i,:) - x_real(k,i-2,:) ) );
                    x_real_max_min(k,i) = x_real_max(k,i) - x_real_min(k,i);
                    y_real_max(k,i) = max( abs(y_real(k,i,:) - y_real(k,i-2,:) )  );
                    y_real_min(k,i) = min( abs(y_real(k,i,:) - y_real(k,i-2,:) )  );
                    y_real_max_min(k,i) = y_real_max(k,i) - y_real_min(k,i);
                    if x_real_max_min(k,i) < thres_dale_max_min &&...
                            y_real_max_min(k,i) < thres_dale_max_min
                        idx_dale_max_min(k) = 1;
                    else
                        idx_dale_max_min(k) = 0;
                    end
                end
                if sum(idx_dale_max_min) == n
                    iter_dale_convrg_max_min(k_dale_max_min,cov) = i+1;
                    k_dale_max_min = k_dale_max_min + 1;
                    if k_dale_max_min > 3
                        break;
                    end
                end
                idx_dale_max_min = [];
            end
        end
        if n<=5
            k_dale_max_min_2_hop = 1;
            for i=3:iter_2
                for k=1:n
                    x_real_max_2_hop(k,i) = max( abs(x_real_2_hop(k,i,:) - x_real_2_hop(k,i-2,:) ) );
                    x_real_min_2_hop(k,i) = min( abs(x_real_2_hop(k,i,:) - x_real_2_hop(k,i-2,:) ) );
                    x_real_max_min_2_hop(k,i) = x_real_max_2_hop(k,i) - x_real_min_2_hop(k,i);
                    y_real_max_2_hop(k,i) = max( abs(y_real_2_hop(k,i,:) - y_real_2_hop(k,i-2,:) ) );
                    y_real_min_2_hop(k,i) = min( abs(y_real_2_hop(k,i,:) - y_real_2_hop(k,i-2,:) ) );
                    y_real_max_min_2_hop(k,i) = y_real_max_2_hop(k,i) - y_real_min_2_hop(k,i);
                    if x_real_max_min_2_hop(k,i) < thres_dale_max_min &&...
                            y_real_max_min_2_hop(k,i) < thres_dale_max_min
                        idx_dale_max_min_2_hop(k) = 1;
                    else
                        idx_dale_max_min_2_hop(k) = 0;
                    end
                end
                if sum(idx_dale_max_min_2_hop) == n
                    iter_dale_convrg_max_min_2_hop(k_dale_max_min_2_hop,cov) = i+1;
                    k_dale_max_min_2_hop = k_dale_max_min_2_hop + 1;
                    if k_dale_max_min_2_hop > 3
                        break;
                    end
                end
                idx_dale_max_min_2_hop = [];
            end
        end
        
        % Final mean and variance computation
        for i=1:n
            x_last_vec(:,i) = (x_real(:,iter_dale_convrg_max_min(1,cov)-1,i)+x_real(:,iter_dale_convrg_max_min(1,cov),i))./2;
            y_last_vec(:,i) = (y_real(:,iter_dale_convrg_max_min(1,cov)-1,i)+y_real(:,iter_dale_convrg_max_min(1,cov),i))./2;
            
            x_last_2_hop_vec(:,i) = (x_real_2_hop(:,iter_dale_convrg_max_min_2_hop(1,cov)-1,i)+x_real_2_hop(:,iter_dale_convrg_max_min_2_hop(1,cov),i))./2;
            y_last_2_hop_vec(:,i) = (y_real_2_hop(:,iter_dale_convrg_max_min_2_hop(1,cov)-1,i)+y_real_2_hop(:,iter_dale_convrg_max_min_2_hop(1,cov),i))./2;
        end
        
        x_last = sum(x_last_vec,2)./n;
        y_last = sum(y_last_vec,2)./n;
        
        x_last_2_hop = sum(x_last_2_hop_vec,2)./n;
        y_last_2_hop = sum(y_last_2_hop_vec,2)./n;
        
        mu(cov,1) = b_k_real'*x_last*models{1}.Y_std + models{1}.Y_mean;
        s2(cov,1) = (kss(cov) - b_k_real'*y_last + exp(2*hyp_lik))*(models{1}.Y_std)^2;
        
        mu_2_hop(cov,1) = b_k_real'*x_last_2_hop*models{1}.Y_std + models{1}.Y_mean;
        s2_2_hop(cov,1) = (kss(cov) - b_k_real'*y_last_2_hop + exp(2*hyp_lik))*(models{1}.Y_std)^2;
        
    else
        % Final mean and variance computation for 1 agent
        mu(cov,1) = b_M(non_zero.index,cov)*models{1}.Y_std + models{1}.Y_mean;
        s2(cov,1) = s2_all(non_zero.index,cov)*(models{1}.Y_std)^2;
        
        mu_2_hop(cov,1) = mu(cov,1);
        s2_2_hop(cov,1) = s2(cov,1);
        
        iter_dale_convrg_max_min(1,cov) = 0;
        iter_dale_convrg_max_min_2_hop(1,cov) = 0;
    end
    
    non_zero.index = [];
    A_real = [];
    b_M_real = [];
    b_k_real = [];
    R_real = [];
    P_real = [];
    Q_M_real = [];
    Q_k_real = [];
    x_real = [];
    y_real = [];
    x_last_vec = [];
    y_last_vec = [];
    x_last = [];
    y_last = [];
    x_real_2_hop = [];
    y_real_2_hop = [];
    x_last_2_hop_vec = [];
    y_last_2_hop_vec = [];
    x_last_2_hop = [];
    y_last_2_hop = [];
    n=n_all;
end
end