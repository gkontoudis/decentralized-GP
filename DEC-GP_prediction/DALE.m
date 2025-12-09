function [mu,s2,mu_2_hop,s2_2_hop,...
    iter_dale_convrg_mu_1,iter_dale_convrg_mu_2,iter_dale_convrg_sigma_1,iter_dale_convrg_sigma_2]...
    = dale (nt,A,iter_dale,opts,b_M,b_k,kss,models,hyp_lik,...
    thres_dale_mu_1,thres_dale_mu_2,thres_dale_sigma_1,thres_dale_sigma_2)

% dale(nt,K_M_x,iter_dale,opts,mu_all,k_M_x,kss,models,hyp_lik,...
%    thres_dale_mu_1,thres_dale_mu_2,thres_dale_sigma_1,thres_dale_sigma_2)
% b_M = mu_all';
% A = K_M_x;
% b_k = k_M_x;

n = opts.Ms;
iter = iter_dale;
n_all = n;
b_M = b_M';
if n == 2
    thres = eps; % to ensure n>=2 for DALE
else
    thres = 0.01;
end

JOR_error_vec = [];
DALE_error_vec = [];
l = 1;

for cov=1:nt
%     cov = 1;
    for i =1:n
        R(:,i) = A(i,:,cov)'*inv( A(i,:,cov)*A(i,:,cov)' );
        P(:,:,i) = eye(n) - (R(:,i)*A(i,:,cov) ) ;
        Q_M(:,i) = R(:,i)*b_M(i,cov);
        Q_k(:,i) = R(:,i)*b_k(i,cov);
        x_dale(:,1,i) = ones(n,1);
        y_dale(:,1,i) = ones(n,1);
        for j=1:n
            x_dale(j,1,i) = b_M(j,cov)/A(j,j,cov);
            y_dale(j,1,i) = b_k(j,cov)/A(j,j,cov);
        end
    end
    
    non_zero.index = find(b_k(:,cov)>thres);
    for z_dale=1:n_all
        if length(non_zero.index) == z_dale
            n = z_dale;
        end
    end
    if n_all>4
        if n ==1
            non_zero.index = find(b_k(:,cov)>(thres-.005));
            for z_dale=1:n_all
                if length(non_zero.index) == z_dale
                    n = z_dale;
                end
            end
        end
    else
        if n ==1
            non_zero.index = find(b_k(:,cov)>(thres-.0095555));
            for z_dale=1:n_all
                if length(non_zero.index) == z_dale
                    n = z_dale;
                end
            end
        end
    end
        
    
    A_real = A(non_zero.index(1):non_zero.index(end),...
        non_zero.index(1):non_zero.index(end),cov);
    b_M_real = b_M(non_zero.index(1):non_zero.index(end),cov);
    b_k_real =b_k(non_zero.index(1):non_zero.index(end),cov);
    
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
    
    if n_all == 2
        for i=1:iter
            x_real(:,i+1,1) = (1/1)*P_real(:,:,1)*( x_real(:,i,2) ) + Q_M_real(:,1);
            x_real(:,i+1,2) = (1/1)*P_real(:,:,2)*( x_real(:,i,1) ) + Q_M_real(:,2);
            y_real(:,i+1,1) = (1/1)*P_real(:,:,1)*( y_real(:,i,2) ) + Q_k_real(:,1);
            y_real(:,i+1,2) = (1/1)*P_real(:,:,2)*( y_real(:,i,1) ) + Q_k_real(:,2);
        end
    else
        for i=1:iter
            %     if i < 15
            x_real(:,i+1,1) = (1/1)*P_real(:,:,1)*( x_real(:,i,2) ) + Q_M_real(:,1);
            y_real(:,i+1,1) = (1/1)*P_real(:,:,1)*( y_real(:,i,2) ) + Q_k_real(:,1);
            for j=1:n-2
                x_real(:,i+1,j+1) = (1/2)*P_real(:,:,j+1)*( x_real(:,i,j) + x_real(:,i,j+2) ) + Q_M_real(:,j+1);
                y_real(:,i+1,j+1) = (1/2)*P_real(:,:,j+1)*( y_real(:,i,j) + y_real(:,i,j+2) ) + Q_k_real(:,j+1);
            end
            x_real(:,i+1,n) = (1/1)*P_real(:,:,n)*( x_real(:,i,n-1) ) + Q_M_real(:,n);
            y_real(:,i+1,n) = (1/1)*P_real(:,:,n)*( y_real(:,i,n-1) ) + Q_k_real(:,n);
        end
    end
    
    
    if n_all == 2
        for i=1:iter
            x_real_2_hop(:,i+1,1) = (1/1)*P_real(:,:,1)*( x_real_2_hop(:,i,2) ) + Q_M_real(:,1);
            x_real_2_hop(:,i+1,2) = (1/1)*P_real(:,:,2)*( x_real_2_hop(:,i,1) ) + Q_M_real(:,2);
            y_real_2_hop(:,i+1,1) = (1/1)*P_real(:,:,1)*( y_real_2_hop(:,i,2) ) + Q_k_real(:,1);
            y_real_2_hop(:,i+1,2) = (1/1)*P_real(:,:,2)*( y_real_2_hop(:,i,1) ) + Q_k_real(:,2);
        end
        
    elseif n_all==4
        if n <=3
            for i=1:iter
                %     if i < 15
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
            for i=1:iter
                %     if i < 15
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
            for i=1:iter
                %     if i < 15
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
            for i=1:iter
                %     if i < 15
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
            for i=1:iter
                
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
        
    else %if n_all ==8
        if n <=3
            for i=1:iter
                %     if i < 15
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
            for i=1:iter
                %     if i < 15
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
            for i=1:iter
                
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
%             if n <=3
%                 for i=1:iter
%                     %     if i < 15
%                     x_real_2_hop(:,i+1,1) = (1/1)*P_real(:,:,1)*( x_real_2_hop(:,i,2) ) + Q_M_real(:,1);
%                     y_real_2_hop(:,i+1,1) = (1/1)*P_real(:,:,1)*( y_real_2_hop(:,i,2) ) + Q_k_real(:,1);
%                     for j=1:n-2
%                         x_real_2_hop(:,i+1,j+1) = (1/2)*P_real(:,:,j+1)*( x_real_2_hop(:,i,j) + x_real_2_hop(:,i,j+2) ) + Q_M_real(:,j+1);
%                         y_real_2_hop(:,i+1,j+1) = (1/2)*P_real(:,:,j+1)*( y_real_2_hop(:,i,j) + y_real_2_hop(:,i,j+2) ) + Q_k_real(:,j+1);
%                     end
%                     x_real_2_hop(:,i+1,n) = (1/1)*P_real(:,:,n)*( x_real_2_hop(:,i,n-1) ) + Q_M_real(:,n);
%                     y_real_2_hop(:,i+1,n) = (1/1)*P_real(:,:,n)*( y_real_2_hop(:,i,n-1) ) + Q_k_real(:,n);
%                 end
%                 
%             elseif n==4
%                 for i=1:iter
%                     %     if i < 15
%                     x_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( x_real_2_hop(:,i,2) + x_real_2_hop(:,i,3) ) + Q_M_real(:,1);
%                     y_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( y_real_2_hop(:,i,2) + y_real_2_hop(:,i,3) ) + Q_k_real(:,1);
%                     
%                     x_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( x_real_2_hop(:,i,1) + x_real_2_hop(:,i,3) + x_real_2_hop(:,i,4) ) + Q_M_real(:,2);
%                     y_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( y_real_2_hop(:,i,1) + y_real_2_hop(:,i,3) + y_real_2_hop(:,i,4) ) + Q_k_real(:,2);
%                     
%                     x_real_2_hop(:,i+1,3) = (1/3)*P_real(:,:,3)*( x_real_2_hop(:,i,1) + x_real_2_hop(:,i,2) + x_real_2_hop(:,i,4) ) + Q_M_real(:,3);
%                     y_real_2_hop(:,i+1,3) = (1/3)*P_real(:,:,3)*( y_real_2_hop(:,i,1) + y_real_2_hop(:,i,2) + y_real_2_hop(:,i,4) ) + Q_k_real(:,3);
%                     
%                     x_real_2_hop(:,i+1,4) = (1/2)*P_real(:,:,4)*( x_real_2_hop(:,i,3) + x_real_2_hop(:,i,2)) + Q_M_real(:,4);
%                     y_real_2_hop(:,i+1,4) = (1/2)*P_real(:,:,4)*( y_real_2_hop(:,i,3) + y_real_2_hop(:,i,2)) + Q_k_real(:,4);
%                 end
                
%             else
                for i=1:iter
                    %     if i < 15
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
                end
            end
    

        
%     else
%         for i=1:iter
%             %     if i < 15
%             x_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( x_real_2_hop(:,i,2) + x_real_2_hop(:,i,3) ) + Q_M_real(:,1);
%             y_real_2_hop(:,i+1,1) = (1/2)*P_real(:,:,1)*( y_real_2_hop(:,i,2) + y_real_2_hop(:,i,3) ) + Q_k_real(:,1);
%             
%             x_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( x_real_2_hop(:,i,1) + x_real_2_hop(:,i,3) + x_real_2_hop(:,i,4) ) + Q_M_real(:,2);
%             y_real_2_hop(:,i+1,2) = (1/3)*P_real(:,:,2)*( y_real_2_hop(:,i,1) + y_real_2_hop(:,i,3) + y_real_2_hop(:,i,4) ) + Q_k_real(:,2);
%             
%             for j=2:n-3
%                 x_real_2_hop(:,i+1,j+1) = (1/4)*P_real(:,:,j+1)*( x_real_2_hop(:,i,j-1) + x_real_2_hop(:,i,j) + x_real_2_hop(:,i,j+2) + x_real_2_hop(:,i,j+3) ) + Q_M_real(:,j+1);
%                 y_real_2_hop(:,i+1,j+1) = (1/4)*P_real(:,:,j+1)*( y_real_2_hop(:,i,j-1) + y_real_2_hop(:,i,j) + y_real_2_hop(:,i,j+2) + y_real_2_hop(:,i,j+3) ) + Q_k_real(:,j+1);
%             end
%             
%             x_real_2_hop(:,i+1,n-1) = (1/3)*P_real(:,:,n-1)*( x_real_2_hop(:,i,n) + x_real_2_hop(:,i,n-2) + x_real_2_hop(:,i,n-3) ) + Q_M_real(:,n-1);
%             y_real_2_hop(:,i+1,n-1) = (1/3)*P_real(:,:,n-1)*( y_real_2_hop(:,i,n) + y_real_2_hop(:,i,n-2) + y_real_2_hop(:,i,n-3) ) + Q_k_real(:,n-1);
%             
%             x_real_2_hop(:,i+1,n) = (1/2)*P_real(:,:,n)*( x_real_2_hop(:,i,n-1) + x_real_2_hop(:,i,n-2)) + Q_M_real(:,n);
%             y_real_2_hop(:,i+1,n) = (1/2)*P_real(:,:,n)*( y_real_2_hop(:,i,n-1) + y_real_2_hop(:,i,n-2)) + Q_k_real(:,n);
%         end
    end
    k_dale_mu_1= 1;
    k_dale_sigma_1= 1;
    k_dale_mu_2 = 1;
    k_dale_sigma_2 = 1;
    for i =2:iter
        if sum(sum(abs(x_real(:,i,:) - x_real(:,i-1,:))))/n^2 < thres_dale_mu_1
            iter_dale_convrg_mu_1(k_dale_mu_1,cov) = i+1;
            k_dale_mu_1 = k_dale_mu_1+1;
        end
        if sum(sum(abs(y_real(:,i,:) - y_real(:,i-1,:))))/n^2 < thres_dale_sigma_1
            iter_dale_convrg_sigma_1(k_dale_sigma_1,cov) = i+1;
            k_dale_sigma_1 = k_dale_sigma_1+1;
        end
        
        if sum(sum(abs(x_real_2_hop(:,i,:) - x_real_2_hop(:,i-1,:))))/n^2 < thres_dale_mu_2
            iter_dale_convrg_mu_2(k_dale_mu_2,cov) = i+1;
            k_dale_mu_2 = k_dale_mu_2+1;
        end
        if sum(sum(abs(y_real_2_hop(:,i,:) - y_real_2_hop(:,i-1,:))))/n^2 < thres_dale_sigma_2
            iter_dale_convrg_sigma_2(k_dale_sigma_2,cov) = i+1;
            k_dale_sigma_2 = k_dale_sigma_2+1;
        end
    end
    
    if n_all >= 20
        
        if max(iter_dale_convrg_mu_1(1,:)) < 250
            iter_dale_convrg_mu_1(1,end) = 150 + max(iter_dale_convrg_mu_1(1,:));
        end
        
        if max(iter_dale_convrg_sigma_1(1,:)) < 250
            iter_dale_convrg_sigma_1(1,end) = 150 + max(iter_dale_convrg_sigma_1(1,:));
        end
        
        if max(iter_dale_convrg_mu_2(1,:)) < 200
            iter_dale_convrg_mu_2(1,end) = 200 + max(iter_dale_convrg_mu_2(1,:));
        end
        
        if max(iter_dale_convrg_sigma_2(1,:)) < 200
            iter_dale_convrg_sigma_2(1,end) = 200 + max(iter_dale_convrg_sigma_2(1,:));
        end
        
    end
    
    for i=1:n
        x_last_vec(:,i) = (x_real(:,max(iter_dale_convrg_mu_1(1,:))-1,i)+x_real(:,max(iter_dale_convrg_mu_1(1,:)),i))./2;
        y_last_vec(:,i) = (y_real(:,max(iter_dale_convrg_sigma_1(1,:))-1,i)+y_real(:,max(iter_dale_convrg_sigma_1(1,:)),i))./2;
        
        x_last_2_hop_vec(:,i) = (x_real_2_hop(:,max(iter_dale_convrg_mu_2(1,:))-1,i)+x_real_2_hop(:,max(iter_dale_convrg_mu_2(1,:)),i))./2;
        y_last_2_hop_vec(:,i) = (y_real_2_hop(:,max(iter_dale_convrg_sigma_2(1,:))-1,i)+y_real_2_hop(:,max(iter_dale_convrg_sigma_2(1,:)),i))./2;
    end
    x_last = sum(x_last_vec,2)./n;
    y_last = sum(y_last_vec,2)./n;
    
    x_last_2_hop = sum(x_last_2_hop_vec,2)./n;
    y_last_2_hop = sum(y_last_2_hop_vec,2)./n;
    
    mu(cov,1) = b_k_real'*x_last*models{1}.Y_std + models{1}.Y_mean;
    s2(cov,1) = (kss(cov) - b_k_real'*y_last + exp(2*hyp_lik))*(models{1}.Y_std)^2;
    
    mu_2_hop(cov,1) = b_k_real'*x_last_2_hop*models{1}.Y_std + models{1}.Y_mean;
    s2_2_hop(cov,1) = (kss(cov) - b_k_real'*y_last_2_hop + exp(2*hyp_lik))*(models{1}.Y_std)^2;
    

    
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





%%
%     for i=1:n
%         %         x(:,1,i) = b(i)./A(i,:);
%         %         x(:,1,i) = b(i)/A(i,i)*ones(n,1);
%         x(:,1,i) = 10*ones(n,1);
%         y(:,1,i) = 10*ones(n,1);
%         R(:,i) = A(i,:,j)'*inv( A(i,:,j)*A(i,:,j)' );
%         P(:,:,i) = eye(n) - (R(:,i)*A(i,:,j) ) ;
%         Q1(:,i) = R(:,i)*b1(j,i);
%         Q2(:,i) = R(:,i)*b2(i,j);
%     end
%
%     switch algo
%         case 'MLM'
%             for i=1:iter_dale
%                 x(:,i+1,1) = x(:,i,1) - (1/1)*P(:,:,1)*(   x(:,i,1) - x(:,i,2)            );
%                 for j=1:n-2
%                     x(:,i+1,j+1) = x(:,i,j+1) - (1/2)*P(:,:,j+1)*( 2*x(:,i,j+1) - (x(:,i,j) + x(:,i,j+2)) );
%                 end
%                 x(:,i+1,n) = x(:,i,n) - (1/1)*P(:,:,n)*(   x(:,i,n) - x(:,i,n-1)            );
%             end
%
%         case 'WMS'
%             for i=1:iter_dale
%                 x(:,i+1,1) = x(:,i,1) - (1/1)*P(:,:,1)*(   x(:,i,1) - x(:,i,2) ) + Q1(:,1) - R(:,1)*A(1,:)*x(:,i,1);
%                 for j=1:n-2
%                     x(:,i+1,j+1) = x(:,i,j+1) - (1/2)*P(:,:,j+1)*( 2*x(:,i,j+1) - (x(:,i,j) + x(:,i,j+2)) )+ Q1(:,j+1) - R(:,j+1)*A(j+1,:)*x(:,i,j+1);
%                 end
%                 x(:,i+1,n) = x(:,i,n) - (1/1)*P(:,:,n)*(   x(:,i,n) - x(:,i,n-1) )+ Q1(:,n) - R(:,n)*A(n,:)*x(:,i,n);
%             end
%
%         case 'LMM'
%             for i=1:iter_dale
%                 x(:,i+1,1) = (1/1)*P(:,:,1)*( x(:,i,2)            ) + Q1(:,1);
%                 y(:,i+1,1) = (1/1)*P(:,:,1)*( y(:,i,2)            ) + Q2(:,1);
%                 for k=1:n-2
%                     x(:,i+1,k+1) = (1/2)*P(:,:,k+1)*( x(:,i,k) + x(:,i,k+2) ) + Q1(:,k+1);
%                     y(:,i+1,k+1) = (1/2)*P(:,:,k+1)*( y(:,i,k) + y(:,i,k+2) ) + Q2(:,k+1);
%                 end
%                 x(:,i+1,n) = (1/1)*P(:,:,n)*( x(:,i,n-1)            ) + Q1(:,n);
%                 y(:,i+1,n) = (1/1)*P(:,:,n)*( y(:,i,n-1)            ) + Q2(:,n);
%             end
%             z_dist(j) =  b2(:,j)'*x(1,end,i)';
%             l_dist(j) =  b2(:,j)'*y(1,end,j)';
%
%
%             mu_dist(j) = opts.Ms*z_dist(j)*models{1}.Y_std + models{1}.Y_mean;
%             s2_dist(j) = (kss(j) - opts.Ms*l_dist(j) + exp(2*hyp_lik))*(models{1}.Y_std)^2;
%
%             x = [];
%     end
% end
% end