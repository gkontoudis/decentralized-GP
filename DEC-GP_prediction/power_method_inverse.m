function [eig_A_max,eig_A_min,iterations_pm_end] = power_method_inverse(A,thres_pm,iter_pm)

n = length(A);
x(:,1) = 2*(rand(1,n)-0.5); % unit circle values [-1,1]
for i=1:iter_pm
    p = A*x(:,i);
    n_max_p(i) = max(abs(p));
    x(:,i+1) = (1/n_max_p(i))*p;
    if i>=2
        if abs(n_max_p(i)-n_max_p(i-1))/abs(n_max_p(i)) < thres_pm
            iter_converge_max = i;
            break;
        end
    end
end

eig_A_max = n_max_p(end);

B = A-eig_A_max*eye(n);
z(:,1) = 2*(rand(1,n)-0.5); % unit circle values [-1,1]
for j=1:iter_pm

    p_z = B*z(:,j);
    
    n_max_p_z(j) = max(abs(p_z));

    z(:,j+1) = (1/n_max_p_z(j))*p_z;
    if j>=2
        if abs(n_max_p_z(j)-n_max_p_z(j-1))/abs(n_max_p_z(j)) < thres_pm
            iter_converge_min = j;
            break;
        end
    end
end
eig_A_min = abs(n_max_p_z(end) - n_max_p(end) ); % for B = A - lambda_max*I

iterations_pm_end = iter_converge_min + iter_converge_max;

end

