% A. Gusrialdi and Z. Qu, "Distributed Estimation of All the Eigenvalues and...
% Eigenvectors of Matrices Associated With Strongly Connected Digraphs,"...
% IEEE Control Systems Letters, 2017
%%
clear all; close all; clc;
%%
Q = [1 -1 0 0; 0 1 0 -1; -1 -1 2 0; 0 0 -1 1]; % graph Laplacian
c = 5; % positive scalar
Q_bar = Q + c*eye(4); % equation (4)

Q_bar_inv = inv(Q_bar);

B = eig(Q)

iter = 60;

Z_1(:,:,1) = [0 0 0 0; -1 0 0 0; 0 0 0 0; 0 0 0 0]; % Input 2, thus (2,1)
Z_2(:,:,1) = [0 0 0 0; 0 0 0 0; 0 0 0 0; 0 -1 0 0]; % Input 4, thus (4,2)
Z_3(:,:,1) = [0 0 -1 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]; % Input 1,3 thus (1,3)
Z_4(:,:,1) = [0 0 0 0; 0 0 0 0; 0 0 0 -1; 0 0 0 0]; % Input 3, thus (3,4)
P_1 = eye(4) - Q_bar(1,:)'*inv(Q_bar(1,:)*Q_bar(1,:)')*Q_bar(1,:); % equation (10)
P_2 = eye(4) - Q_bar(2,:)'*inv(Q_bar(2,:)*Q_bar(2,:)')*Q_bar(2,:); % equation (10)
P_3 = eye(4) - Q_bar(3,:)'*inv(Q_bar(3,:)*Q_bar(3,:)')*Q_bar(3,:); % equation (10)
P_4 = eye(4) - Q_bar(4,:)'*inv(Q_bar(4,:)*Q_bar(4,:)')*Q_bar(4,:); % equation (10)

%%
for i = 1:iter

    Z_1(:,:,i+1) = Z_1(:,:,i) - P_1*(Z_1(:,:,i) - Z_2(:,:,i)); % equation (9)
    Z_2(:,:,i+1) = Z_2(:,:,i) - P_2*(Z_2(:,:,i) - Z_4(:,:,i)); % equation (9)
    Z_3(:,:,i+1) = Z_3(:,:,i) - P_3*(Z_3(:,:,i) - .5*( Z_1(:,:,i) + Z_2(:,:,i) ) ); % equation (9)
    Z_4(:,:,i+1) = Z_4(:,:,i) - P_4*(Z_4(:,:,i) - Z_3(:,:,i)); % equation (9)
    
end

Z_1_eig = 1./eig(Z_1(:,:,end)) - c % equation (11)
Z_2_eig = 1./eig(Z_1(:,:,end)) - c % equation (11)
Z_3_eig = 1./eig(Z_1(:,:,end)) - c % equation (11)
Z_4_eig = 1./eig(Z_1(:,:,end)) - c % equation (11)