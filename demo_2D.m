%% #### -------------------------------------------------------------- #### 
%% George Kontoudis, g.kontoudis12@gmail.com www.georgekontoudis.com              
%% Virginia Tech, Bradley Department of Electrical & Computer Engineering                         
%% Decentralized Nested Gaussian Processes For Multi-Robot Systems     
%% ICRA 2021
%% Demo for a synthetic random field (2D) - Goldstein-Price function
%% #### -------------------------------------------------------------- ####  
clear all; close all; clc;
%% Test function
rng(99); % Control random number generator, Syntax rng(seed)
n = 1000; % number of training data
sn = 0.25; % variance of true process
% number of experts or agents - works for 2, 4, 5, 10, 20, 25, 40 agents
agents = n/200; 
nt = 1e-2*n; % number of prediction points 
        
gr = 1/1000; % grid resolution
[xx,xy] = meshgrid(0:gr:1, 0:gr:1);
f.fnc = goldpr(xx,xy);
f.avg = sum(sum(f.fnc))/(length(f.fnc)^2);

% plot test function
figure
[~,c]=contourf(xx,xy,f.fnc);hold on
surf(xx,xy,f.fnc);hold on
c.LineWidth = 1.5;
shading interp
set(gca,'fontsize',16)
set(gcf,'color','w')
cbar = colorbar;
hAx = gca;
hAx.CLim = [-2 2];
cbar.Limits = [-2 2];
view(0,90)
xlabel('x') ; ylabel('y') ;
title('Reference')
set(gca,'fontsize',16)
set(gcf,'color','w')
box on; grid off; hold off;
        
%% Training and Prediction Data
% training data
x = linspace(0,1,n)'; 
y = rand(n,1); % select randomly the y-coordinate of inputs
X = [x,y]; % inputs
z = goldpr(x,y)+sn*randn(n,1); % noisy data

% test data (prediction points)
xt = linspace(0,1,nt)'; 
yt = rand(nt,1); % select randomly the y-coordinate of test points
Xt = [xt,yt]; % test points
zt =  goldpr(xt,yt); % true values at test points
zt_normal = max(zt)-min(zt); % for normalized RMSPE

%% Graph Topologies
% complete
m6=25;
s_comp =[];
r_comp =[];
for i =1:agents
s_comp_add = i*ones(1,agents-i); 
s_comp = [s_comp s_comp_add];
r_comp_add = linspace(i+1,agents,agents-i);
r_comp = [r_comp r_comp_add];
end

% path graph
s_1 = 1:1:(agents-1) ; % source nodes
r_1 = 2:1:agents ; % target nodes

% augmented path graph
s_2 =[];
r_2 =[];
for i =1:agents-1
    if i == agents-1
        s_co = i;
    else
        s_co = i*ones(1,2);
    end
    s_2 = [s_2 s_co];
end
for i =2:agents
    if i == 2
        r_co = i;
    else
        r_co = i*ones(1,2);
    end
    r_2 = [r_2 r_co];
end

w_comp = ones(1, length(s_comp)); % weights
w_1 = ones(1, length(s_1)); 
w_2 = ones(1, length(s_2)); 
Acompgraph = graph(s_comp, r_comp, w_comp ); % create directed graph
A1graph = graph(s_1, r_1, w_1); 
A2graph = graph(s_2, r_2, w_2 );
D_compi = degree(Acompgraph); % in degree matrix
D_1i = degree(A1graph); 
D_2i = degree(A2graph); 
D_comp = diag(D_compi); % diagonal in-degree matrix
D_1 = diag(D_1i); 
D_2 = diag(D_2i); 
A_comp = full(adjacency(Acompgraph)); % adjacency matrix
A_1 = full(adjacency(A1graph)); 
A_2 = full(adjacency(A2graph)); 
L_comp = D_comp - A_comp; % Laplacian matrix
L_1 = D_1 - A_1; 
L_2 = D_2 - A_2; 
Delta_comp = max(D_compi); % maximum degree
Delta_1 = max(D_1i);
Delta_2 = max(D_2i);
epsilon_comp = 1/Delta_comp - eps; % parameter of Perron matrix
epsilon_1 = 1/Delta_1 - eps;
epsilon_2 = 1/Delta_2 - eps;

%% Aggregation GP parameters
% model parameters for the SE covariance function
sf2 = 1; % initial value for output scale amplitude \sigma_f^2
ell = 1; % initial value for length-scale l_1 and l_2
sn2 = 0.1; % initial value for noise variance \sigma_{\epsilon}^2

% Training
opts.Xnorm = '' ; % normalize
opts.Ynorm = 'Y' ;
opts.Ms = agents ; % number of experts
opts.sf2 = sf2 ;
opts.ell = ell ;
opts.sn2 = sn2 ;
opts.meanfunc = [] ; % other options .meanfunc, .covfunc, .likfunc, .inffunc
opts.covfunc = @covSEard; % separable squared exponential covarince fcn
opts.likfunc = @likGauss;
opts.inffunc = @infGaussLik ;
opts.numOptFC = 25 ; % optimization setting for min-NLML 

%% Factorized Training
t1 = clock;
[models,t_dGP_train,xs,ys,n_per] = aggregation_train_2D(X,z,opts) ;
t2 =clock;
t.training.nested = etime(t2,t1) ;
hyp_lik = models{1}.hyp.lik ; % noise estimated parameter

%% Nested Pointwise Aggregation of GP Experts (NPAE)
t3 = clock;
[mu_npae,s2_npae,t.prediction.nested,k_M_x,K_M_x,mu_all,s2_all,kss] ...
    = aggregation_predict(Xt,models) ;
t4 =clock;
t.prediction.npae = etime(t4,t3) ;

%% DEC-NPAE <- JOR and DAC 
thres_jor_max_min = 1e-3; % threshold for maximin JOR stopping criterion
thres_dac_max_min = 1e-4; % threshold for maximin DAC stopping criterion
iter_jor = 5000; % maximum number of iterations until convergence
iter_dac = 5000;

thres_cbnn = -1; % negative number corresponds to zero agent removal

t9 = clock;

[mu_dec_npae_complete,s2_dec_npae_complete,mu_dec_npae_1_hop,...
    s2_dec_npae_1_hop,mu_dec_npae_2_hop,s2_dec_npae_2_hop,...
    iter_jor_convrg_max_min,iter_dac_convrg_max_min,iter_dac_convrg_max_min_1,...
    iter_dac_convrg_max_min_2,nearest_neighbors_dec_npae,omega]...
    = dec_npae(nt,K_M_x,k_M_x,iter_jor,iter_dac,opts,kss,mu_all,models,...
    hyp_lik,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,thres_jor_max_min,...
    thres_dac_max_min,thres_cbnn);
t10 = clock ;
t.prediction.dec_npae = etime(t10,t9) ;

nearestNeighbors.dec_NPAE = nearest_neighbors_dec_npae;
convergence.jor = round( sum(iter_jor_convrg_max_min(1,:))/nt );
convergence.dac_complete = round( sum(iter_dac_convrg_max_min(1,:))/nt );
convergence.dac_1_hop = round( sum(iter_dac_convrg_max_min_1(1,:))/nt );
convergence.dac_2_hop = round( sum(iter_dac_convrg_max_min_2(1,:))/nt );
convergence.dec_npae_complete = convergence.jor + convergence.dac_complete;
convergence.dec_npae_1_hop = ( convergence.jor*(opts.Ms-1) ) + convergence.dac_1_hop;
mu_dec_npae_complete = mu_dec_npae_complete';
s2_dec_npae_complete = s2_dec_npae_complete';
mu_dec_npae_1_hop = mu_dec_npae_1_hop';
s2_dec_npae_1_hop = s2_dec_npae_1_hop';
mu_dec_npae_2_hop = mu_dec_npae_2_hop';
s2_dec_npae_2_hop = s2_dec_npae_2_hop';

%% DIST-NPAE <- DALE
thres_cbnn = 1e-2; % threshold for covariance based nearest neighbor (CBNN)
iter_dale_1_hop = 200; % maximum number of iterations until convergence
iter_dale_2_hop = 200;

% The dale threshold is very sensitive - change according to the number of
% agents - the smaller the threshold the better the inverse approximation
% Recommended values: 2,4 agents: 1e-4; 5,10 agents: 8*1e-3; 
% 20 agents: 2*1e-2; 40 agents: 3*1e-2
% For large networks Assumption 1 (independence) is violated
thres_dale_max_min = 2*1e-2; 

t17 =clock;
[mu_dist_npae,s2_dist_npae,mu_dist_npae_2_hop,s2_dist_npae_2_hop,...
    iter_dale_convrg_max_min_1_hop,iter_dale_convrg_max_min_2_hop,...
    nearest_neighbors_dist_npae] = ...
    dist_npae (nt,K_M_x,k_M_x,iter_dale_1_hop,iter_dale_2_hop,opts,kss,...
    mu_all,s2_all,models,hyp_lik,thres_dale_max_min,thres_cbnn);
t18=clock;
t.prediction.dist_npae = etime(t18,t17);

nearestNeighbors.dist_NPAE = nearest_neighbors_dist_npae
convergence.dist_npae_1_hop = round( sum(iter_dale_convrg_max_min_1_hop(1,:))/nt );
convergence.dist_npae_2_hop = round( sum(iter_dale_convrg_max_min_2_hop(1,:))/nt )

%% Metrics 
trial = 1;
MSPE.nested(trial) = sum((zt - mu_npae).^2)/nt;
MSPE.dec_npae_complete(trial) = sum((zt - mu_dec_npae_complete).^2)/nt;
MSPE.dec_npae_1_hop(trial) = sum((zt - mu_dec_npae_1_hop).^2)/nt;
MSPE.dist_npae_1_hop(trial) = sum((zt - mu_dist_npae).^2)/nt;
MSPE.dist_npae_2_hop(trial) = sum((zt - mu_dist_npae_2_hop).^2)/nt;

RMSPE.nested(trial) = sqrt(sum((zt - mu_npae).^2)/nt);
RMSPE.dec_npae_complete(trial) = sqrt(sum((zt - mu_dec_npae_complete).^2)/nt);
RMSPE.dec_npae_1_hop(trial) = sqrt(sum((zt - mu_dec_npae_1_hop).^2)/nt);
RMSPE.dist_npae_1_hop(trial) = sqrt(sum((zt - mu_dist_npae).^2)/nt);
RMSPE.dist_npae_2_hop(trial) = sqrt(sum((zt - mu_dist_npae_2_hop).^2)/nt)

NRMSPE.nested(trial) = ( sqrt(sum((zt - mu_npae).^2)/nt) )/zt_normal;
NRMSPE.dec_npae_complete(trial) = ( sqrt(sum((zt - mu_dec_npae_complete).^2)/nt) )/zt_normal;
NRMSPE.dec_npae_1_hop(trial) =  ( sqrt(sum((zt - mu_dec_npae_1_hop).^2)/nt) )/zt_normal;
NRMSPE.dist_npae_1_hop(trial) = ( sqrt(sum((zt - mu_dist_npae).^2)/nt) )/zt_normal;
NRMSPE.dist_npae_2_hop(trial) = ( sqrt(sum((zt - mu_dist_npae_2_hop).^2)/nt) )/zt_normal

MVE_nest.dec_npae_complete(trial) =  sum(s2_dec_npae_complete - s2_npae)/nt;
MVE_nest.dec_npae_1_hop(trial) = sum(s2_dec_npae_1_hop - s2_npae)/nt;
MVE_nest.dist_npae_1_hop(trial) = sum(s2_dist_npae - s2_npae)/nt;
MVE_nest.dist_npae_2_hop(trial) = sum(s2_dist_npae_2_hop - s2_npae)/nt

MAPE.nested(trial) = sum(abs(zt - mu_npae))/nt;
MAPE.dec_npae_complete(trial) = sum(abs(zt - mu_dec_npae_complete))/nt;
MAPE.dec_npae_1_hop(trial) = sum(abs(zt - mu_dec_npae_1_hop))/nt;
MAPE.dist_npae_1_hop(trial) = sum(abs(zt - mu_dist_npae))/nt;
MAPE.dist_npae_2_hop(trial) = sum(abs(zt - mu_dist_npae_2_hop))/nt;

distribution.nested = normpdf(zt,mu_npae,s2_npae);
distribution.dec_npae_complete = normpdf(zt,mu_dec_npae_complete,abs(s2_dec_npae_complete));
distribution.dec_npae_1_hop = normpdf(zt,mu_dec_npae_1_hop,abs(s2_dec_npae_1_hop));
distribution.dist_npae_1_hop = normpdf(zt,mu_dist_npae,s2_dist_npae);
distribution.dist_npae_2_hop = normpdf(zt,mu_dist_npae_2_hop,s2_dist_npae_2_hop);

NLPD.nested(trial) = -(1/nt)*sum(log(distribution.nested));
NLPD.dec_npae_complete(trial) = -(1/nt)*sum(log(distribution.dec_npae_complete));
NLPD.dec_npae_1_hop(trial) = -(1/nt)*sum(log(distribution.dec_npae_1_hop));
NLPD.dist_npae_1_hop(trial) = -(1/nt)*sum(log(distribution.dist_npae_1_hop));
NLPD.dist_npae_2_hop(trial) = -(1/nt)*sum(log(distribution.dist_npae_2_hop))