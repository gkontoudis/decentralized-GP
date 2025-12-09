% demo of distributed GPs for a random field (2D) example
clear all; close all; clc;

%% Test function
rng(99); % Control random number generator, Syntax rng(seed)
%%
n_trial = 4; % replications
n = 1000; % number of training data
sn = 0.25; % variance of true process
agents = n/50 ; % number of experts; 4=n/250, 10=n/100, 20=n/50, 40=n/25
nt = 5*1e-2*n; % number of prediction points (e.g., 0.05 for 50 prediction points nt = 0.05*n, for 5000 observations 2*1e-3*n)
oservations_per_agent = n./agents;
gr = 1/1000; % grid resolution
[xx,xy] = meshgrid(0:gr:1, 0:gr:1);
random_filed = 'goldstein-price'; % 'goldstein-price'
f.fnc = goldpr(xx,xy);
f.avg = sum(sum(f.fnc))/(length(f.fnc)^2);

figure
surf(xx,xy,f.fnc);hold on
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
box on; grid off; hold on;
%% Training and Prediction Data
for trial=1:n_trial
trial
% training data
switch random_filed
    case 'goldstein-price'
        x = linspace(0,1,n)';
        y = rand(n,1);
        X = [x,y];
        z = goldpr(x,y)+sn*randn(n,1);
end
% test data (prediction points)
switch random_filed
    case 'goldstein-price'
        xt = linspace(0,1,nt)';
        yt = rand(nt,1);
        Xt = [xt,yt];
        zt =  goldpr(xt,yt);
end
%% Aggregation GP
% model parameters for the SE covariance function
opts.mean = [];
opts.cov = [log(2),log(0.5),log(1)]; % inital values for lengthscales el1, el2 and signal std sf2, essentialy el1 = el2 = sf = exp(0) = 1
opts.lik = log(1); % initial value for noise std se2, essentially se = exp(-1) = 0.368
partitionCriterion = 'sequential' ; % 'random', 'kmeans', 'sequential'

% Training
opts.Xnorm = '';
opts.Ynorm = 'Y';
opts.Ms = agents; % number of experts
opts.partitionCriterion = partitionCriterion;
opts.ell = exp(opts.cov(1));
opts.el2 = exp(opts.cov(2));
opts.sf2 = exp(opts.cov(3));
opts.sn2 = exp(opts.lik);

opts.meanfunc = []; % .meanfunc, .covfunc, .likfunc, .inffunc: GP configurations
opts.covfunc = @covSEard;
opts.likfunc = @likGauss;
opts.inffunc = @infGaussLik;
opts.numOptFC = -250; % optimization setting for min-NLML (negative log maximum likelihood) - essentially # of linesearch
hyplength = length(opts.cov) + length(opts.lik);
%% Graph (line directed graph) % check plot_topologies.m for clarificationsin notation
s_comp =[];
r_comp =[];
for i =1:opts.Ms
s_comp_add = i*ones(1,opts.Ms-i); 
s_comp = [s_comp s_comp_add];
r_comp_add = linspace(i+1,opts.Ms,opts.Ms-i);
r_comp = [r_comp r_comp_add];
end

% 1-hop graph
s_1 = 1:1:(opts.Ms-1) ; % source nodes
r_1 = 2:1:opts.Ms ; % target nodes

% 2-hop graph
s_2 =[];
r_2 =[];
for i =1:opts.Ms-1
    if i == opts.Ms-1
        s_co = i;
    else
        s_co = i*ones(1,2);
    end
    s_2 = [s_2 s_co];
end
for i =2:opts.Ms
    if i == 2
        r_co = i;
    else
        r_co = i*ones(1,2);
    end
    r_2 = [r_2 r_co];
end

w_comp = ones(1, length(s_comp)); % weights
w_1 = ones(1, length(s_1)); % weights
w_2 = ones(1, length(s_2)); % weights
Acompgraph = graph(s_comp, r_comp, w_comp );
A1graph = graph(s_1, r_1, w_1); % create directed graph
A2graph = graph(s_2, r_2, w_2 );
D_compi = degree(Acompgraph); % in degrees
D_1i = degree(A1graph); % in degrees
D_2i = degree(A2graph); % in degrees
D_comp = diag(D_compi); % diagonal in-degree matrix
D_1 = diag(D_1i); % diagonal in-degree matrix
D_2 = diag(D_2i); % diagonal in-degree matrix
A_comp = full(adjacency(Acompgraph)); % adjacency matrix
A_1 = full(adjacency(A1graph)); % adjacency matrix
A_2 = full(adjacency(A2graph)); % adjacency matrix
L_comp = D_comp - A_comp; % Laplacian matrix
L_1 = D_1 - A_1; % Laplacian matrix
L_2 = D_2 - A_2; % Laplacian matrix
Delta_comp = max(D_compi);
Delta_1 = max(D_1i);
Delta_2 = max(D_2i);
epsilon_comp = 1/Delta_comp - .01;
epsilon_1 = 1/Delta_1 - .01;
epsilon_2 = 1/Delta_2 - .01;
%% Generalized Factorized Training (g-FACT-GP)
partitionCriterion = 'sequential_random' ; % 'random', 'kmeans', 'sequential', 'sequential_random'
opts.partitionCriterion = partitionCriterion ;
t1 = clock;
[models_grbcm,t_dGP_train,xs,ys,n_per,iter.gfactGP] = aggregation_train_2D(X,z,opts);
t2 =clock;
t.training.gfactGP(trial) = etime(t2,t1) ;
hyp_lik = models_grbcm{1}.hyp.lik ; % noise estimated parameter
for i=1:agents 
    models{1,i} = models_grbcm{1,i};
end
%% NPAE
criterion = 'NPAE';
t3 = clock;
[mu_npae,s2_npae,t.prediction.nested(trial),k_M_x,K_M_x,mu_all,s2_all,kss] ...
    = aggregation_predict(Xt,models,criterion) ;
t4 =clock;
s2_all_inv = 1./s2_all;
t.prediction.npae(trial) = etime(t4,t3) ;
%% DEC-NPAE <- JOR and DAC 
thres_pm = 5e-3;
thres_jor_max_min = 1e-3;
thres_dac_max_min = 1e-3;
iter_jor = 3000;
iter_dac = 3000;
iter_pm = 3000;
hyp_lik = models{1}.hyp.lik ; % noise estimated parameter
thres_cbnn = -1; % negative number corresponds to zero agent removal (covariance entries are all positive)
t9 = clock;
[mu_dec_npae_complete,s2_dec_npae_complete,mu_dec_npae_1_hop,...
    s2_dec_npae_1_hop,mu_dec_npae_2_hop,s2_dec_npae_2_hop,...
    iter_jor_convrg_max_min,iter_dac_convrg_max_min,iter_dac_convrg_max_min_1,...
    iter_dac_convrg_max_min_2,nearest_neighbors_dec_npae,iterations_pm_end_0,...
    omega]...
    = dec_npae(nt,K_M_x,k_M_x,iter_jor,iter_dac,opts,kss,mu_all,models,...
    hyp_lik,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,thres_jor_max_min,...
    thres_dac_max_min,thres_cbnn);
t10 = clock ;
t.prediction.dec_npae(trial) = etime(t10,t9) ;
convergence.pm_ipm_0(trial) = round( sum(iterations_pm_end_0)/nt );
convergence.jor(trial) = round( sum(iter_jor_convrg_max_min(1,:))/nt );
convergence.dac_complete(trial) = round( sum(iter_dac_convrg_max_min(1,:))/nt );
convergence.dac_1_hop(trial) = round( sum(iter_dac_convrg_max_min_1(1,:))/nt );
convergence.dac_2_hop(trial) = round( sum(iter_dac_convrg_max_min_2(1,:))/nt );
convergence.dec_npae_complete(trial) = convergence.jor(trial) + convergence.dac_complete(trial);
convergence.dec_npae_1_hop(trial) = ( convergence.jor(trial)*(opts.Ms-1) ) + convergence.dac_1_hop(trial);
convergence.dec_npae_2_hop(trial) = ( convergence.jor(trial)*(opts.Ms-1) ) + convergence.dac_2_hop(trial);
mu_dec_npae_complete = mu_dec_npae_complete';
s2_dec_npae_complete = s2_dec_npae_complete';
mu_dec_npae_1_hop = mu_dec_npae_1_hop';
s2_dec_npae_1_hop = s2_dec_npae_1_hop';
mu_dec_npae_2_hop = mu_dec_npae_2_hop';
s2_dec_npae_2_hop = s2_dec_npae_2_hop';

%% DEC-NPAE* <- JOR* and DAC 
thres_cbnn = -1; % negative number corresponds to zero agent removal (covariance entries are all positive)
t11 = clock;
[mu_dec_npae_star_complete,s2_dec_npae_star_complete,mu_dec_npae_star_1_hop,...
    s2_dec_npae_star_1_hop,mu_dec_npae_star_2_hop,s2_dec_npae_star_2_hop,...
    iter_jor_star_convrg_max_min,...
    iter_dac_star_convrg_max_min,iter_dac_star_convrg_max_min_1,...
    iter_dac_star_convrg_max_min_2,...
    nearest_neighbors_dec_npae_star,iterations_pm_end,omega_star]...
    = dec_npae_star(nt,K_M_x,k_M_x,iter_jor,iter_dac,opts,kss,mu_all,models,...
    hyp_lik,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,thres_jor_max_min,...
    thres_dac_max_min,thres_cbnn,thres_pm,iter_pm);
t12 = clock ;
t.prediction.dec_npae_star(trial) = etime(t12,t11) ;

convergence.pm_ipm(trial) = round( sum(iterations_pm_end)/nt );
convergence.jor_star(trial) = round( sum(iter_jor_star_convrg_max_min(1,:))/nt );
convergence.dac_star_complete(trial) = round( sum(iter_dac_star_convrg_max_min(1,:))/nt );
convergence.dac_star_1_hop(trial) = round( sum(iter_dac_star_convrg_max_min_1(1,:))/nt );
convergence.dac_star_2_hop(trial) = round( sum(iter_dac_star_convrg_max_min_2(1,:))/nt );
convergence.dec_npae_star_complete(trial) = convergence.jor_star(trial) + convergence.dac_star_complete(trial) + convergence.pm_ipm(trial);
convergence.dec_npae_star_1_hop(trial) = ( (convergence.jor_star(trial) + convergence.pm_ipm(trial))*(opts.Ms-1) ) + convergence.dac_star_1_hop(trial);
mu_dec_npae_star_complete = mu_dec_npae_star_complete';
s2_dec_npae_star_complete = s2_dec_npae_star_complete';
mu_dec_npae_star_1_hop = mu_dec_npae_star_1_hop';
s2_dec_npae_star_1_hop = s2_dec_npae_star_1_hop';
mu_dec_npae_star_2_hop = mu_dec_npae_star_2_hop';
s2_dec_npae_star_2_hop = s2_dec_npae_star_2_hop';
%% DIST-NPAE <- DALE
thres_cbnn = .6;
iter_dale_1_hop = 4000;
iter_dale_2_hop = 4000;
% iter_dale_r = 4000;
thres_dale_max_min = 1e-3;
t17 =clock;
[mu_dist_npae,s2_dist_npae,mu_dist_npae_2_hop,s2_dist_npae_2_hop,...mu_dist_npae_r,s2_dist_npae_r,
    iter_dale_convrg_max_min_1_hop,iter_dale_convrg_max_min_2_hop,...iter_dale_convrg_max_min_r,...
    nearest_neighbors_dist_npae] = ...
    dist_npae (nt,K_M_x,k_M_x,iter_dale_1_hop,iter_dale_2_hop,...iter_dale_r,
    opts,kss,mu_all,s2_all,models,hyp_lik,thres_dale_max_min,thres_cbnn);
t18=clock;
t.prediction.dist_npae(trial) = etime(t18,t17);

convergence.dist_npae_1_hop(trial) = round( sum(iter_dale_convrg_max_min_1_hop(1,:))/nt );
convergence.dist_npae_2_hop(trial) = round( sum(iter_dale_convrg_max_min_2_hop(1,:))/nt );
%% PoE
t19 = clock;
[mu_poe,s2_poe] = poe(nt,mu_all,s2_all,models);
t20 =clock;

t.prediction.poe(trial) = etime(t20,t19) ;

%% DEC-PoE
thres_cbnn = -1;
t27 = clock;
[mu_dec_poe_complete,s2_dec_poe_complete,mu_dec_poe_1_hop,...
    s2_dec_poe_1_hop,mu_dec_poe_2_hop,s2_dec_poe_2_hop,...
    iter_dac_poe_convrg_max_min,iter_dac_poe_convrg_max_min_1,...
    iter_dac_poe_convrg_max_min_2,nearest_neighbors_dec_poe]=...
    dec_poe(mu_all,s2_all,s2_all_inv,models,...
        nt,k_M_x,iter_dac,opts,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,...
        thres_dac_max_min,thres_cbnn);
t28 =clock;
t.prediction.dec_poe(trial) = etime(t28,t27) ;

convergence.dec_poe_complete(trial) = round( sum(iter_dac_poe_convrg_max_min(1,:))/nt );
convergence.dec_poe_1_hop(trial) = round( sum(iter_dac_poe_convrg_max_min_1(1,:))/nt );
convergence.dec_poe_2_hop(trial) = round( sum(iter_dac_poe_convrg_max_min_2(1,:))/nt );
mu_dec_poe_complete = mu_dec_poe_complete';
s2_dec_poe_complete = s2_dec_poe_complete';
mu_dec_poe_1_hop = mu_dec_poe_1_hop';
s2_dec_poe_1_hop = s2_dec_poe_1_hop';
mu_dec_poe_2_hop = mu_dec_poe_2_hop';
s2_dec_poe_2_hop = s2_dec_poe_2_hop';

%% DEC-NN-PoE
thres_cbnn = .6;
t13 = clock;
[mu_dec_nn_poe_complete,s2_dec_nn_poe_complete,mu_dec_nn_poe_1_hop,...
    s2_dec_nn_poe_1_hop,mu_dec_nn_poe_2_hop,s2_dec_nn_poe_2_hop,...
    iter_dac_nn_poe_convrg_max_min,iter_dac_nn_poe_convrg_max_min_1,...
    iter_dac_nn_poe_convrg_max_min_2,nearest_neighbors_dec_nn_poe]=...
    dec_poe(mu_all,s2_all,s2_all_inv,models,...
        nt,k_M_x,iter_dac,opts,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,...
        thres_dac_max_min,thres_cbnn);
t14 =clock;
t.prediction.dec_nn_poe(trial) = etime(t14,t13) ;

convergence.dec_nn_poe_complete(trial) = round( sum(iter_dac_nn_poe_convrg_max_min(1,:))/nt );
convergence.dec_nn_poe_1_hop(trial) = round( sum(iter_dac_nn_poe_convrg_max_min_1(1,:))/nt );
convergence.dec_nn_poe_2_hop(trial) = round( sum(iter_dac_nn_poe_convrg_max_min_2(1,:))/nt );
mu_dec_nn_poe_complete = mu_dec_nn_poe_complete';
s2_dec_nn_poe_complete = s2_dec_nn_poe_complete';
mu_dec_nn_poe_1_hop = mu_dec_nn_poe_1_hop';
s2_dec_nn_poe_1_hop = s2_dec_nn_poe_1_hop';
mu_dec_nn_poe_2_hop = mu_dec_nn_poe_2_hop';
s2_dec_nn_poe_2_hop = s2_dec_nn_poe_2_hop';

%% gPoE
t21 = clock;
[mu_gpoe,s2_gpoe] = gpoe(nt,mu_all,s2_all,models);
t22 =clock;

t.prediction.gpoe(trial) = etime(t22,t21) ;

%% DEC-gPoE
thres_cbnn = -1;
t29 = clock;
[mu_dec_gpoe_complete,s2_dec_gpoe_complete,mu_dec_gpoe_1_hop,...
    s2_dec_gpoe_1_hop,mu_dec_gpoe_2_hop,s2_dec_gpoe_2_hop,...
    iter_dac_gpoe_convrg_max_min,iter_dac_gpoe_convrg_max_min_1,...
    iter_dac_gpoe_convrg_max_min_2,nearest_neighbors_dec_gpoe]=...
    dec_gpoe(mu_all,s2_all,s2_all_inv,models,...
        nt,k_M_x,iter_dac,opts,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,...
        thres_dac_max_min,thres_cbnn);
t30 =clock;
t.prediction.dec_gpoe(trial) = etime(t30,t29) ;

convergence.dec_gpoe_complete(trial) = round( sum(iter_dac_gpoe_convrg_max_min(1,:))/nt );
convergence.dec_gpoe_1_hop(trial) = round( sum(iter_dac_gpoe_convrg_max_min_1(1,:))/nt );
convergence.dec_gpoe_2_hop(trial) = round( sum(iter_dac_gpoe_convrg_max_min_2(1,:))/nt );
mu_dec_gpoe_complete = mu_dec_gpoe_complete';
s2_dec_gpoe_complete = s2_dec_gpoe_complete';
mu_dec_gpoe_1_hop = mu_dec_gpoe_1_hop';
s2_dec_gpoe_1_hop = s2_dec_gpoe_1_hop';
mu_dec_gpoe_2_hop = mu_dec_gpoe_2_hop';
s2_dec_gpoe_2_hop = s2_dec_gpoe_2_hop';

%% DEC-NN-gPoE
thres_cbnn = .6;
t31 = clock;
[mu_dec_nn_gpoe_complete,s2_dec_nn_gpoe_complete,mu_dec_nn_gpoe_1_hop,...
    s2_dec_nn_gpoe_1_hop,mu_dec_nn_gpoe_2_hop,s2_dec_nn_gpoe_2_hop,...
    iter_dac_nn_gpoe_convrg_max_min,iter_dac_nn_gpoe_convrg_max_min_1,...
    iter_dac_nn_gpoe_convrg_max_min_2,nearest_neighbors_dec_nn_gpoe]=...
    dec_gpoe(mu_all,s2_all,s2_all_inv,models,...
        nt,k_M_x,iter_dac,opts,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,...
        thres_dac_max_min,thres_cbnn);
t32 =clock;
t.prediction.dec_nn_gpoe(trial) = etime(t32,t31) ;

convergence.dec_nn_gpoe_complete(trial) = round( sum(iter_dac_nn_gpoe_convrg_max_min(1,:))/nt );
convergence.dec_nn_gpoe_1_hop(trial) = round( sum(iter_dac_nn_gpoe_convrg_max_min_1(1,:))/nt );
convergence.dec_nn_gpoe_2_hop(trial) = round( sum(iter_dac_nn_gpoe_convrg_max_min_2(1,:))/nt );
mu_dec_nn_gpoe_complete = mu_dec_nn_gpoe_complete';
s2_dec_nn_gpoe_complete = s2_dec_nn_gpoe_complete';
mu_dec_nn_gpoe_1_hop = mu_dec_nn_gpoe_1_hop';
s2_dec_nn_gpoe_1_hop = s2_dec_nn_gpoe_1_hop';
mu_dec_nn_gpoe_2_hop = mu_dec_nn_gpoe_2_hop';
s2_dec_nn_gpoe_2_hop = s2_dec_nn_gpoe_2_hop';

%% BCM
t23 = clock;
[mu_bcm,s2_bcm] = bcm(nt,mu_all,s2_all,models,kss);
t24 =clock;

t.prediction.bcm(trial) = etime(t24,t23) ;

%% DEC-BCM
thres_cbnn = -1;
t33 = clock;
[mu_dec_bcm_complete,s2_dec_bcm_complete,mu_dec_bcm_1_hop,...
    s2_dec_bcm_1_hop,mu_dec_bcm_2_hop,s2_dec_bcm_2_hop,...
    iter_dac_bcm_convrg_max_min,iter_dac_bcm_convrg_max_min_1,...
    iter_dac_bcm_convrg_max_min_2,nearest_neighbors_dec_bcm]=...
    dec_bcm(mu_all,s2_all,s2_all_inv,models,...
        nt,k_M_x,iter_dac,opts,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,...
        thres_dac_max_min,thres_cbnn,kss);
t34 =clock;
t.prediction.dec_bcm(trial) = etime(t34,t33) ;

convergence.dec_bcm_complete(trial) = round( sum(iter_dac_bcm_convrg_max_min(1,:))/nt );
convergence.dec_bcm_1_hop(trial) = round( sum(iter_dac_bcm_convrg_max_min_1(1,:))/nt );
convergence.dec_bcm_2_hop(trial) = round( sum(iter_dac_bcm_convrg_max_min_2(1,:))/nt );
mu_dec_bcm_complete = mu_dec_bcm_complete';
s2_dec_bcm_complete = s2_dec_bcm_complete';
mu_dec_bcm_1_hop = mu_dec_bcm_1_hop';
s2_dec_bcm_1_hop = s2_dec_bcm_1_hop';
mu_dec_bcm_2_hop = mu_dec_bcm_2_hop';
s2_dec_bcm_2_hop = s2_dec_bcm_2_hop';

%% DEC-NN-BCM
thres_cbnn = .6;
t35 = clock;
[mu_dec_nn_bcm_complete,s2_dec_nn_bcm_complete,mu_dec_nn_bcm_1_hop,...
    s2_dec_nn_bcm_1_hop,mu_dec_nn_bcm_2_hop,s2_dec_nn_bcm_2_hop,...
    iter_dac_nn_bcm_convrg_max_min,iter_dac_nn_bcm_convrg_max_min_1,...
    iter_dac_nn_bcm_convrg_max_min_2,nearest_neighbors_dec_nn_bcm]=...
    dec_bcm(mu_all,s2_all,s2_all_inv,models,...
        nt,k_M_x,iter_dac,opts,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,...
        thres_dac_max_min,thres_cbnn,kss);
t36 =clock;
t.prediction.dec_nn_bcm(trial) = etime(t36,t35) ;

convergence.dec_nn_bcm_complete(trial) = round( sum(iter_dac_nn_bcm_convrg_max_min(1,:))/nt );
convergence.dec_nn_bcm_1_hop(trial) = round( sum(iter_dac_nn_bcm_convrg_max_min_1(1,:))/nt );
convergence.dec_nn_bcm_2_hop(trial) = round( sum(iter_dac_nn_bcm_convrg_max_min_2(1,:))/nt );
mu_dec_nn_bcm_complete = mu_dec_nn_bcm_complete';
s2_dec_nn_bcm_complete = s2_dec_nn_bcm_complete';
mu_dec_nn_bcm_1_hop = mu_dec_nn_bcm_1_hop';
s2_dec_nn_bcm_1_hop = s2_dec_nn_bcm_1_hop';
mu_dec_nn_bcm_2_hop = mu_dec_nn_bcm_2_hop';
s2_dec_nn_bcm_2_hop = s2_dec_nn_bcm_2_hop';

%% rBCM
t25 = clock;
[mu_rbcm,s2_rbcm] = rbcm(nt,mu_all,s2_all,models,kss);
t26 =clock;

t.prediction.rbcm(trial) = etime(t26,t25) ;

%% DEC-rBCM
thres_cbnn = -1;
t37 = clock;
[mu_dec_rbcm_complete,s2_dec_rbcm_complete,mu_dec_rbcm_1_hop,...
    s2_dec_rbcm_1_hop,mu_dec_rbcm_2_hop,s2_dec_rbcm_2_hop,...
    iter_dac_rbcm_convrg_max_min,iter_dac_rbcm_convrg_max_min_1,...
    iter_dac_rbcm_convrg_max_min_2,nearest_neighbors_dec_rbcm]=...
    dec_rbcm(mu_all,s2_all,s2_all_inv,models,...
        nt,k_M_x,iter_dac,opts,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,...
        thres_dac_max_min,thres_cbnn,kss);
t38 =clock;
t.prediction.dec_rbcm(trial) = etime(t38,t37) ;

convergence.dec_rbcm_complete(trial) = round( sum(iter_dac_rbcm_convrg_max_min(1,:))/nt );
convergence.dec_rbcm_1_hop(trial) = round( sum(iter_dac_rbcm_convrg_max_min_1(1,:))/nt );
convergence.dec_rbcm_2_hop(trial) = round( sum(iter_dac_rbcm_convrg_max_min_2(1,:))/nt );
mu_dec_rbcm_complete = mu_dec_rbcm_complete';
s2_dec_rbcm_complete = s2_dec_rbcm_complete';
mu_dec_rbcm_1_hop = mu_dec_rbcm_1_hop';
s2_dec_rbcm_1_hop = s2_dec_rbcm_1_hop';
mu_dec_rbcm_2_hop = mu_dec_rbcm_2_hop';
s2_dec_rbcm_2_hop = s2_dec_rbcm_2_hop';

%% DEC-NN-rBCM
thres_cbnn = .6;
t39 = clock;
[mu_dec_nn_rbcm_complete,s2_dec_nn_rbcm_complete,mu_dec_nn_rbcm_1_hop,...
    s2_dec_nn_rbcm_1_hop,mu_dec_nn_rbcm_2_hop,s2_dec_nn_rbcm_2_hop,...
    iter_dac_nn_rbcm_convrg_max_min,iter_dac_nn_rbcm_convrg_max_min_1,...
    iter_dac_nn_rbcm_convrg_max_min_2,nearest_neighbors_dec_nn_rbcm]=...
    dec_rbcm(mu_all,s2_all,s2_all_inv,models,...
        nt,k_M_x,iter_dac,opts,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,...
        thres_dac_max_min,thres_cbnn,kss);
t40 =clock;
t.prediction.dec_nn_rbcm(trial) = etime(t40,t39) ;

convergence.dec_nn_rbcm_complete(trial) = round( sum(iter_dac_nn_rbcm_convrg_max_min(1,:))/nt);
convergence.dec_nn_rbcm_1_hop(trial) = round( sum(iter_dac_nn_rbcm_convrg_max_min_1(1,:))/nt);
convergence.dec_nn_rbcm_2_hop(trial) = round( sum(iter_dac_nn_rbcm_convrg_max_min_2(1,:))/nt);
mu_dec_nn_rbcm_complete = mu_dec_nn_rbcm_complete';
s2_dec_nn_rbcm_complete = s2_dec_nn_rbcm_complete';
mu_dec_nn_rbcm_1_hop = mu_dec_nn_rbcm_1_hop';
s2_dec_nn_rbcm_1_hop = s2_dec_nn_rbcm_1_hop';
mu_dec_nn_rbcm_2_hop = mu_dec_nn_rbcm_2_hop';
s2_dec_nn_rbcm_2_hop = s2_dec_nn_rbcm_2_hop';

%% grBCM
t43 = clock;
[mu_grbcm,mu_grbcm_all,mu_grbcm_comm,s2_grbcm,s2_grbcm_all,s2_grbcm_comm,models_cross]...
    = grbcm(nt,models_grbcm,Xt,n_per) ;
t44 =clock;
s2_grbcm_all_inv = 1./s2_grbcm_all;
s2_grbcm_comm_inv = 1./s2_grbcm_comm;

t.prediction.grbcm(trial) = etime(t44,t43) ;

%% DEC-grBCM
thres_cbnn = -1;
t45 = clock;
[mu_dec_grbcm_complete,s2_dec_grbcm_complete,mu_dec_grbcm_1_hop,...
    s2_dec_grbcm_1_hop,mu_dec_grbcm_2_hop,s2_dec_grbcm_2_hop,...
    iter_dac_grbcm_convrg_max_min,iter_dac_grbcm_convrg_max_min_1,...
    iter_dac_grbcm_convrg_max_min_2,nearest_neighbors_dec_grbcm]=...
    dec_grbcm(mu_grbcm_all,mu_grbcm_comm,s2_grbcm_all,s2_grbcm_comm,...
        s2_grbcm_all_inv,s2_grbcm_comm_inv,models_cross,...
        nt,k_M_x,iter_dac,opts,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,...
        thres_dac_max_min,thres_cbnn);
t46=clock;
t.prediction.dec_grbcm(trial) = etime(t46,t45) ;

convergence.dec_grbcm_complete(trial) = round( sum(iter_dac_grbcm_convrg_max_min(1,:))/nt );
convergence.dec_grbcm_1_hop(trial) = round( sum(iter_dac_grbcm_convrg_max_min_1(1,:))/nt );
convergence.dec_grbcm_2_hop(trial) = round( sum(iter_dac_grbcm_convrg_max_min_2(1,:))/nt );
mu_dec_grbcm_complete = mu_dec_grbcm_complete';
s2_dec_grbcm_complete = s2_dec_grbcm_complete';
mu_dec_grbcm_1_hop = mu_dec_grbcm_1_hop';
s2_dec_grbcm_1_hop = s2_dec_grbcm_1_hop';
mu_dec_grbcm_2_hop = mu_dec_grbcm_2_hop';
s2_dec_grbcm_2_hop = s2_dec_grbcm_2_hop';

%% DEC-NN-grBCM
thres_cbnn = .6;
t47 = clock;
[mu_dec_nn_grbcm_complete,s2_dec_nn_grbcm_complete,mu_dec_nn_grbcm_1_hop,...
    s2_dec_nn_grbcm_1_hop,mu_dec_nn_grbcm_2_hop,s2_dec_nn_grbcm_2_hop,...
    iter_dac_nn_grbcm_convrg_max_min,iter_dac_nn_grbcm_convrg_max_min_1,...
    iter_dac_nn_grbcm_convrg_max_min_2,nearest_neighbors_dec_nn_grbcm]=...
    dec_grbcm(mu_grbcm_all,mu_grbcm_comm,s2_grbcm_all,s2_grbcm_comm,...
        s2_grbcm_all_inv,s2_grbcm_comm_inv,models_cross,...
        nt,k_M_x,iter_dac,opts,Delta_1,L_1,Delta_2,L_2,Delta_comp,L_comp,...
        thres_dac_max_min,thres_cbnn);
t48=clock;
t.prediction.dec_nn_grbcm(trial) = etime(t48,t47) ;

convergence.dec_nn_grbcm_complete(trial) = round( sum(iter_dac_nn_grbcm_convrg_max_min(1,:))/nt );
convergence.dec_nn_grbcm_1_hop(trial) = round( sum(iter_dac_nn_grbcm_convrg_max_min_1(1,:))/nt );
convergence.dec_nn_grbcm_2_hop(trial) = round( sum(iter_dac_nn_grbcm_convrg_max_min_2(1,:))/nt );
mu_dec_nn_grbcm_complete = mu_dec_nn_grbcm_complete';
s2_dec_nn_grbcm_complete = s2_dec_nn_grbcm_complete';
mu_dec_nn_grbcm_1_hop = mu_dec_nn_grbcm_1_hop';
s2_dec_nn_grbcm_1_hop = s2_dec_nn_grbcm_1_hop';
mu_dec_nn_grbcm_2_hop = mu_dec_nn_grbcm_2_hop';
s2_dec_nn_grbcm_2_hop = s2_dec_nn_grbcm_2_hop';

%% Metrics 
RMSPE.npae(trial) = sqrt(sum((zt - mu_npae).^2)/nt);
RMSPE.poe(trial) = sqrt(sum((zt - mu_poe).^2)/nt);
RMSPE.dec_poe_1_hop(trial) = sqrt(sum((zt - mu_dec_poe_1_hop).^2)/nt);
RMSPE.dec_poe_2_hop(trial) = sqrt(sum((zt - mu_dec_poe_2_hop).^2)/nt);
RMSPE.dec_nn_poe_1_hop(trial) = sqrt(sum((zt - mu_dec_nn_poe_1_hop).^2)/nt);
RMSPE.dec_nn_poe_2_hop(trial) = sqrt(sum((zt - mu_dec_nn_poe_2_hop).^2)/nt);
RMSPE.gpoe(trial) = sqrt(sum((zt - mu_gpoe).^2)/nt);
RMSPE.dec_gpoe_1_hop(trial) = sqrt(sum((zt - mu_dec_gpoe_1_hop).^2)/nt);
RMSPE.dec_gpoe_2_hop(trial) = sqrt(sum((zt - mu_dec_gpoe_2_hop).^2)/nt);
RMSPE.dec_nn_gpoe_1_hop(trial) = sqrt(sum((zt - mu_dec_nn_gpoe_1_hop).^2)/nt);
RMSPE.dec_nn_gpoe_2_hop(trial) = sqrt(sum((zt - mu_dec_nn_gpoe_2_hop).^2)/nt);
RMSPE.bcm(trial) = sqrt(sum((zt - mu_bcm).^2)/nt);
RMSPE.dec_bcm_1_hop(trial) = sqrt(sum((zt - mu_dec_bcm_1_hop).^2)/nt);
RMSPE.dec_bcm_2_hop(trial) = sqrt(sum((zt - mu_dec_bcm_2_hop).^2)/nt);
RMSPE.dec_nn_bcm_1_hop(trial) = sqrt(sum((zt - mu_dec_nn_bcm_1_hop).^2)/nt);
RMSPE.dec_nn_bcm_2_hop(trial) = sqrt(sum((zt - mu_dec_nn_bcm_2_hop).^2)/nt);
RMSPE.rbcm(trial) = sqrt(sum((zt - mu_rbcm).^2)/nt);
RMSPE.dec_rbcm_1_hop(trial) = sqrt(sum((zt - mu_dec_rbcm_1_hop).^2)/nt);
RMSPE.dec_rbcm_2_hop(trial) = sqrt(sum((zt - mu_dec_rbcm_2_hop).^2)/nt);
RMSPE.dec_nn_rbcm_1_hop(trial) = sqrt(sum((zt - mu_dec_nn_rbcm_1_hop).^2)/nt);
RMSPE.dec_nn_rbcm_2_hop(trial) = sqrt(sum((zt - mu_dec_nn_rbcm_2_hop).^2)/nt);
RMSPE.grbcm(trial) = sqrt(sum((zt - mu_grbcm).^2)/nt);
RMSPE.dec_grbcm_1_hop(trial) = sqrt(sum((zt - mu_dec_grbcm_1_hop).^2)/nt);
RMSPE.dec_grbcm_2_hop(trial) = sqrt(sum((zt - mu_dec_grbcm_2_hop).^2)/nt);
RMSPE.dec_nn_grbcm_1_hop(trial) = sqrt(sum((zt - mu_dec_nn_grbcm_1_hop).^2)/nt);
RMSPE.dec_nn_grbcm_2_hop(trial) = sqrt(sum((zt - mu_dec_nn_grbcm_2_hop).^2)/nt);
RMSPE.dec_npae_complete(trial) = sqrt(sum((zt - mu_dec_npae_complete).^2)/nt);
RMSPE.dec_npae_star_complete(trial) = sqrt(sum((zt - mu_dec_npae_star_complete).^2)/nt);
RMSPE.dec_npae_1_hop(trial) = sqrt(sum((zt - mu_dec_npae_1_hop).^2)/nt);
RMSPE.dec_npae_star_1_hop(trial) = sqrt(sum((zt - mu_dec_npae_star_1_hop).^2)/nt);
RMSPE.dist_npae_1_hop(trial) = sqrt(sum((zt - mu_dist_npae).^2)/nt);
RMSPE.dist_npae_2_hop(trial) = sqrt(sum((zt - mu_dist_npae_2_hop).^2)/nt);

NN.dec_nn_bcm(trial) = mean(nearest_neighbors_dec_nn_bcm);
NN.dec_nn_rbcm(trial) = mean(nearest_neighbors_dec_nn_rbcm);
NN.dec_nn_grbcm(trial) = mean(nearest_neighbors_dec_nn_grbcm);
NN.dec_nn_poe(trial) = mean(nearest_neighbors_dec_nn_poe);
NN.dec_nn_gpoe(trial) = mean(nearest_neighbors_dec_nn_gpoe);
NN.dist_npae(trial) = mean(nearest_neighbors_dist_npae);

end
%%
NN
RMSPE