%% #### -------------------------------------------------------------- ####
%% George Kontoudis, george.kontoudis@mines.edu, www.georgekontoudis.com
%% Colorado School of Mines, Mechanical Engineering
%% Scalable, Federated Gaussian Process Training for Decentralized
%% Multi-Agent Systems, IEEE Access 2024
%% Demo for a synthetic random field (2D)
%% #### -------------------------------------------------------------- ####
clear all; close all; clc;
%% Generate test data
rng(100)
% number of training data points
n = 961; % 961, 8100 , 16900, 32400
trials = 3;
agents = 10; % 4, 10, 20

% Parameters
alpha = 1e-5; % gradient descent step size
cADMM_tol = 1e-3; % ADMM tolerance
cl_pxADMM_tol = cADMM_tol; % ADMM tolerance
cl_gpxADMM_tol = cADMM_tol; % ADMM tolerance
max_iter_dec_cADMM = 1e2;
max_iter_dec_cl_pxADMM = 1e2;
max_iter_dec_cl_gpxADMM = 1e2;
rho = 500; % penalty term ADMM
Lip = 5000; % Lipschitz constant

switch n
    case 961
        % load true hyperparameter values
        load src/hyperparameters_obs_961_input_2D_hyp_4_repl_5.dat
        el1.true = hyperparameters_obs_961_input_2D_hyp_4_repl_5(1);
        el2.true = hyperparameters_obs_961_input_2D_hyp_4_repl_5(2);
        sf.true = hyperparameters_obs_961_input_2D_hyp_4_repl_5(3);
        sn.true = hyperparameters_obs_961_input_2D_hyp_4_repl_5(4);
        hyplength = length(hyperparameters_obs_961_input_2D_hyp_4_repl_5);

        load src/input_x_obs_961_input_2D_hyp_4_repl_5.dat
        xx = input_x_obs_961_input_2D_hyp_4_repl_5';
        xx_length = length(xx); xx_length_2 = xx_length^2;
        xx_vec = reshape(xx,xx_length_2,1);

        load src/input_y_obs_961_input_2D_hyp_4_repl_5.dat
        xy = input_y_obs_961_input_2D_hyp_4_repl_5';
        xy_vec = reshape(xy,xx_length_2,1);

        load src/training_data_obs_961_input_2D_hyp_4_repl_5.dat
        Y_all = training_data_obs_961_input_2D_hyp_4_repl_5';
end

% Squared hyperparameters
el12.true = el1.true^2;
el22.true = el2.true^2;
sf2.true = sf.true^2;
sn2.true = sn.true^2;

ky = 0;
for i=1:trials
    Y_each(:,:,i) = Y_all(:,ky*xx_length+1:i*xx_length); % 31 for 961, 90 for 8100, 130 for 16900, 180 for 32400
    ky=ky+1;
end
x(:,1) = xx_vec;
x(:,2) = xy_vec;
Y = reshape(Y_each,xx_length_2,1,trials);

for i=1:trials
    f=figure(i);
    surf(xx,xy,Y_each(:,:,i));hold on
    hold on;
    [~,c]=contourf(xx,xy,Y_each(:,:,i));hold on
    shading interp
    set(gca,'fontsize',16)
    h = colormap(f,pink);
    brighten(0.6);
    set(gcf,'color','w')
    cbar = colorbar;
    hAx = gca;
    hAx.CLim = [-3 3];
    cbar.Limits = [-3 3];
    view(0,90)
    xlabel('x') ; ylabel('y') ;
    %     title('Reference')
    set(gca,'fontsize',16)
    set(gcf,'color','w')
    box off; grid off; hold on;
end
%% Training of full GP
meanfunc = [];
covfunc = @covSEard;
likfunc = @likGauss;
% pause
%%
for trial=1:trials
    trial
    %     for initial_condition = 1:random_initializations
    opts.mean = [];
    opts.cov = [log(2),log(0.5),log(1)]; % inital values for lengthscales el1, el2 and signal std sf2, essentialy el1 = el2 = sf = exp(0) = 1
    opts.lik = log(1); % initial value for noise std se2, essentially se = exp(-1) = 0.368

    % Full GP training
    t1 = clock;
    hyp.fullGP = minimize(opts, @gp, -250, @infGaussLik, meanfunc,...
        covfunc, likfunc, x, Y(:,:,trial));
    t2 = clock;

    % Going back to the parameter space exp(log(x)) = x
    el1.fullGP = exp(hyp.fullGP.cov(1));
    el2.fullGP = exp(hyp.fullGP.cov(2));
    sf.fullGP = exp(hyp.fullGP.cov(3));
    sn.fullGP = exp(hyp.fullGP.lik);

    % Squared parameters exp(2*log(x)) = x^2
    el12.fullGP = exp(2*hyp.fullGP.cov(1));
    el22.fullGP = exp(2*hyp.fullGP.cov(2));
    sf2.fullGP = exp(2*hyp.fullGP.cov(3));
    sn2.fullGP = exp(2*hyp.fullGP.lik);

    % Time for training
    t.training.fullGP = etime(t2,t1) ;
    %% Factorized training
    partitionCriterion = 'sequential' ; % 'random', 'kmeans', 'sequential'

    opts.Xnorm = '' ; % doesn't change anything if I select 'X' or 'Y' or even if it's empty ''
    opts.Ynorm = 'Y' ;
    opts.Ms = agents; % number of experts
    opts.partitionCriterion = partitionCriterion ;
    opts.ell = exp(opts.cov(1)) ;
    opts.el2 = exp(opts.cov(2)) ;
    opts.sf2 = exp(opts.cov(3)) ;
    opts.sn2 = exp(opts.lik) ;

    opts.meanfunc = [] ; % .meanfunc, .covfunc, .likfunc, .inffunc: GP configurations
    opts.covfunc = @covSEard;
    opts.likfunc = @likGauss;
    opts.inffunc = @infGaussLik ;

    opts.numOptFC = -250 ;

    t3 = clock;
    [models,t_dGP_train,xs,ys,iter.outer.factGP] = aggregation_train_2D(x,Y(:,:,trial),opts);
    t4 = clock;

    for i=1:opts.Ms
        el1.factGP(i) = exp(models{1,i}.hyp.cov(1));
        el2.factGP(i) = exp(models{1,i}.hyp.cov(2));
        sf.factGP(i) = exp(models{1,i}.hyp.cov(3));
        sn.factGP(i) = exp(models{1,i}.hyp.lik);

        el12.factGP(i) = exp(2*models{1,i}.hyp.cov(1));
        el22.factGP(i) = exp(2*models{1,i}.hyp.cov(2));
        sf2.factGP(i) = exp(2*models{1,i}.hyp.cov(3));
        sn2.factGP(i) = exp(2*models{1,i}.hyp.lik);
    end

    t.training.factGP = etime(t4,t3) ;
    %% Factorized GRBCM training
    partitionCriterion = 'sequential_random' ; % 'random', 'kmeans', 'sequential', 'sequential_random'
    opts.partitionCriterion = partitionCriterion ;

    t5 = clock;
    [models_grbcm,t_dGP_train,xs,ys,iter.outer.factGRBCM] = aggregation_train_2D(x,Y(:,:,trial),opts);
    t6 =clock;

    for i=1:opts.Ms
        el1.factGRBCM(i) = exp(models_grbcm{1,i}.hyp.cov(1));
        el2.factGRBCM(i) = exp(models_grbcm{1,i}.hyp.cov(2));
        sf.factGRBCM(i) = exp(models_grbcm{1,i}.hyp.cov(3));
        sn.factGRBCM(i) = exp(models_grbcm{1,i}.hyp.lik);

        el12.factGRBCM(i) = exp(2*models_grbcm{1,i}.hyp.cov(1));
        el12.factGRBCM(i) = exp(2*models_grbcm{1,i}.hyp.cov(2));
        sf2.factGRBCM(i) = exp(2*models_grbcm{1,i}.hyp.cov(3));
        sn2.factGRBCM(i) = exp(2*models_grbcm{1,i}.hyp.lik);
    end

    t.training.factGRBCM = etime(t6,t5) ;
    %% Consensus ADMM Training with gradient information
    [t.training.cADMM, theta.cADMM, iter.outer.cADMM, iter.inner.cADMM] = ...
        cADMM_2D(rho, alpha, opts, models, hyplength, cADMM_tol)

    for i=1:opts.Ms
        el1.cADMM(i) = exp(theta.cADMM(1,i));
        el2.cADMM(i) = exp(theta.cADMM(2,i));
        sf.cADMM(i) = exp(theta.cADMM(3,i));
        sn.cADMM(i) = exp(theta.cADMM(4,i));

        el12.cADMM(i) = exp(2*theta.cADMM(1,i));
        el22.cADMM(i) = exp(2*theta.cADMM(2,i));
        sf2.cADMM(i) = exp(2*theta.cADMM(3,i));
        sn2.cADMM(i) = exp(2*theta.cADMM(4,i));
    end
    %% Closed-form proximal ADMM Training with gradient information
    [t.training.cl_pxADMM, theta.cl_pxADMM, iter.outer.cl_pxADMM] = ...
        cl_pxADMM_2D(rho, opts, models, hyplength, Lip, cl_pxADMM_tol)

    for i=1:opts.Ms
        el1.cl_pxADMM(i) = exp(theta.cl_pxADMM(1,i));
        el2.cl_pxADMM(i) = exp(theta.cl_pxADMM(2,i));
        sf.cl_pxADMM(i) = exp(theta.cl_pxADMM(3,i));
        sn.cl_pxADMM(i) = exp(theta.cl_pxADMM(4,i));

        el12.cl_pxADMM(i) = exp(2*theta.cl_pxADMM(1,i));
        el22.cl_pxADMM(i) = exp(2*theta.cl_pxADMM(2,i));
        sf2.cl_pxADMM(i) = exp(2*theta.cl_pxADMM(3,i));
        sn2.cl_pxADMM(i) = exp(2*theta.cl_pxADMM(4,i));
    end
    %% Closed-form proximal ADMM Training with gradient information and communication dataset
    [t.training.cl_gpxADMM, theta.cl_gpxADMM, iter.outer.cl_gpxADMM] = ...
        cl_pxADMM_2D(rho, opts, models_grbcm, hyplength, Lip, cl_gpxADMM_tol)

    for i=1:opts.Ms
        el1.cl_gpxADMM(i) = exp(theta.cl_gpxADMM(1,i));
        el2.cl_gpxADMM(i) = exp(theta.cl_gpxADMM(2,i));
        sf.cl_gpxADMM(i) = exp(theta.cl_gpxADMM(3,i));
        sn.cl_gpxADMM(i) = exp(theta.cl_gpxADMM(4,i));

        el12.cl_gpxADMM(i) = exp(2*theta.cl_gpxADMM(1,i));
        el22.cl_gpxADMM(i) = exp(2*theta.cl_gpxADMM(2,i));
        sf2.cl_gpxADMM(i) = exp(2*theta.cl_gpxADMM(3,i));
        sn2.cl_gpxADMM(i) = exp(2*theta.cl_gpxADMM(4,i));
    end
    %% Decentralized exact consensus ADMM Training with gradient information
    [t.training.dec_cADMM, theta.dec_cADMM, iter.outer.dec_cADMM, iter.inner.dec_cADMM] = ...
        dec_cADMM_2D(rho, alpha, opts, models, hyplength, max_iter_dec_cADMM)

    for i=1:opts.Ms
        el1.dec_cADMM(i) = exp(theta.dec_cADMM(1,i));
        el2.dec_cADMM(i) = exp(theta.dec_cADMM(2,i));
        sf.dec_cADMM(i) = exp(theta.dec_cADMM(3,i));
        sn.dec_cADMM(i) = exp(theta.dec_cADMM(4,i));

        el12.dec_cADMM(i) = exp(2*theta.dec_cADMM(1,i));
        el22.dec_cADMM(i) = exp(2*theta.dec_cADMM(2,i));
        sf2.dec_cADMM(i) = exp(2*theta.dec_cADMM(3,i));
        sn2.dec_cADMM(i) = exp(2*theta.dec_cADMM(4,i));
    end
    %% Closed-form decentralized inexact (proximal) consensus ADMM Training with gradient information
    [t.training.dec_cl_pxADMM, theta.dec_cl_pxADMM, iter.outer.dec_cl_pxADMM] = ...
        dec_cl_pxADMM_2D(rho, opts, models, hyplength, Lip, max_iter_dec_cl_pxADMM)

    for i=1:opts.Ms
        el1.dec_cl_pxADMM(i) = exp(theta.dec_cl_pxADMM(1,i));
        el2.dec_cl_pxADMM(i) = exp(theta.dec_cl_pxADMM(2,i));
        sf.dec_cl_pxADMM(i) = exp(theta.dec_cl_pxADMM(3,i));
        sn.dec_cl_pxADMM(i) = exp(theta.dec_cl_pxADMM(4,i));

        el12.dec_cl_pxADMM(i) = exp(2*theta.dec_cl_pxADMM(1,i));
        el22.dec_cl_pxADMM(i) = exp(2*theta.dec_cl_pxADMM(2,i));
        sf2.dec_cl_pxADMM(i) = exp(2*theta.dec_cl_pxADMM(3,i));
        sn2.dec_cl_pxADMM(i) = exp(2*theta.dec_cl_pxADMM(4,i));
    end

    %% Closed-form decentralized inexact (proximal) consensus ADMM Training with gradient information and communication dataset
    [t.training.dec_cl_gpxADMM, theta.dec_cl_gpxADMM, iter.outer.dec_cl_gpxADMM] = ...
        dec_cl_pxADMM_2D(rho, opts, models_grbcm, hyplength, Lip, max_iter_dec_cl_gpxADMM)

    for i=1:opts.Ms
        el1.dec_cl_gpxADMM(i) = exp(theta.dec_cl_gpxADMM(1,i));
        el2.dec_cl_gpxADMM(i) = exp(theta.dec_cl_gpxADMM(2,i));
        sf.dec_cl_gpxADMM(i) = exp(theta.dec_cl_gpxADMM(3,i));
        sn.dec_cl_gpxADMM(i) = exp(theta.dec_cl_gpxADMM(4,i));

        el12.dec_cl_gpxADMM(i) = exp(2*theta.dec_cl_gpxADMM(1,i));
        el22.dec_cl_gpxADMM(i) = exp(2*theta.dec_cl_gpxADMM(2,i));
        sf2.dec_cl_gpxADMM(i) = exp(2*theta.dec_cl_gpxADMM(3,i));
        sn2.dec_cl_gpxADMM(i) = exp(2*theta.dec_cl_gpxADMM(4,i));
    end
    %% Store data
    el1.save.true(trial)             = el1.true;
    el1.save.fullGP(trial)           = el1.fullGP;
    el1.save.factGP(trial,:)         = el1.factGP;
    el1.save.factGRBCM(trial,:)      = el1.factGRBCM;
    el1.save.cADMM(trial,:)          = el1.cADMM;
    el1.save.cl_pxADMM(trial,:)      = el1.cl_pxADMM;
    el1.save.cl_gpxADMM(trial,:)     = el1.cl_gpxADMM;
    el1.save.dec_cADMM(trial,:)      = el1.dec_cADMM;
    el1.save.dec_cl_pxADMM(trial,:)  = el1.dec_cl_pxADMM;
    el1.save.dec_cl_gpxADMM(trial,:) = el1.dec_cl_gpxADMM;

    el2.save.true(trial)             = el2.true;
    el2.save.fullGP(trial)           = el2.fullGP;
    el2.save.factGP(trial,:)         = el2.factGP;
    el2.save.factGRBCM(trial,:)      = el2.factGRBCM;
    el2.save.cADMM(trial,:)          = el2.cADMM;
    el2.save.cl_pxADMM(trial,:)      = el2.cl_pxADMM;
    el2.save.cl_gpxADMM(trial,:)     = el2.cl_gpxADMM;
    el2.save.dec_cADMM(trial,:)      = el2.dec_cADMM;
    el2.save.dec_cl_pxADMM(trial,:)  = el2.dec_cl_pxADMM;
    el2.save.dec_cl_gpxADMM(trial,:) = el2.dec_cl_gpxADMM;

    sf.save.true(trial)             = sf.true;
    sf.save.fullGP(trial)           = sf.fullGP;
    sf.save.factGP(trial,:)         = sf.factGP;
    sf.save.factGRBCM(trial,:)      = sf.factGRBCM;
    sf.save.cADMM(trial,:)          = sf.cADMM;
    sf.save.cl_pxADMM(trial,:)      = sf.cl_pxADMM;
    sf.save.cl_gpxADMM(trial,:)     = sf.cl_gpxADMM;
    sf.save.dec_cADMM(trial,:)      = sf.dec_cADMM;
    sf.save.dec_cl_pxADMM(trial,:)  = sf.dec_cl_pxADMM;
    sf.save.dec_cl_gpxADMM(trial,:) = sf.dec_cl_gpxADMM;

    sn.save.true(trial)             = sn.true;
    sn.save.fullGP(trial)           = sn.fullGP;
    sn.save.factGP(trial,:)         = sn.factGP;
    sn.save.factGRBCM(trial,:)      = sn.factGRBCM;
    sn.save.cADMM(trial,:)          = sn.cADMM;
    sn.save.cl_pxADMM(trial,:)      = sn.cl_pxADMM;
    sn.save.cl_gpxADMM(trial,:)     = sn.cl_gpxADMM;
    sn.save.dec_cADMM(trial,:)      = sn.dec_cADMM;
    sn.save.dec_cl_pxADMM(trial,:)  = sn.dec_cl_pxADMM;
    sn.save.dec_cl_gpxADMM(trial,:) = sn.dec_cl_gpxADMM;

    t.save.fullGP(trial)         = t.training.fullGP;
    t.save.factGP(trial)         = t.training.factGP;
    t.save.factGRBCM(trial)      = t.training.factGRBCM;
    t.save.cADMM(trial)          = t.training.cADMM;
    t.save.cl_pxADMM(trial)      = t.training.cl_pxADMM;
    t.save.cl_gpxADMM(trial)     = t.training.cl_gpxADMM;
    t.save.dec_cADMM(trial)      = t.training.dec_cADMM;
    t.save.dec_cl_pxADMM(trial)  = t.training.dec_cl_pxADMM;
    t.save.dec_cl_gpxADMM(trial) = t.training.dec_cl_gpxADMM;

    iter.inner.save.cADMM(trial)      = iter.inner.cADMM;
    iter.inner.save.dec_cADMM(trial)  = iter.inner.dec_cADMM;

    iter.outer.save.factGP(trial)          = iter.outer.factGP;
    iter.outer.save.factGRBCM(trial)       = iter.outer.factGRBCM;
    iter.outer.save.cADMM(trial)          = iter.outer.cADMM;
    iter.outer.save.cl_pxADMM(trial)      = iter.outer.cl_pxADMM;
    iter.outer.save.cl_gpxADMM(trial)     = iter.outer.cl_gpxADMM;
    iter.outer.save.dec_cADMM(trial)      = iter.outer.dec_cADMM;
    iter.outer.save.dec_cl_pxADMM(trial)  = iter.outer.dec_cl_pxADMM;
    iter.outer.save.dec_cl_gpxADMM(trial) = iter.outer.dec_cl_gpxADMM;
    opts = [];
    trial
end

%% Plots
for i=1:trials
    el1.save.avg.factGP(i) = (1/agents)*sum(el1.save.factGP(i,:));
    el1.save.std.factGP(i) = std(el1.save.factGP(i,:));
    el1.save.avg.factGRBCM(i) = (1/agents)*sum(el1.save.factGRBCM(i,:));
    el1.save.std.factGRBCM(i) = std(el1.save.factGRBCM(i,:));
    el1.save.avg.cADMM(i) = (1/agents)*sum(el1.save.cADMM(i,:));
    el1.save.std.cADMM(i) = std(el1.save.cADMM(i,:));
    el1.save.avg.cl_pxADMM(i) = (1/agents)*sum(el1.save.cl_pxADMM(i,:));
    el1.save.std.cl_pxADMM(i) = std(el1.save.cl_pxADMM(i,:));
    el1.save.avg.cl_gpxADMM(i) = (1/agents)*sum(el1.save.cl_gpxADMM(i,:));
    el1.save.std.cl_gpxADMM(i) = std(el1.save.cl_gpxADMM(i,:));
    el1.save.avg.dec_cADMM(i) = (1/agents)*sum(el1.save.dec_cADMM(i,:));
    el1.save.std.dec_cADMM(i) = std(el1.save.dec_cADMM(i,:));
    el1.save.avg.dec_cl_pxADMM(i) = (1/agents)*sum(el1.save.dec_cl_pxADMM(i,:));
    el1.save.std.dec_cl_pxADMM(i) = std(el1.save.dec_cl_pxADMM(i,:));
    el1.save.avg.dec_cl_gpxADMM(i) = (1/agents)*sum(el1.save.dec_cl_gpxADMM(i,:));
    el1.save.std.dec_cl_gpxADMM(i) = std(el1.save.dec_cl_gpxADMM(i,:));

    el2.save.avg.factGP(i) = (1/agents)*sum(el2.save.factGP(i,:));
    el2.save.std.factGP(i) = std(el2.save.factGP(i,:));
    el2.save.avg.factGRBCM(i) = (1/agents)*sum(el2.save.factGRBCM(i,:));
    el2.save.std.factGRBCM(i) = std(el2.save.factGRBCM(i,:));
    el2.save.avg.cADMM(i) = (1/agents)*sum(el2.save.cADMM(i,:));
    el2.save.std.cADMM(i) = std(el2.save.cADMM(i,:));
    el2.save.avg.cl_pxADMM(i) = (1/agents)*sum(el2.save.cl_pxADMM(i,:));
    el2.save.std.cl_pxADMM(i) = std(el2.save.cl_pxADMM(i,:));
    el2.save.avg.cl_gpxADMM(i) = (1/agents)*sum(el2.save.cl_gpxADMM(i,:));
    el2.save.std.cl_gpxADMM(i) = std(el2.save.cl_gpxADMM(i,:));
    el2.save.avg.dec_cADMM(i) = (1/agents)*sum(el2.save.dec_cADMM(i,:));
    el2.save.std.dec_cADMM(i) = std(el2.save.dec_cADMM(i,:));
    el2.save.avg.dec_cl_pxADMM(i) = (1/agents)*sum(el2.save.dec_cl_pxADMM(i,:));
    el2.save.std.dec_cl_pxADMM(i) = std(el2.save.dec_cl_pxADMM(i,:));
    el2.save.avg.dec_cl_gpxADMM(i) = (1/agents)*sum(el2.save.dec_cl_gpxADMM(i,:));
    el2.save.std.dec_cl_gpxADMM(i) = std(el2.save.dec_cl_gpxADMM(i,:));

    sf.save.avg.factGP(i) = (1/agents)*sum(sf.save.factGP(i,:));
    sf.save.std.factGP(i) = std(sf.save.factGP(i,:));
    sf.save.avg.factGRBCM(i) = (1/agents)*sum(sf.save.factGRBCM(i,:));
    sf.save.std.factGRBCM(i) = std(sf.save.factGRBCM(i,:));
    sf.save.avg.cADMM(i) = (1/agents)*sum(sf.save.cADMM(i,:));
    sf.save.std.cADMM(i) = std(sf.save.cADMM(i,:));
    sf.save.avg.cl_pxADMM(i) = (1/agents)*sum(sf.save.cl_pxADMM(i,:));
    sf.save.std.cl_pxADMM(i) = std(sf.save.cl_pxADMM(i,:));
    sf.save.avg.cl_gpxADMM(i) = (1/agents)*sum(sf.save.cl_gpxADMM(i,:));
    sf.save.std.cl_gpxADMM(i) = std(sf.save.cl_gpxADMM(i,:));
    sf.save.avg.dec_cADMM(i) = (1/agents)*sum(sf.save.dec_cADMM(i,:));
    sf.save.std.dec_cADMM(i) = std(sf.save.dec_cADMM(i,:));
    sf.save.avg.dec_cl_pxADMM(i) = (1/agents)*sum(sf.save.dec_cl_pxADMM(i,:));
    sf.save.std.dec_cl_pxADMM(i) = std(sf.save.dec_cl_pxADMM(i,:));
    sf.save.avg.dec_cl_gpxADMM(i) = (1/agents)*sum(sf.save.dec_cl_gpxADMM(i,:));
    sf.save.std.dec_cl_gpxADMM(i) = std(sf.save.dec_cl_gpxADMM(i,:));

    sn.save.avg.factGP(i) = (1/agents)*sum(sn.save.factGP(i,:));
    sn.save.std.factGP(i) = std(sn.save.factGP(i,:));
    sn.save.avg.factGRBCM(i) = (1/agents)*sum(sn.save.factGRBCM(i,:));
    sn.save.std.factGRBCM(i) = std(sn.save.factGRBCM(i,:));
    sn.save.avg.cADMM(i) = (1/agents)*sum(sn.save.cADMM(i,:));
    sn.save.std.cADMM(i) = std(sn.save.cADMM(i,:));
    sn.save.avg.cl_pxADMM(i) = (1/agents)*sum(sn.save.cl_pxADMM(i,:));
    sn.save.std.cl_pxADMM(i) = std(sn.save.cl_pxADMM(i,:));
    sn.save.avg.cl_gpxADMM(i) = (1/agents)*sum(sn.save.cl_gpxADMM(i,:));
    sn.save.std.cl_gpxADMM(i) = std(sn.save.cl_gpxADMM(i,:));
    sn.save.avg.dec_cADMM(i) = (1/agents)*sum(sn.save.dec_cADMM(i,:));
    sn.save.std.dec_cADMM(i) = std(sn.save.dec_cADMM(i,:));
    sn.save.avg.dec_cl_pxADMM(i) = (1/agents)*sum(sn.save.dec_cl_pxADMM(i,:));
    sn.save.std.dec_cl_pxADMM(i) = std(sn.save.dec_cl_pxADMM(i,:));
    sn.save.avg.dec_cl_gpxADMM(i) = (1/agents)*sum(sn.save.dec_cl_gpxADMM(i,:));
    sn.save.std.dec_cl_gpxADMM(i) = std(sn.save.dec_cl_gpxADMM(i,:));
end

% Mean absolute error (MAE)
MAE.el1.avg.fullGP = mean(abs(el1.save.fullGP - el1.save.true));
MAE.el1.std.fullGP = std(abs(el1.save.fullGP - el1.save.true));
MAE.el1.avg.factGP = mean(abs(el1.save.avg.factGP - el1.save.true));
MAE.el1.std.factGP = std(abs(el1.save.avg.factGP - el1.save.true));
MAE.el1.avg.factGRBCM = mean(abs(el1.save.avg.factGRBCM - el1.save.true));
MAE.el1.std.factGRBCM = std(abs(el1.save.avg.factGRBCM - el1.save.true));
MAE.el1.avg.cADMM = mean(abs(el1.save.avg.cADMM - el1.save.true));
MAE.el1.std.cADMM = std(abs(el1.save.avg.cADMM - el1.save.true));
MAE.el1.avg.cl_pxADMM = mean(abs(el1.save.avg.cl_pxADMM - el1.save.true));
MAE.el1.std.cl_pxADMM = std(abs(el1.save.avg.cl_pxADMM - el1.save.true));
MAE.el1.avg.cl_gpxADMM = mean(abs(el1.save.avg.cl_gpxADMM - el1.save.true));
MAE.el1.std.cl_gpxADMM = std(abs(el1.save.avg.cl_gpxADMM - el1.save.true));
MAE.el1.avg.dec_cADMM = mean(abs(el1.save.avg.dec_cADMM - el1.save.true));
MAE.el1.std.dec_cADMM = std(abs(el1.save.avg.dec_cADMM - el1.save.true));
MAE.el1.avg.dec_cl_pxADMM = mean(abs(el1.save.avg.dec_cl_pxADMM - el1.save.true));
MAE.el1.std.dec_cl_pxADMM = std(abs(el1.save.avg.dec_cl_pxADMM - el1.save.true));
MAE.el1.avg.dec_cl_gpxADMM = mean(abs(el1.save.avg.dec_cl_gpxADMM - el1.save.true));
MAE.el1.std.dec_cl_gpxADMM = std(abs(el1.save.avg.dec_cl_gpxADMM - el1.save.true));

MAE.el2.avg.fullGP = mean(abs(el2.save.fullGP - el2.save.true));
MAE.el2.std.fullGP = std(abs(el2.save.fullGP - el2.save.true));
MAE.el2.avg.factGP = mean(abs(el2.save.avg.factGP - el2.save.true));
MAE.el2.std.factGP = std(abs(el2.save.avg.factGP - el2.save.true));
MAE.el2.avg.factGRBCM = mean(abs(el2.save.avg.factGRBCM - el2.save.true));
MAE.el2.std.factGRBCM = std(abs(el2.save.avg.factGRBCM - el2.save.true));
MAE.el2.avg.cADMM = mean(abs(el2.save.avg.cADMM - el2.save.true));
MAE.el2.std.cADMM = std(abs(el2.save.avg.cADMM - el2.save.true));
MAE.el2.avg.cl_pxADMM = mean(abs(el2.save.avg.cl_pxADMM - el2.save.true));
MAE.el2.std.cl_pxADMM = std(abs(el2.save.avg.cl_pxADMM - el2.save.true));
MAE.el2.avg.cl_gpxADMM = mean(abs(el2.save.avg.cl_gpxADMM - el2.save.true));
MAE.el2.std.cl_gpxADMM = std(abs(el2.save.avg.cl_gpxADMM - el2.save.true));
MAE.el2.avg.dec_cADMM = mean(abs(el2.save.avg.dec_cADMM - el2.save.true));
MAE.el2.std.dec_cADMM = std(abs(el2.save.avg.dec_cADMM - el2.save.true));
MAE.el2.avg.dec_cl_pxADMM = mean(abs(el2.save.avg.dec_cl_pxADMM - el2.save.true));
MAE.el2.std.dec_cl_pxADMM = std(abs(el2.save.avg.dec_cl_pxADMM - el2.save.true));
MAE.el2.avg.dec_cl_gpxADMM = mean(abs(el2.save.avg.dec_cl_gpxADMM - el2.save.true));
MAE.el2.std.dec_cl_gpxADMM = std(abs(el2.save.avg.dec_cl_gpxADMM - el2.save.true));

MAE.sf.avg.fullGP = mean(abs(sf.save.fullGP - sf.save.true));
MAE.sf.std.fullGP = std(abs(sf.save.fullGP - sf.save.true));
MAE.sf.avg.factGP = mean(abs(sf.save.avg.factGP - sf.save.true));
MAE.sf.std.factGP = std(abs(sf.save.avg.factGP - sf.save.true));
MAE.sf.avg.factGRBCM = mean(abs(sf.save.avg.factGRBCM - sf.save.true));
MAE.sf.std.factGRBCM = std(abs(sf.save.avg.factGRBCM - sf.save.true));
MAE.sf.avg.cADMM = mean(abs(sf.save.avg.cADMM - sf.save.true));
MAE.sf.std.cADMM = std(abs(sf.save.avg.cADMM - sf.save.true));
MAE.sf.avg.cl_pxADMM = mean(abs(sf.save.avg.cl_pxADMM - sf.save.true));
MAE.sf.std.cl_pxADMM = std(abs(sf.save.avg.cl_pxADMM - sf.save.true));
MAE.sf.avg.cl_gpxADMM = mean(abs(sf.save.avg.cl_gpxADMM - sf.save.true));
MAE.sf.std.cl_gpxADMM = std(abs(sf.save.avg.cl_gpxADMM - sf.save.true));
MAE.sf.avg.dec_cADMM = mean(abs(sf.save.avg.dec_cADMM - sf.save.true));
MAE.sf.std.dec_cADMM = std(abs(sf.save.avg.dec_cADMM - sf.save.true));
MAE.sf.avg.dec_cl_pxADMM = mean(abs(sf.save.avg.dec_cl_pxADMM - sf.save.true));
MAE.sf.std.dec_cl_pxADMM = std(abs(sf.save.avg.dec_cl_pxADMM - sf.save.true));
MAE.sf.avg.dec_cl_gpxADMM = mean(abs(sf.save.avg.dec_cl_gpxADMM - sf.save.true));
MAE.sf.std.dec_cl_gpxADMM = std(abs(sf.save.avg.dec_cl_gpxADMM - sf.save.true));

MAE.sn.avg.fullGP = mean(abs(sn.save.fullGP - sn.save.true));
MAE.sn.std.fullGP = std(abs(sn.save.fullGP - sn.save.true));
MAE.sn.avg.factGP = mean(abs(sn.save.avg.factGP - sn.save.true));
MAE.sn.std.factGP = std(abs(sn.save.avg.factGP - sn.save.true));
MAE.sn.avg.factGRBCM = mean(abs(sn.save.avg.factGRBCM - sn.save.true));
MAE.sn.std.factGRBCM = std(abs(sn.save.avg.factGRBCM - sn.save.true));
MAE.sn.avg.cADMM = mean(abs(sn.save.avg.cADMM - sn.save.true));
MAE.sn.std.cADMM = std(abs(sn.save.avg.cADMM - sn.save.true));
MAE.sn.avg.cl_pxADMM = mean(abs(sn.save.avg.cl_pxADMM - sn.save.true));
MAE.sn.std.cl_pxADMM = std(abs(sn.save.avg.cl_pxADMM - sn.save.true));
MAE.sn.avg.cl_gpxADMM = mean(abs(sn.save.avg.cl_gpxADMM - sn.save.true));
MAE.sn.std.cl_gpxADMM = std(abs(sn.save.avg.cl_gpxADMM - sn.save.true));
MAE.sn.avg.dec_cADMM = mean(abs(sn.save.avg.dec_cADMM - sn.save.true));
MAE.sn.std.dec_cADMM = std(abs(sn.save.avg.dec_cADMM - sn.save.true));
MAE.sn.avg.dec_cl_pxADMM = mean(abs(sn.save.avg.dec_cl_pxADMM - sn.save.true));
MAE.sn.std.dec_cl_pxADMM = std(abs(sn.save.avg.dec_cl_pxADMM - sn.save.true));
MAE.sn.avg.dec_cl_gpxADMM = mean(abs(sn.save.avg.dec_cl_gpxADMM - sn.save.true));
MAE.sn.std.dec_cl_gpxADMM = std(abs(sn.save.avg.dec_cl_gpxADMM - sn.save.true));

RMSE.el1.avg.fullGP = sqrt((1/trials)*sum((el1.save.fullGP - el1.save.true).^2));
RMSE.el1.avg.factGP = sqrt((1/trials)*sum((el1.save.avg.factGP - el1.save.true).^2));
RMSE.el1.avg.factGRBCM = sqrt((1/trials)*sum((el1.save.avg.factGRBCM - el1.save.true).^2));
RMSE.el1.avg.cADMM = sqrt((1/trials)*sum((el1.save.avg.cADMM - el1.save.true).^2));
RMSE.el1.avg.cl_pxADMM = sqrt((1/trials)*sum((el1.save.avg.cl_pxADMM - el1.save.true).^2));
RMSE.el1.avg.cl_gpxADMM = sqrt((1/trials)*sum((el1.save.avg.cl_gpxADMM - el1.save.true).^2));
RMSE.el1.avg.dec_cADMM = sqrt((1/trials)*sum((el1.save.avg.dec_cADMM - el1.save.true).^2));
RMSE.el1.avg.dec_cl_pxADMM = sqrt((1/trials)*sum((el1.save.avg.dec_cl_pxADMM - el1.save.true).^2));
RMSE.el1.avg.dec_cl_gpxADMM = sqrt((1/trials)*sum((el1.save.avg.dec_cl_gpxADMM - el1.save.true).^2));

RMSE.el2.avg.fullGP =sqrt((1/trials)*sum((el2.save.fullGP - el2.save.true).^2));
RMSE.el2.avg.factGP =sqrt((1/trials)*sum((el2.save.avg.factGP - el2.save.true).^2));
RMSE.el2.avg.factGRBCM =sqrt((1/trials)*sum((el2.save.avg.factGRBCM - el2.save.true).^2));
RMSE.el2.avg.cADMM =sqrt((1/trials)*sum((el2.save.avg.cADMM - el2.save.true).^2));
RMSE.el2.avg.cl_pxADMM =sqrt((1/trials)*sum((el2.save.avg.cl_pxADMM - el2.save.true).^2));
RMSE.el2.avg.cl_gpxADMM =sqrt((1/trials)*sum((el2.save.avg.cl_gpxADMM - el2.save.true).^2));
RMSE.el2.avg.dec_cADMM =sqrt((1/trials)*sum((el2.save.avg.dec_cADMM - el2.save.true).^2));
RMSE.el2.avg.dec_cl_pxADMM =sqrt((1/trials)*sum((el2.save.avg.dec_cl_pxADMM - el2.save.true).^2));
RMSE.el2.avg.dec_cl_gpxADMM =sqrt((1/trials)*sum((el2.save.avg.dec_cl_gpxADMM - el2.save.true).^2));

RMSE.sf.avg.fullGP =sqrt((1/trials)*sum((sf.save.fullGP - sf.save.true).^2));
RMSE.sf.avg.factGP =sqrt((1/trials)*sum((sf.save.avg.factGP - sf.save.true).^2));
RMSE.sf.avg.factGRBCM =sqrt((1/trials)*sum((sf.save.avg.factGRBCM - sf.save.true).^2));
RMSE.sf.avg.cADMM =sqrt((1/trials)*sum((sf.save.avg.cADMM - sf.save.true).^2));
RMSE.sf.avg.cl_pxADMM =sqrt((1/trials)*sum((sf.save.avg.cl_pxADMM - sf.save.true).^2));
RMSE.sf.avg.cl_gpxADMM =sqrt((1/trials)*sum((sf.save.avg.cl_gpxADMM - sf.save.true).^2));
RMSE.sf.avg.dec_cADMM =sqrt((1/trials)*sum((sf.save.avg.dec_cADMM - sf.save.true).^2));
RMSE.sf.avg.dec_cl_pxADMM =sqrt((1/trials)*sum((sf.save.avg.dec_cl_pxADMM - sf.save.true).^2));
RMSE.sf.avg.dec_cl_gpxADMM =sqrt((1/trials)*sum((sf.save.avg.dec_cl_gpxADMM - sf.save.true).^2));

RMSE.sn.avg.fullGP =sqrt((1/trials)*sum((sn.save.fullGP - sn.save.true).^2));
RMSE.sn.avg.factGP =sqrt((1/trials)*sum((sn.save.avg.factGP - sn.save.true).^2));
RMSE.sn.avg.factGRBCM =sqrt((1/trials)*sum((sn.save.avg.factGRBCM - sn.save.true).^2));
RMSE.sn.avg.cADMM =sqrt((1/trials)*sum((sn.save.avg.cADMM - sn.save.true).^2));
RMSE.sn.avg.cl_pxADMM =sqrt((1/trials)*sum((sn.save.avg.cl_pxADMM - sn.save.true).^2));
RMSE.sn.avg.cl_gpxADMM =sqrt((1/trials)*sum((sn.save.avg.cl_gpxADMM - sn.save.true).^2));
RMSE.sn.avg.dec_cADMM =sqrt((1/trials)*sum((sn.save.avg.dec_cADMM - sn.save.true).^2));
RMSE.sn.avg.dec_cl_pxADMM =sqrt((1/trials)*sum((sn.save.avg.dec_cl_pxADMM - sn.save.true).^2));
RMSE.sn.avg.dec_cl_gpxADMM =sqrt((1/trials)*sum((sn.save.avg.dec_cl_gpxADMM - sn.save.true).^2));

t.save.avg.fullGP = mean(t.save.fullGP);
t.save.std.fullGP = std(t.save.fullGP);
t.save.avg.factGP = (1/agents)*mean(t.save.factGP);
t.save.std.factGP = std(t.save.factGP);
t.save.avg.factGRBCM = (1/agents)*mean(t.save.factGRBCM);
t.save.std.factGRBCM = std(t.save.factGRBCM);
t.save.avg.cADMM = (1/agents)*mean(t.save.cADMM);
t.save.std.cADMM = std(t.save.cADMM);
t.save.avg.cl_pxADMM = (1/agents)*mean(t.save.cl_pxADMM);
t.save.std.cl_pxADMM = std(t.save.cl_pxADMM);
t.save.avg.cl_gpxADMM = (1/agents)*mean(t.save.cl_gpxADMM);
t.save.std.cl_gpxADMM = std(t.save.cl_gpxADMM);
t.save.avg.dec_cADMM = (1/agents)*mean(t.save.dec_cADMM);
t.save.std.dec_cADMM = std(t.save.dec_cADMM);
t.save.avg.dec_cl_pxADMM = (1/agents)*mean(t.save.dec_cl_pxADMM);
t.save.std.dec_cl_pxADMM = std(t.save.dec_cl_pxADMM);
t.save.avg.dec_cl_gpxADMM = (1/agents)*mean(t.save.dec_cl_gpxADMM);
t.save.std.dec_cl_gpxADMM = std(t.save.dec_cl_gpxADMM);

iter.outer.save.avg.factGP = mean(iter.outer.save.factGP);
iter.outer.save.std.factGP = std(iter.outer.save.factGP);
iter.outer.save.avg.factGRBCM = mean(iter.outer.save.factGRBCM);
iter.outer.save.std.factGRBCM = std(iter.outer.save.factGRBCM);
iter.outer.save.avg.cADMM = mean(iter.outer.save.cADMM);
iter.outer.save.std.cADMM = std(iter.outer.save.cADMM);
iter.outer.save.avg.cl_pxADMM = mean(iter.outer.save.cl_pxADMM);
iter.outer.save.std.cl_pxADMM = std(iter.outer.save.cl_pxADMM);
iter.outer.save.avg.cl_gpxADMM = mean(iter.outer.save.cl_gpxADMM);
iter.outer.save.std.cl_gpxADMM = std(iter.outer.save.cl_gpxADMM);
iter.outer.save.avg.dec_cADMM = mean(iter.outer.save.dec_cADMM);
iter.outer.save.std.dec_cADMM = std(iter.outer.save.dec_cADMM);
iter.outer.save.avg.dec_cl_pxADMM = mean(iter.outer.save.dec_cl_pxADMM);
iter.outer.save.std.dec_cl_pxADMM = std(iter.outer.save.dec_cl_pxADMM);
iter.outer.save.avg.dec_cl_gpxADMM = mean(iter.outer.save.dec_cl_gpxADMM);
iter.outer.save.std.dec_cl_gpxADMM = std(iter.outer.save.dec_cl_gpxADMM);

iter.inner.save.avg.cADMM = mean(iter.inner.save.cADMM);
iter.inner.save.std.cADMM = std(iter.inner.save.cADMM);
iter.inner.save.avg.dec_cADMM = mean(iter.inner.save.dec_cADMM);
iter.inner.save.std.dec_cADMM = std(iter.inner.save.dec_cADMM);

el1.save.avg.all = [el1.save.fullGP;...
    el1.save.avg.factGP;...
    el1.save.avg.factGRBCM;...
    el1.save.avg.cADMM; ...
    el1.save.avg.cl_pxADMM;...
    el1.save.avg.cl_gpxADMM;...
    el1.save.avg.dec_cADMM;...
    el1.save.avg.dec_cl_pxADMM;...
    el1.save.avg.dec_cl_gpxADMM...
    ];

el2.save.avg.all = [el2.save.fullGP;...
    el2.save.avg.factGP;...
    el2.save.avg.factGRBCM;...
    el2.save.avg.cADMM; ...
    el2.save.avg.cl_pxADMM;...
    el2.save.avg.cl_gpxADMM;...
    el2.save.avg.dec_cADMM;...
    el2.save.avg.dec_cl_pxADMM;...
    el2.save.avg.dec_cl_gpxADMM...
    ];

sf.save.avg.all = [sf.save.fullGP;...
    sf.save.avg.factGP;...
    sf.save.avg.factGRBCM;...
    sf.save.avg.cADMM; ...
    sf.save.avg.cl_pxADMM;...
    sf.save.avg.cl_gpxADMM;...
    sf.save.avg.dec_cADMM;...
    sf.save.avg.dec_cl_pxADMM;...
    sf.save.avg.dec_cl_gpxADMM...
    ];

sn.save.avg.all = [sn.save.fullGP;...
    sn.save.avg.factGP;...
    sn.save.avg.factGRBCM;...
    sn.save.avg.cADMM; ...
    sn.save.avg.cl_pxADMM;...
    sn.save.avg.cl_gpxADMM;...
    sn.save.avg.dec_cADMM;...
    sn.save.avg.dec_cl_pxADMM;...
    sn.save.avg.dec_cl_gpxADMM...
    ];

t.save.avg.all = [t.save.avg.fullGP;...
    t.save.avg.factGP;...
    t.save.avg.factGRBCM;...
    t.save.avg.cADMM; ...
    t.save.avg.cl_pxADMM;...
    t.save.avg.cl_gpxADMM;...
    t.save.avg.dec_cADMM;...
    t.save.avg.dec_cl_pxADMM;...
    t.save.avg.dec_cl_gpxADMM...
    ];
t.save.std.all = [t.save.std.fullGP;...
    t.save.std.factGP;...
    t.save.std.factGRBCM;...
    t.save.std.cADMM; ...
    t.save.std.cl_pxADMM;...
    t.save.std.cl_gpxADMM;...
    t.save.std.dec_cADMM;...
    t.save.std.dec_cl_pxADMM;...
    t.save.std.dec_cl_gpxADMM...
    ];

iter.outer.save.avg.all = [iter.outer.save.avg.factGP;...
    iter.outer.save.avg.factGRBCM;...
    iter.outer.save.avg.cADMM; ...
    iter.outer.save.avg.cl_pxADMM;...
    iter.outer.save.avg.cl_gpxADMM;...
    iter.outer.save.avg.dec_cADMM;...
    iter.outer.save.avg.dec_cl_pxADMM;...
    iter.outer.save.avg.dec_cl_gpxADMM...
    ];
iter.outer.save.std.all = [iter.outer.save.std.factGP;...
    iter.outer.save.std.factGRBCM;...
    iter.outer.save.std.cADMM; ...
    iter.outer.save.std.cl_pxADMM;...
    iter.outer.save.std.cl_gpxADMM;...
    iter.outer.save.std.dec_cADMM;...
    iter.outer.save.std.dec_cl_pxADMM;...
    iter.outer.save.std.dec_cl_gpxADMM...
    ];

iter.inner.save.avg.all = [iter.inner.save.avg.cADMM; ...
    iter.inner.save.avg.dec_cADMM];
iter.inner.save.std.all = [iter.inner.save.std.cADMM; ...
    iter.inner.save.std.dec_cADMM];

MAE.el1.save.avg.all = [MAE.el1.avg.fullGP;...
    MAE.el1.avg.factGP;...
    MAE.el1.avg.factGRBCM; ...
    MAE.el1.avg.cADMM; ...
    MAE.el1.avg.cl_pxADMM;...
    MAE.el1.avg.cl_gpxADMM;...
    MAE.el1.avg.dec_cADMM; ...
    MAE.el1.avg.dec_cl_pxADMM;...
    MAE.el1.avg.dec_cl_gpxADMM...
    ];
MAE.el1.save.std.all = [MAE.el1.std.fullGP;...
    MAE.el1.std.factGP;...
    MAE.el1.std.factGRBCM;...
    MAE.el1.std.cADMM; ...
    MAE.el1.std.cl_pxADMM;...
    MAE.el1.std.cl_gpxADMM;...
    MAE.el1.std.dec_cADMM; ...
    MAE.el1.std.dec_cl_pxADMM;...
    MAE.el1.std.dec_cl_gpxADMM...
    ];

MAE.el2.save.avg.all = [MAE.el2.avg.fullGP;...
    MAE.el2.avg.factGP;...
    MAE.el2.avg.factGRBCM; ...
    MAE.el2.avg.cADMM; ...
    MAE.el2.avg.cl_pxADMM;...
    MAE.el2.avg.cl_gpxADMM;...
    MAE.el2.avg.dec_cADMM; ...
    MAE.el2.avg.dec_cl_pxADMM;...
    MAE.el2.avg.dec_cl_gpxADMM...
    ];
MAE.el2.save.std.all = [MAE.el2.std.fullGP;...
    MAE.el2.std.factGP;...
    MAE.el2.std.factGRBCM;...
    MAE.el2.std.cADMM; ...
    MAE.el2.std.cl_pxADMM;...
    MAE.el2.std.cl_gpxADMM;...
    MAE.el2.std.dec_cADMM; ...
    MAE.el2.std.dec_cl_pxADMM;...
    MAE.el2.std.dec_cl_gpxADMM...
    ];

MAE.sf.save.avg.all = [MAE.sf.avg.fullGP;...
    MAE.sf.avg.factGP;...
    MAE.sf.avg.factGRBCM; ...
    MAE.sf.avg.cADMM; ...
    MAE.sf.avg.cl_pxADMM;...
    MAE.sf.avg.cl_gpxADMM;...
    MAE.sf.avg.dec_cADMM; ...
    MAE.sf.avg.dec_cl_pxADMM;...
    MAE.sf.avg.dec_cl_gpxADMM...
    ];
MAE.sf.save.std.all = [MAE.sf.std.fullGP;...
    MAE.sf.std.factGP;...
    MAE.sf.std.factGRBCM;...
    MAE.sf.std.cADMM; ...
    MAE.sf.std.cl_pxADMM;...
    MAE.sf.std.cl_gpxADMM;...
    MAE.sf.std.dec_cADMM; ...
    MAE.sf.std.dec_cl_pxADMM;...
    MAE.sf.std.dec_cl_gpxADMM...
    ];

MAE.sn.save.avg.all = [MAE.sn.avg.fullGP;...
    MAE.sn.avg.factGP;...
    MAE.sn.avg.factGRBCM; ...
    MAE.sn.avg.cADMM; ...
    MAE.sn.avg.cl_pxADMM;...
    MAE.sn.avg.cl_gpxADMM;...
    MAE.sn.avg.dec_cADMM; ...
    MAE.sn.avg.dec_cl_pxADMM;...
    MAE.sn.avg.dec_cl_gpxADMM...
    ];
MAE.sn.save.std.all = [MAE.sn.std.fullGP;...
    MAE.sn.std.factGP;...
    MAE.sn.std.factGRBCM;...
    MAE.sn.std.cADMM; ...
    MAE.sn.std.cl_pxADMM;...
    MAE.sn.std.cl_gpxADMM;...
    MAE.sn.std.dec_cADMM; ...
    MAE.sn.std.dec_cl_pxADMM;...
    MAE.sn.std.dec_cl_gpxADMM...
    ];

RMSE.el1.save.avg.all = [RMSE.el1.avg.fullGP; ...
    RMSE.el1.avg.factGP;...
    RMSE.el1.avg.factGRBCM;...
    RMSE.el1.avg.cADMM; ...
    RMSE.el1.avg.cl_pxADMM;...
    RMSE.el1.avg.cl_gpxADMM;...
    RMSE.el1.avg.dec_cADMM; ...
    RMSE.el1.avg.dec_cl_pxADMM;...
    RMSE.el1.avg.dec_cl_gpxADMM...
    ];

RMSE.el2.save.avg.all = [RMSE.el2.avg.fullGP; ...
    RMSE.el2.avg.factGP;...
    RMSE.el2.avg.factGRBCM;...
    RMSE.el2.avg.cADMM; ...
    RMSE.el2.avg.cl_pxADMM;...
    RMSE.el2.avg.cl_gpxADMM;...
    RMSE.el2.avg.dec_cADMM; ...
    RMSE.el2.avg.dec_cl_pxADMM;...
    RMSE.el2.avg.dec_cl_gpxADMM...
    ];

RMSE.sf.save.avg.all = [RMSE.sf.avg.fullGP; ...
    RMSE.sf.avg.factGP;...
    RMSE.sf.avg.factGRBCM;...
    RMSE.sf.avg.cADMM; ...
    RMSE.sf.avg.cl_pxADMM;...
    RMSE.sf.avg.cl_gpxADMM;...
    RMSE.sf.avg.dec_cADMM; ...
    RMSE.sf.avg.dec_cl_pxADMM;...
    RMSE.sf.avg.dec_cl_gpxADMM...
    ];

RMSE.sn.save.avg.all = [RMSE.sn.avg.fullGP;...
    RMSE.sn.avg.factGP;...
    RMSE.sn.avg.factGRBCM;...
    RMSE.sn.avg.cADMM; ...
    RMSE.sn.avg.cl_pxADMM;...
    RMSE.sn.avg.cl_gpxADMM;...
    RMSE.sn.avg.dec_cADMM; ...
    RMSE.sn.avg.dec_cl_pxADMM;...
    RMSE.sn.avg.dec_cl_gpxADMM...
    ];

%%
orange_VT = [.909 .467 .133];
maroon_VT = [.506 .117 .245];

bp1 = figure('position',[50    50   1200   400])
subplot(1,4,1)
boxplot(el1.save.avg.all','Labels',{'FULL-GP', 'FACT-GP','g-FACT-GP',...
    'c-ADMM-GP','apx-ADMM-GP',...
    'gapx-ADMM-GP',...
    'D-c-ADMM-GP','D-apx-ADMM-GP',...
    'D-gapx-ADMM-GP'...
    });
set(gca,'XTickLabelRotation',90)
lines = findobj(gcf, 'type', 'line', 'Tag', 'Box');
set(lines(1:4), 'Color', maroon_VT);
ylabel('Lengthscale 1');
med_lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(med_lines, 'Color', orange_VT);
set(findobj(gca,'type','line'),'linew',1.5)
ylim([0.5 2.5])
hAx=gca;                                   % retrieve the axes handle
xtk=hAx.XTick;                             % and the xtick values to plot() at...
hold on
hL=plot(xtk,ones(1,length(xtk))*el1.save.true(1),'k--');
box on; grid off; hold on;

subplot(1,4,2)
boxplot(el2.save.avg.all','Labels',{'FULL-GP', 'FACT-GP','g-FACT-GP',...
    'c-ADMM-GP','apx-ADMM-GP',...
    'gapx-ADMM-GP',...
    'D-c-ADMM-GP','D-apx-ADMM-GP',...
    'D-gapx-ADMM-GP'...
    });
set(gca,'XTickLabelRotation',90)
ylabel('Lengthscale 2');
ylim([0.25 0.55])
hAx=gca;                                   % retrieve the axes handle
xtk=hAx.XTick;                             % and the xtick values to plot() at...
hold on
hL=plot(xtk,ones(1,length(xtk))*el2.save.true(1),'k--');
box on; grid off; hold on;

subplot(1,4,3)
boxplot(sf.save.avg.all','Labels',{'FULL-GP', 'FACT-GP','g-FACT-GP',...
    'c-ADMM-GP','apx-ADMM-GP',...
    'gapx-ADMM-GP',...
    'D-c-ADMM-GP','D-apx-ADMM-GP',...
    'D-gapx-ADMM-GP'...
    });
set(gca,'XTickLabelRotation',90)
ylabel('Signal std');
ylim([0.9, 2.2]);
hAx=gca;                                   % retrieve the axes handle
xtk=hAx.XTick;                             % and the xtick values to plot() at...
hold on
hL=plot(xtk,ones(1,length(xtk))*sf.save.true(1),'k--');
box on; grid off; hold on;

subplot(1,4,4)
boxplot(sn.save.avg.all','Labels',{'FULL-GP', 'FACT-GP','g-FACT-GP',...
    'c-ADMM-GP','apx-ADMM-GP',...
    'gapx-ADMM-GP',...
    'D-c-ADMM-GP','D-apx-ADMM-GP',...
    'D-gapx-ADMM-GP'...
    });
set(gca,'XTickLabelRotation',90)
ylabel('Noise std');
ylim([0.05, 1.05]);
hAx=gca;                                   % retrieve the axes handle
xtk=hAx.XTick;                             % and the xtick values to plot() at...
hold on
hL=plot(xtk,ones(1,length(xtk))*sn.save.true(1),'k--');
box on; grid off; hold on;

saveas(bp1,['figures/boxplots_g_obs_' num2str(n) '_input_2D_hyp_4_agents_' num2str(agents) '_repl_' num2str(trials) '.png'],'png');
saveas(bp1,['figures/boxplots_g_obs_' num2str(n) '_input_2D_hyp_4_agents_' num2str(agents) '_repl_' num2str(trials) '.fig'],'fig');

%% Sava data
% Computational time
fmt = sprintf('%s\n',repmat( ' %u',1,size(t.save.avg.all,1)));
fid = fopen(['data/time/training_time_avg_agents_' num2str(agents) '_obs_' num2str(n) '_repl_' num2str(trials) '.dat'],'wt');
fprintf(fid,fmt,t.save.avg.all);
fclose(fid);

fmt = sprintf('%s\n',repmat( ' %u',1,size(t.save.std.all,1)));
fid = fopen(['data/time/training_time_std_agents_' num2str(agents) '_obs_' num2str(n) '_repl_' num2str(trials) '.dat'],'wt');
fprintf(fid,fmt,t.save.std.all);
fclose(fid);

% Communications
fmt = sprintf('%s\n',repmat( ' %u',1,size(iter.outer.save.avg.all,1)));
fid = fopen(['data/communications/communications_avg_agents_' num2str(agents) '_obs_' num2str(n) '_repl_' num2str(trials) '.dat'],'wt');
fprintf(fid,fmt,iter.outer.save.avg.all);
fclose(fid);

fmt = sprintf('%s\n',repmat( ' %u',1,size(iter.outer.save.std.all,1)));
fid = fopen(['data/communications/communications_std_agents_' num2str(agents) '_obs_' num2str(n) '_repl_' num2str(trials) '.dat'],'wt');
fprintf(fid,fmt,iter.outer.save.std.all);
fclose(fid);

% Inner iterations
fmt = sprintf('%s\n',repmat( ' %u',1,size(iter.inner.save.avg.all,1)));
fid = fopen(['data/inner_iterations/inner_iterations_avg_agents_' num2str(agents) '_obs_' num2str(n) '_repl_' num2str(trials) '.dat'],'wt');
fprintf(fid,fmt,iter.inner.save.avg.all);
fclose(fid);

fmt = sprintf('%s\n',repmat( ' %u',1,size(iter.inner.save.std.all,1)));
fid = fopen(['data/inner_iterations/inner_iterations_std_agents_' num2str(agents) '_obs_' num2str(n) '_repl_' num2str(trials) '.dat'],'wt');
fprintf(fid,fmt,iter.inner.save.std.all);
fclose(fid);

% RMSE hyperparameters
fmt = sprintf('%s\n',repmat( ' %u',1,size(RMSE.el1.save.avg.all,1)));
fid = fopen(['data/RMSE_el/RMSE_el1_avg_agents_' num2str(agents) '_obs_' num2str(n) '_repl_' num2str(trials) '.dat'],'wt');
fprintf(fid,fmt,RMSE.el1.save.avg.all);
fclose(fid);

fmt = sprintf('%s\n',repmat( ' %u',1,size(RMSE.el2.save.avg.all,1)));
fid = fopen(['data/RMSE_el/RMSE_el2_avg_agents_' num2str(agents) '_obs_' num2str(n) '_repl_' num2str(trials) '.dat'],'wt');
fprintf(fid,fmt,RMSE.el2.save.avg.all);
fclose(fid);

fmt = sprintf('%s\n',repmat( ' %u',1,size(RMSE.sf.save.avg.all,1)));
fid = fopen(['data/RMSE_sf/RMSE_sf_avg_agents_' num2str(agents) '_obs_' num2str(n) '_repl_' num2str(trials) '.dat'],'wt');
fprintf(fid,fmt,RMSE.sf.save.avg.all);
fclose(fid);

fmt = sprintf('%s\n',repmat( ' %u',1,size(RMSE.sn.save.avg.all,1)));
fid = fopen(['data/RMSE_sn/RMSE_sn_avg_agents_' num2str(agents) '_obs_' num2str(n) '_repl_' num2str(trials) '.dat'],'wt');
fprintf(fid,fmt,RMSE.sn.save.avg.all);
fclose(fid);
%
% Hyperparameters
fmt = sprintf('%s\n',repmat( ' %u',1,size(el1.save.avg.all,1)));
fid = fopen(['data/RMSE_el/el1_avg_agents_' num2str(agents) '_obs_' num2str(n) '_repl_' num2str(trials) '.dat'],'wt');
fprintf(fid,fmt,el1.save.avg.all);
fclose(fid);

fmt = sprintf('%s\n',repmat( ' %u',1,size(el2.save.avg.all,1)));
fid = fopen(['data/RMSE_el/el2_avg_agents_' num2str(agents) '_obs_' num2str(n) '_repl_' num2str(trials) '.dat'],'wt');
fprintf(fid,fmt,el2.save.avg.all);
fclose(fid);

fmt = sprintf('%s\n',repmat( ' %u',1,size(sf.save.avg.all,1)));
fid = fopen(['data/RMSE_sf/sf_avg_agents_' num2str(agents) '_obs_' num2str(n) '_repl_' num2str(trials) '.dat'],'wt');
fprintf(fid,fmt,sf.save.avg.all);
fclose(fid);

fmt = sprintf('%s\n',repmat( ' %u',1,size(sn.save.avg.all,1)));
fid = fopen(['data/RMSE_sn/sn_avg_agents_' num2str(agents) '_obs_' num2str(n) '_repl_' num2str(trials) '.dat'],'wt');
fprintf(fid,fmt,sn.save.avg.all);
fclose(fid);