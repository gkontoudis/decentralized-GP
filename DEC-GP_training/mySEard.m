function K = mySEard(hyp, x, z, i)

% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x,z) = sf^2 * exp(-(x-z)'*inv(P)*(x-z)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         log(sf) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch

if nargin<2, K = '(D+1)'; return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

D = size(x, 2);
ell = exp(hyp(1:D));
sf2 = exp(2*hyp(D+1));
sf = exp(hyp(D+1));

m = @(d2) exp(-d2/2);

% precompute distances
if dg                                                               % vector kxx
    K = zeros(size(x,1),1);
else
    if xeqz                                                 % symmetric matrix Kxx
        K = sq_dist(diag(1./ell)*x');
    else                                                   % cross covariances Kxz
        K = sq_dist(diag(1./ell)*x',diag(1./ell)*z');
    end
end

if nargin<4                                                        % covariances
    K = sf2*m(K);
else                                                               % derivatives
    if i<=D                                               % length scale parameter
        if dg
            Ki = zeros(size(x,1),1);
        else
            if xeqz
                Ki = sq_dist(1/ell(i)*x(:,i)');
            else
                Ki = sq_dist(1/ell(i)*x(:,i)',1/ell(i)*z(:,i)');
            end
        end
        K = sf2*m(K).*Ki;
        %K(Ki<1e-12) = 0;                   % fix limit case(It's a direct copy, I don't understand this line)
    elseif i==D+1                                            % magnitude parameter
        K = 2*sf*m(K); % change that from sf2 to sf
    else
        error('Unknown hyperparameter')
    end
end

end