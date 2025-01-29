function [f] = goldpr(xx,xy)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% GOLDSTEIN-PRICE FUNCTION
%
% Authors: Sonja Surjanovic, Simon Fraser University
%          Derek Bingham, Simon Fraser University
% Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
%
% Copyright 2013. Derek Bingham, Simon Fraser University.
%
% THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
% FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
% derivative works, such modified software should be clearly marked.
% Additionally, this program is free software; you can redistribute it 
% and/or modify it under the terms of the GNU General Public License as 
% published by the Free Software Foundation; version 2.0 of the License. 
% Accordingly, this program is distributed in the hope that it will be 
% useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
% of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
% General Public License for more details.
%
% For function details and reference information, see:
% http://www.sfu.ca/~ssurjano/
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INPUT:
%
% xx = [x1, x2]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x1 = xx;
x2 = xy;
x1bar = 4*x1 - 2;
x2bar = 4*x2 - 2;
fact1a = (4*x1 + 4*x2 -3).^2;
fact1b = 75 - 56*(x1 +x2) +3*x1bar.^2 +6*x1bar.*x2bar + 3*x2bar.^2;
fact1 = 1 + fact1a.*fact1b;

fact2a = (8*x1 - 12*x2 +2).^2;
fact2b = -14 -128*x1 + 12*x1bar.^2 + 192*x2 - 36*x1bar.*x2bar + 27*x2bar.^2;
fact2 = 30 + fact2a.*fact2b;

prod = fact1.*fact2;

f = (log(prod) - 8.693) / 2.427;

end