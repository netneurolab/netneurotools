function wz = zrand(X,Y)
%
% ZRAND(X,Y) calculates the z-Rand score of two vectors X and Y, where X and Y
% are two potential clustering assignments for a group of samples.
%
% Author: Bratislav Mišić
%

n = length(X);

indx = dummyvar(X);
indy = dummyvar(Y);

Xa = indx*indx';
Ya = indy*indy';

M = n*(n - 1)/2;
M1 = nnz(Xa)/2;
M2 = nnz(Ya)/2;

wab = nnz(Xa & Ya)/2;
muab = M1*M2/M;

nx = sum(indx);
ny = sum(indy);

C1 = n*(n^2 - 3*n - 2) - 8*(n + 1)*M1 + 4*sum(nx.^3);
C2 = n*(n^2 - 3*n - 2) - 8*(n + 1)*M2 + 4*sum(ny.^3);

a = M/16;
b = ((4*M1 - 2*M).^2).*((4*M2 - 2*M).^2)/(256*(M^2));
c = C1*C2/(16*n*(n - 1)*(n - 2));
d = (((4*M1 - 2*M)^2) - 4*C1 - 4*M)*(((4*M2 - 2*M)^2) - 4*C2 - 4*M)/(64*n*(n - 1)*(n - 2)*(n - 3));

sigw2 = a - b + c + d;
sigw = sqrt(sigw2);

wz = (wab - muab)/sigw;
