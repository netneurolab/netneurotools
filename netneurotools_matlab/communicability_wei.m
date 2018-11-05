function F = communicability_wei(CIJ)
%
% COMMUNICABILITY_WEI(A) computes the communicability of pairs of nodes in the
% network represented by the weighted adjacency matrix A. It returns a matrix
% whose elements G(i,j) = G(j,i) give the the communicability between nodes i
% and j.
%
% Author: Bratislav Mišić
%

N = size(CIJ,1);

B = sum(CIJ')';
C = diag(B);
D = C^(-(1/2));
E = D * CIJ * D;
F = expm(E);
F = F.*~eye(N);
