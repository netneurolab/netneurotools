function G = communicability(A)
%
% COMMUNICABILITY(A) computes the communicability of pairs of nodes in the
% network represented by the unweighted adjacency matrix A. It returns a matrix
% whose elements G(i,j) = G(j,i) give the the communicability between nodes i
% and j.
%
% Author: Bratislav Mišić
%

G = expm(A);    % Compute the matrix exponential of A.
