function   F = communicability_wei(CIJ)

%inputs
%           CIJ    weighted connection matrix
%           i      row
%           j      column
%
%outputs
%           F      communicability
%
% Author: Bratislav Mišić
%=================================================

N = size(CIJ,1);

B = sum(CIJ')';
C = diag(B);
D = C^(-(1/2));
E = D * CIJ * D;
F = expm(E);
F = F.*~eye(N);
