%%Riordino la matrice secondo Walsh

function [WH]=orderW(H,N)


HadIdx = 0:N-1;                          % Hadamard index
b = log2(N)+1;                           % Number of bits to represent the index


binHadIdx = fliplr(dec2bin(HadIdx,b))-'0'; % Bit reversing of the binary index
binSeqIdx = zeros(N,b-1);                  % Pre-allocate memory
for k = b:-1:2
    % Binary sequency index
    binSeqIdx(:,k) = xor(binHadIdx(:,k),binHadIdx(:,k-1));
end
SeqIdx = binSeqIdx*pow2((b-1:-1:0)');    % Binary to integer sequency index
WH = H(SeqIdx+1,:); % 1-based indexing