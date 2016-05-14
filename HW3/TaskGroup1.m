clear; clc

%% Task 2

transitionProb = [0.85, 0.15; 0.10, 0.90];

emissionProb = [1/3, 1/4, 5/12; 1/4, 1/4, 1/2];

for i = 1:50
   [seqs_train{i}, states_train{i}] =  hmmgenerate(10, transitionProb, emissionProb);
end

A = cell2mat(seqs_train);

sequences = reshape(A, [50, 10]);

B = cell2mat(states_train);

states = reshape(B, [50, 10]);


csvwrite('TG1sequences.txt', sequences)

csvwrite('TG1states.txt', states)

%% Task 3

[estimatedTrans, estimatedEmission] = hmmestimate(seqs_train{2}, states_train{5});

csvwrite('TG1estTrans.txt', estimatedTrans)

csvwrite('TG1estEm.txt', estimatedEmission)
%% Task 4

viterbiStates = hmmviterbi(seqs_train{2}, transitionProb, emissionProb);

csvwrite('TG1Viterbi.txt', viterbiStates)
%% Task 5

Pstates = hmmdecode(seqs_train{2}, transitionProb, emissionProb);

csvwrite('Pstates.txt', viterbiStates)

