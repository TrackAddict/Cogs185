%clear;clc

%% Task 6

data = load('sp500.mat');

startProb = data.prior;

emissionProb = data.emission;

transitionProb = data.transition;

sequences = data.price_change;

train_set = sequences(1:100);

test_set = sequences(101:128);

%% Task 7

TRANS_HAT = [0 startProb'; zeros(size(transitionProb,1),1) transitionProb];

EMIS_HAT = [zeros(1,size(emissionProb,2)); emissionProb];

%% Task 8

PStates = hmmdecode(train_set, TRANS_HAT, EMIS_HAT);

Posteriors = PStates(2:4, :)

figure;

hold on;

temp = 1:100;

plot(temp, Posteriors(1, :), 'r')

plot(temp, Posteriors(2, :), 'b')

plot(temp, Posteriors(3, :), 'g')

%hold off;

%% Task 9

viterbiStates = hmmviterbi(train_set, TRANS_HAT, EMIS_HAT);

%figure;

plot(temp, viterbiStates.*0.25, 'p')

