# Gesture-Recognition-System

Given a dictionary containing 10000 words, here I'm using SHARK2 algorithm to decode a user input gesture and output the best decoded word

Sampling : SHARK2 actually is doing camparations between user input pattern with standard templates of each word.

Pruning : Compute start-to-start and end-to-end distances between a template and the unknown gesture.
Normalization is achieved by scaling the largest side of the bounding box to a parameter L.
s = L/max(W,H)

