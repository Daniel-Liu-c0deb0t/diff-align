# diff-align
Differentiable position-specific probability matrix alignment.

This is a naive reimplementation of the differentiable edit distance dynamic programming
algorithm from the DeepConsensus [paper](https://www.biorxiv.org/content/10.1101/2021.08.31.458403v1).
It compares a correct target sequence with a predicted sequence that has nucleotide
probabilities at every position.
This is used as part of the loss function for training a transformer to predict the correct
sequences.

The main idea compared to classic edit distance is to keep track of DP computation in a computational
graph with Tensorflow, and replace the `min` operation with a differentiable version (log-sum-exp).
To handle insertion errors, the predicted sequence is allowed to have `$`
characters, which symbolizes a character that can be skipped. To handle deletion errors,
we use a fixed penalty.

To demonstrate that this works, I created a simple example program to optimize a given initial sequence
so that it becomes a certain target sequence. Optimization is done using gradient descent, and this
is able to handle substitutions, insertions, and deletions between the two sequences by
using differentiable DP.

If you run `python3 main.py`, you should get something like
```
Target: ATATCGG
Initial: AAATCAGG
Loss: 7.110587
AAATCAGG
Loss: 6.450298
AAATCAGG
Loss: 5.8413467
AAATCAGG
Loss: 5.2877736
AAATCAGG
Loss: 4.788586
AAATCAGG
Loss: 4.3385553
ATATCAGG
Loss: 3.931414
ATATC$GG
Loss: 3.5620618
ATATC$GG
Loss: 3.226541
ATATC$GG
Loss: 2.9216578
ATATC$GG
[[0.8223933  0.04488795 0.04488805 0.0436576  0.04417309]
 [0.16281272 0.06111485 0.06112102 0.43531853 0.27963284]
 [0.74050695 0.03928063 0.03934939 0.03710027 0.14376274]
 [0.03961079 0.04023565 0.0407337  0.76307327 0.11634663]
 [0.03598844 0.6739386  0.03602815 0.03442852 0.21961632]
 [0.13547596 0.1467242  0.32167402 0.04927655 0.34684923]
 [0.0367826  0.03635236 0.6487619  0.03672411 0.24137901]
 [0.03688385 0.03686266 0.6476075  0.03688243 0.24176349]]
Final: ATATC$GG
```
