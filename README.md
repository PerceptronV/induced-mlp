# How do we induce a layered MLP from a _complete network_?

If we start from an MLP where every neuron is connected to every other, train it on some task, and then prune the network aggressively, will we see the emergence of information processing organised into layers? If so, how and why does this process occur? What inductive biases in weight initialisation, training, and pruning are needed to induce such a phenomenon? What are some theoretical and practical insights we can derive from this?
