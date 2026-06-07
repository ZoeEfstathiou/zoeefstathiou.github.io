---
title: "Attention Mechanisms in Deep Learning: A Visual Guide"
date: 2025-01-15
categories:
  - deep-learning
tags:
  - attention
  - transformers
  - nlp
excerpt: "A bottom-up walkthrough of attention — from the original seq2seq mechanism to multi-head self-attention in Transformers — with intuitive explanations and the math behind them."
---

Attention mechanisms are one of the most important ideas in modern deep learning. They let models focus on the parts of an input that are most relevant to a given task, rather than treating all inputs equally. In this post I'll build up the concept from first principles.

## Why Attention?

Before attention, sequence-to-sequence models (like those used for machine translation) compressed an entire input sentence into a single fixed-size vector — the final hidden state of an RNN. The decoder then had to generate the entire output from just that one vector.

This is clearly a bottleneck. A 50-word input sentence crammed into a 512-dimensional vector loses a lot of information, especially for long inputs.

**The insight behind attention:** instead of forcing the encoder to compress everything, let the decoder *look back* at all encoder hidden states and pick what's relevant at each step.

## Bahdanau Attention (2015)

The first widely-used attention mechanism was introduced by [Bahdanau et al. (2015)](https://arxiv.org/abs/1409.0473). The key idea:

At each decoding step $t$, compute a *context vector* $c_t$ as a weighted sum of all encoder hidden states $h_1, \ldots, h_T$:

$$c_t = \sum_{i=1}^{T} \alpha_{ti} h_i$$

where the attention weights $\alpha_{ti}$ sum to 1 and are computed as:

$$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{k=1}^{T} \exp(e_{tk})}$$

The *energy scores* $e_{ti}$ measure how well the decoder state $s_{t-1}$ matches each encoder state $h_i$:

$$e_{ti} = a(s_{t-1}, h_i)$$

Here $a$ is a small learned neural network (an alignment model).

## Scaled Dot-Product Attention

The Transformer paper ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) introduced a simpler, more parallelizable form of attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- **Q** (queries): what we're looking for
- **K** (keys): what each position offers
- **V** (values): the actual content at each position
- $d_k$: dimension of keys (used for scaling to prevent vanishing gradients)

The scaling by $\sqrt{d_k}$ is important — without it, dot products grow large in high dimensions, pushing softmax into regions with tiny gradients.

## Multi-Head Attention

Rather than performing a single attention computation, Transformers run $h$ attention operations in parallel ("heads"), each with different learned projections:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V)$$

Each head can learn to attend to different kinds of information — one head might capture syntactic relationships, another semantic similarity, another coreference.

## Self-Attention

In the Transformer encoder, Q, K, and V all come from the *same* sequence. This allows every position to attend to every other position in the same layer — capturing long-range dependencies without the sequential bottleneck of RNNs.

This is why Transformers train so much faster than LSTMs: all positions are processed in parallel, and gradients flow directly between any two positions.

## Key Takeaways

- Attention lets models dynamically weight which parts of the input to focus on
- Scaled dot-product attention is efficient and parallelizable
- Multi-head attention lets the model jointly attend to information from different subspaces
- Self-attention is the foundation of all modern large language models

In a future post, I'll cover positional encodings and how Transformers handle sequence order without recurrence.

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Weng, L. (2018). [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
