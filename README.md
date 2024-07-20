# Causal Relation Networks (CausalRNs)

PyTorch implementation of "An All-MLP Sequence Modeling Architecture That Excels at Copying"

Paper accepted at the ICML 2024 Workshop: [Next Generation of Sequence Modeling Architectures](https://sites.google.com/view/ngsmworkshop).

## Abstract

Recent work demonstrated Transformers’ ability to efficiently copy strings of exponential sizes, distinguishing them from other architectures. We present the Causal Relation Network (CausalRN), an all-MLP sequence modeling architecture that can match Transformers on the copying task. Extending Relation Networks (RNs), we implemented key innovations to support autoregressive sequence modeling while maintaining computational feasibility. We discovered that exponentiallyactivated RNs are reducible to linear time complexity, and pre-activation normalization induces an infinitely growing memory pool, similar to a KV cache. In ablation study, we found both exponential activation and pre-activation normalization are indispensable for Transformer-level copying. Our findings provide new insights into what actually constitutes strong in-context retrieval. ![arXiv](https://img.shields.io/badge/arXiv-2406.16168-red)


## Citation

```
@misc{cui2024allmlpsequencemodelingarchitecture,
      title={An All-MLP Sequence Modeling Architecture That Excels at Copying}, 
      author={Chenwei Cui and Zehao Yan and Gedeon Muhawenayo and Hannah Kerner},
      year={2024},
      eprint={2406.16168},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.16168}, 
}
```

