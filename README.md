# Poor Man's Quality Estimation

Code for the paper [Poor Man's Quality Estimation: Predicting Reference-Based MT Metrics Without the Reference](https://arxiv.org/abs/2301.09008) by Vilém Zouhar, Shehzaad Dhuliawala, Wangchunshu Zhou, Nico Daheim, Tom Kocmi, Yuchen Eleanor Jiang and Mrinmaya Sachan from ETH Zürich.

> Machine translation quality estimation (QE) predicts human judgements of a translation hypothesis without seeing the reference. State-of-the-art QE systems based on pretrained language models have been achieving remarkable correlations with human judgements yet they are computationally heavy and require human annotations, which are slow and expensive to create. To address these limitations, we define the problem of metric estimation (ME) where one predicts the automated metric scores also without the reference. We show that even without access to the reference, our model can estimate automated metrics (ρ=60% for BLEU, ρ=51% for other metrics) at the sentence-level. Because automated metrics correlate with human judgements, we can leverage the ME task for pre-training a QE model. For the QE task, we find that pre-training on TER is better (ρ=23%) than training for scratch (ρ=20%).

The paper will be presented at EACL 2023. For now, cite as:

```
@article{zouhar2023poor,
  doi = {10.48550/ARXIV.2301.09008},
  url = {https://arxiv.org/abs/2301.09008},
  author = {Zouhar, Vilém and Dhuliawala, Shehzaad and Zhou, Wangchunshu and Daheim, Nico and Kocmi, Tom and Jiang, Yuchen Eleanor and Sachan, Mrinmaya},
  title = {Poor Man's Quality Estimation: {P}redicting Reference-Based {MT} Metrics Without the Reference},
  publisher = {arXiv},
  year = {2023},
}
```
