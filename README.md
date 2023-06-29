# AGRA
### Adaptive GRAdient-based outlier removal


This repository contains code used in our paper: </br>
**["Learning with Noisy Labels by Adaptive Gradient-Based Outlier Removal"](https://arxiv.org/pdf/2306.04502.pdf)**
to be presented at ECML PKDD'23 🚀 </br>
by Anastasiia Sedova, Lena Zellinger, and Benjamin Roth. 

For any questions please [get in touch](mailto:anastasiia.sedova@univie.ac.at).

--- 

### Gradient-based outlier removal

We propose a new approach to the problem of learning with noisy labels. 
- Instead of tracing the mislabeled samples in the dataset, we focus on obtaining a model that remains unaffected by inconsistent and noisy samples.
- Instead of removing the noisy samples once and forever, we decide for each sample whether it is _useful_ or not for a model at the current training stage.
- Instead of denoising the data _before_ classifier training, we join the denoising and training into a single process 
where denoising happens _during_ the model training. 

These ideas are realized in our new method **AGRA** for Adaptive GRAdient-based outlier removal. 

--- 

### AGRA

AGRA does not rely on some static, model-independent, implied properties but leverages gradients to measure the 
sample-specific impact on the current model.
During classifier training, AGRA decides for each sample whether it is useful or not for a model at the current training
stage by comparing its gradient with an accumulated gradient of another batch that is independently sampled from the 
same dataset. Depending on the state of the classifier and the experimental setup, the sample is either used in the 
model update, excluded from the update, or assigned to an alternative label.


<p align="center">
  <img src="agra.png" alt="AGRA" width="100%" height="100%">
</p>

---

### Usage 

---

### Citation

When using our work please cite our ArXiV preprint:  

```
@misc{sedova2023learning,
      title={Learning with Noisy Labels by Adaptive Gradient-Based Outlier Removal}, 
      author={Anastasiia Sedova and Lena Zellinger and Benjamin Roth},
      year={2023},
      eprint={2306.04502},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Acknowledgements 💎

This research has been funded by the Vienna Science and Technology Fund (WWTF)[10.47379/VRG19008] and by the Deutsche 
Forschungsgemeinschaft (DFG, German Research Foundation) RO 5127/2-1.
