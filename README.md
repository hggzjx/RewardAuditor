



<div align="center">
  <h1>Reward Auditor: Inference on Reward Modeling Suitability in Real-World Perturbed Scenarios</h1>
  <br />
  <span style="color:red">📢 <strong><i>If you are interested in our work, please star ⭐ our project.</i></strong></span>

  <h4>
    <a href="https://arxiv.org/abs/2512.00920"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:Reward Auditor&color=red&logo=arxiv"></a>
    <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">
  </h4>
</div>


## 🌈 Introduction


**Reward Auditor** is a novel hypothesis-testing framework designed to evaluate the **suitability**—conditional reliability under real-world perturbations of Reward Models. Moving beyond simple preference accuracy, it employs scientific auditing to quantify the statistical significance and effect size of distribution degradation in RM confidence. By inferring the certainty and severity of systematic vulnerabilities across diverse scenarios, Reward Auditor provides a rigorous foundation for building verifiable, robust, and trustworthy LLM alignment systems.

![](assets/framework.png)


## 😍 Evaluation

Our codebase is largely based on the Reward Bench and RM Bench Thus for the environment setup, you may follow the instructions in the Reward Bench Setup. After git clone the repository, you can run the following command to evaluate the reward model on RM-Bench with Reward Auditor:

```bash
bash run_rm_auditing.sh # evaluate the suitability of the reward models
bash run_dpo_auditing.sh # evaluate the suitability of the DPO based model
```



## 📝License
Distributed under the Apache-2.0 License. See LICENSE for more information.




## 📖Citation

if you find this work helpful, please cite it as:

```
@article{zang2025reward,
  title={Reward Auditor: Inference on Reward Modeling Suitability in Real-World Perturbed Scenarios},
  author={Zang, Jianxiang and Wei, Yongda and Bai, Ruxue and Jiang, Shiyu and Mo, Nijia and Li, Binhong and Sun, Qiang and Liu, Hui},
  journal={arXiv preprint arXiv:2512.00920},
  year={2025}
}
```

