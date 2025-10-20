# AgentAsk: Multi-Agent Systems Need to Ask

##  Why AgentAsk?
AgentAsk brings **edge-level clarification** to multi-agent systems (MAS). It treats each inter-agent message as a potential failure point and inserts a **minimal necessary** question to prevent error cascades, improving reliability with low latency and cost.

![intro](assets/intro.pdf)

##  Method Overview
AgentAsk is a lightweight, plug-and-play clarifier that:
* Detects edge-level issues which are based on our taxonomy: **Data Gap (DG)**, **Signal Corruption (SC)**, **Referential Drift (RD)**, **Capability Gap (CG)**.


![taxonomy](assets/taxonomy.png)
* Decides **when/what/whom/how** to ask using a factored policy.
* Trains in two stages: **Supervised Fine-Tuning (SFT)** from failure traces and **E-GRPO** reinforcement for accuracy–latency–cost trade-offs.
![pipeline](assets/pipline.pdf)

## 📚 Paper[]
[AgentAsk: Multi-Agent Systems Need to Ask](https://arxiv.org/abs/2305.09062)



##  🧭 Quick Start

### 🔑 Add API keys
Add API keys in `template.env` and change its name to `.env`.

###  Datasets
Please download `GSM8K`, `HumanEval`, `MATH`, `MBPP`, `MMLU` and place them in the `Datasets` folder

### run
To run the code of phase 1, run the following command:

```bash
bash scripts/run_sft.sh
```

To run the code of phase 2, run the following command:

```bash
python Experiments\run_xxx.py
```

## 📝 Citation
If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{zhang2023agentask,
  title={AgentAsk: Multi-Agent Systems Need to Ask},
  author={Zhang, Zhiyuan and Li, Yifan and Wang, Zhenyu and Wang, Yuxuan and Liu, Yuxuan and Chen, Zhiyuan and Li, Jie},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
  pages={6317--6328},
  year={2023}
}
```

