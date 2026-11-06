# Pick and Place Simulation

This project focuses on developing a pick-and-place robot simulation environment using **MuJoCo**. The objective is to develop robot policies that can generalize their learned skills using supervised and unsupervised learning approaches.

The project follows three main stages:

1. **Data Collection:** Collect robot demonstration data using `run_record.py`.
2. **Policy Fine-tuning:** Train and fine-tune the Action Chunking with Transformers (ACT) policy using `run_train_eval.py`.
3. **Inference and Evaluation:** Evaluate the trained ACT policy in the MuJoCo simulation environment to perform pick-and-place tasks.

The project aims to understand and implement the complete pipeline, from demonstration data collection to policy training and evaluation.
