As of April 2024, “multimodal AI models“ almost surely refers to a natural language modality plus typically either speech or image recognition. 

This repository addresses the need to develop more open source and truly multimodal AI models, taking input data consisting of time sequences of jointly distributed audio and human motion capture data. These latent representations can then be used for downstream data processing tasks in interactive media and information retrieval systems where both audio and human body motion inputs are needed.

The repository assembles state-of-art machine learning components for embedding audio and human motion capture mesh data, together with the RWKV language/sequence modelling architecture, with the goal to obtain latent representations of the input data sequence which maximises mutual information between the motion capture and audio input signals. 

For this goal, a mutual information neural estimator (MINE) is trained jointly with the latent representations, and the result is embeddings of high mutual information between inputs of audio and human body motion.
