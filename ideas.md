## Reviewing papers:
- Graph Neural Networks for Financial Fraud Detection: A Review

## Dataset choice notes:
Real-world datasets are seldom available because of privacy concerns. For academical purposes the usage of synthetic datasets is highly recomended. The best options for a synthetic dataset gemerators for AML are:
- **IBM AMLSim** (https://github.com/IBM/AMLSim/blob/master/README.md): widely used multi-agent simulator that injects classic AML typologies into background transaction noise
- **AMLGentex** (https://arxiv.org/pdf/2506.13989): published in 2025 represents the state-of-the-art for synthetic AML datasets generation.

The most known one in this field used to be PaySim but this is deprecated since it didn't hold the topological properties of real world financial graphs.

Some of the suggested papers are tested only on real private datasets. We will be able to replicate these experiments by using synthetic ones but we won't have a fair comparison between the results.

## Graph Classification
### Task: Subgraph / Community Labeling (supernode labeling)
- **Anti-Money Laundering by Group-Aware Deep Graph Learning (2023)**
    - Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10114503
    - Notes: 
        - It focuses on identifying money laundering organizations (rings etc) instead of individual fraudolent nodes.
        - It's suitable for AMLSim and AMLGentex datasets
        - I found a thesis from 2025 that is based on it: **Gang Prediction in Graphs for Anti Money Laundering Detection (2025)**
            - Link: https://webthesis.biblio.polito.it/38622/1/tesi.pdf
            - Datasets:
                - IBM AMLSim https://github.com/IBM/AMLSim/blob/master/README.md
                - Cora
                - AMLGentex
            - Code: https://github.com/whiitex/Gang-Prediction
    - Datasets:
        - UnionPay (Real World and private)
## Edge Classification
### Task: E-commerce Fraud Detection
- **An E-commerce Fraud Detection System via Competitive Graph Neural Networks (2022)**
    - Link: https://dl.acm.org/doi/10.1145/3474379
    - Code: https://github.com/GeZhangMQ/eFraudCom
    - Datasets:
        - Bitcoin-Alpha
        - MOOC student drop-out
## Node Classification
### Task: Fraud detection
- **Money Laundering Detection Using Graph Neural Networks Enhanced with Autoencoder Components (2025)**
    - Link: https://www.researchgate.net/publication/398340548_Money_Laundering_Detection_Using_Graph_Neural_Networks_Enhanced_with_Autoencoder_Components
    - Datasets:
        - IBM AMLSim https://github.com/IBM/AMLSim/blob/master/README.md
- HybridFL: A Federated Learning Approach for Financial Crime Detection (2026)
    - Link: https://arxiv.org/pdf/2602.19207
    - Datasets:
        - IBM AMLSim https://github.com/IBM/AMLSim/blob/master/README.md
    - Tex: https://arxiv.org/src/2602.19207
- **Finding Money Launderers Using Heterogeneous Graph Neural Networks (2023)**
    - Link: https://arxiv.org/pdf/2307.13499
    - Code: https://github.com/fredjo89/heterogeneous-mpnn
    - Datasets:
        - Private: DNB (Norway's largest bank)
        - Suitable alternatives:
            - AMLGentex https://arxiv.org/pdf/2506.13989 
            - IBM AMLSim https://github.com/IBM/AMLSim/blob/master/README.md
    - Tex: https://arxiv.org/abs/2307.13499