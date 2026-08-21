Possibili miglioramenti rispetto al paper originale:

provare a vedere se si può includere il tempo

ChinaPay dataset

1. loss pesata per cercare di riconoscere i laundering
2. explainability con il riconoscimento del pattern di laundering con HI-Large_Patterns.txt
-> capisco come applicarlo

3. Explainable AI (XAI) for Regulatory Compliance

Mechanism: Financial regulators mandate that flagged transactions have clear audit trails and rationale.

SOTA Implementations: Frameworks integrate post-hoc interpretability models like SHAP (SHapley Additive exPlanations) or GNNExplainer directly into deep graph architectures to identify subgraphs, feature importance, and behavioral deviations driving the alert.

Exploring Explainable AI in the Financial Sector: Perspectives of Banks and Supervisory Authorities

4. Differential privacy allows multiple banking institutions to train a unified AML model collaboratively without disclosing customer PII or raw transaction data

3. Graph Transformers, Graph Isomorphism Networks
-> studio e implemento al posto di GAT

- provo differenti tipi di GNN come Local Message Passing Neural Network nei GT

- scrivo in esperimenti le caratteristiche di tutti i modelli (funzione attivazione, head, ...)

pyarrow invece di numpy per salvare il dataset


add rule based to the algorithm prediction
TODO:


PEr ora TIME = WEEK + DAY OF THE WEEK

Scaling methods