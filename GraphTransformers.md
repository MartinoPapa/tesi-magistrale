# Architettura dei Graph Transformers

I classici Graph Neural Networks (GNN), come GCN, GIN o GAT, basano il loro funzionamento sul paradigma del *Message Passing* locale. Sebbene estremamente efficaci, presentano alcune limitazioni intrinseche, in particolar modo quando è necessario costruire reti profonde per propagare informazioni a lunghe distanze all'interno della rete:
- **Oversquashing**: difficoltà nel comprimere informazioni provenienti da nodi distanti nel grafo all'interno di vettori (embedding) a dimensionalità fissa, specialmente quando il campo recettivo cresce esponenzialmente (problema dei "bottleneck" strutturali).
- **Oversmoothing**: applicando molti layer di message passing, le rappresentazioni dei nodi tendono a uniformarsi e diventare indistinguibili.
- **Potere Espressivo limitato**: i classici MPNN (Message Passing Neural Networks) sono limitati superiormente nel loro potere espressivo dal test di isomorfismo di Weisfeiler-Lehman (1-WL test), rendendoli incapaci di distinguere certe strutture sub-grafiche regolari, come i cicli di diversa lunghezza. Inoltre il message passing locale non permette di catturare pattern di profondità maggiore in grafi estesi in modo efficiente.

Per superare questi limiti, l'architettura dei **Graph Transformers** adatta il meccanismo di *Self-Attention*, tipico dei modelli del Natural Language Processing (NLP), ai grafi. Questo permette di modellare dinamicamente l'importanza delle connessioni e catturare dipendenze a lungo raggio.

Di seguito analizziamo nel dettaglio l'architettura di riferimento utilizzata per modellare il comportamento dei Graph Transformers: **GPSConv**.

---

## GPS (General, Powerful, Scalable Graph Transformer) e GPSConv

L'architettura **GPS (Recipe for a General, Powerful, Scalable Graph Transformer)**, di cui il modulo `GPSConv` costituisce il blocco portante, propone un profondo cambio di paradigma rispetto alle reti GNN standard o ai normali Transformer. Invece di usare l'attenzione unicamente come meccanismo di aggregazione sui vicini locali, o viceversa trattare il grafo come un insieme completamente connesso ignorando la sparsità (e sprecando computazione), GPS **disaccoppia del tutto l'elaborazione della struttura locale da quella dell'attenzione globale**.

Ogni layer `GPSConv` calcola in parallelo l'output di un *Local Message Passing Neural Network* (MPNN) e un modulo di *Global Attention* (su tutti i nodi). Questo approccio garantisce simultaneamente il rigoroso potere espressivo (strutturale) garantito dal test 1-WL dei MPNN locali, combinandolo con la capacità del Transformer globale di catturare relazioni a lungo raggio in un solo passo, risolvendo alla radice il problema dell'oversquashing.

### Formulazione Matematica

Un singolo strato $l$ del `GPSConv` riceve in input l'insieme degli embedding dei nodi $H^{(l)} = \{h_1^{(l)}, \dots, h_N^{(l)}\}$. L'aggiornamento che produce $H^{(l+1)}$ si compone di tre fasi principali:

**1. Local Message Passing (MPNN)**:
Si applica un operatore GNN standard per estrarre il bias strutturale basato sulla reale topologia sparsa $\mathcal{E}$ del grafo. Nelle implementazioni più performanti di GPS, questa componente è rappresentata da layer ad alto potere espressivo come GIN (*Graph Isomorphism Network*) o GatedGCN.
$$ h_{i}^{MPNN} = \text{MPNN} \Big( h_i^{(l)}, \{ h_j^{(l)} \}_{j \in \mathcal{N}(i)} \Big) $$
Se ad esempio viene utilizzato GIN come operatore locale, l'aggiornamento assume la forma:
$$ h_{i}^{MPNN} = \text{MLP} \Big( (1 + \epsilon) h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \text{ReLU}(W \, h_j^{(l)}) \Big) $$
Questa componente assicura che il modello sia in grado di distinguere accuratamente la morfologia locale del grafo di base (gradi dei nodi, percorsi, isomorfismi limitrofi).

**2. Global Attention (Transformer Globale)**:
In parallelo all'operatore locale, si calcola l'attenzione **globale** fully-connected, permettendo un'interazione paritetica tra tutti i nodi del grafo, indipendentemente dall'esistenza di un arco fisico che li connetta:
$$ h_{i}^{Global} = \text{MultiHeadAttention} \Big( h_i^{(l)}, \{ h_j^{(l)} \}_{j=1}^{N} \Big) $$
L'equazione esatta ricalca quella canonica del Transformer:
- **Query, Key, Value**: $q_i = W_Q \, h_i^{(l)}$, $k_j = W_K \, h_j^{(l)}$, $v_j = W_V \, h_j^{(l)}$
- **Attenzione (Scaled Dot-Product)**: $\alpha_{i,j} = \frac{\langle q_i, k_j \rangle}{\sqrt{d_k}}$
- **Aggregazione globale**: $h_i^{Global} = \sum_{j=1}^{N} \text{Softmax}_j (\alpha_{i,j}) v_j$

Per mantenere la scalabilità computazionale su grafi estesi, l'implementazione pratica del modulo globale in GPS utilizza spesso varianti ad *attenzione lineare* (come Performer o BigBird), abbattendo la complessità da $\mathcal{O}(N^2)$ a $\mathcal{O}(N)$.

**3. Fusione e Feed-Forward Network (FFN)**:
I due segnali vengono sommati (o combinati) insieme all'input originale del layer tramite connessioni residue (Residual Connection) seguite da Layer Normalization:
$$ \tilde{h}_i = \text{LayerNorm} \big( h_i^{(l)} + h_{i}^{MPNN} + h_{i}^{Global} \big) $$
Successivamente, come nell'architettura originale del Transformer di base (Vaswani et al.), l'embedding viene raffinato passando attraverso una rete completamente connessa a due strati (FFN):
$$ h_i^{(l+1)} = \text{LayerNorm} \big( \tilde{h}_i + \text{FFN}(\tilde{h}_i) \big) $$

*Nota sull'Espressività (PE/SE)*: Affinché il modulo *Global Attention* non tratti tutti i nodi indiscriminatamente come in un set non strutturato privo di posizione spaziale, all'input iniziale del modello vengono spesso pre-sommati dei **Positional / Structural Encodings (PE/SE)** (ad esempio le Random Walk probabilities o gli autovettori Laplaciani). Questo fornisce all'attenzione globale un sistema di coordinate intrinseco per comprendere la "distanza" geometrica e strutturale tra i nodi, proiettando il potere espressivo della rete strettamente oltre il limite del test 1-WL.

### Vantaggi dell'Architettura GPSConv

1. **Receptive Field Globale Immediato**: Crea una *shortcut* (scorciatoia) diretta tra ogni coppia di nodi fin dal layer 1.
2. **Mitigazione totale dell'Oversquashing**: La separazione del canale globale annulla la necessità di far fluire o comprimere le informazioni a lungo raggio ("long-range dependencies") attraverso i bottleneck morfologici del grafo.
3. **Alto Potere Espressivo**: Combina il meglio di entrambi i mondi, sfruttando la rigorosa espressione strutturale dell'MPNN combinato con i Positional Encodings e arricchendolo con l'astrazione globale dell'attenzione.

Queste caratteristiche rendono il modulo `GPSConv` ideale per task in cui la scoperta di pattern complessi (come anelli di riciclaggio o strutture multilivello di distanziamento dei fondi nelle frodi finanziarie) richiede di analizzare contemporaneamente sia la densità delle interazioni locali, sia le correlazioni sfuggenti tra cluster di account molto distanti all'interno della rete transazionale.
