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
In parallelo all'operatore locale, si calcola l'attenzione **globale** fully-connected, permettendo un'interazione paritetica tra tutti i nodi del grafo. Questo meccanismo impiega comunemente la **Multi-Head Attention (MHA)**, che permette di prestare attenzione a diversi sottospazi rappresentazionali.
Per una singola testa d'attenzione (head $h$), si calcolano **Query, Key, Value** per ogni nodo:
$$ q_{i,h} = W_{Q,h} \, h_i^{(l)}, \quad k_{j,h} = W_{K,h} \, h_j^{(l)}, \quad v_{j,h} = W_{V,h} \, h_j^{(l)} $$
L'attenzione globale (Scaled Dot-Product) per la testa $h$ calcola i pesi tra il nodo $i$ e tutti i nodi $j \in \{1, \dots, N\}$:
$$ \alpha_{i,j,h} = \text{Softmax}_j \Bigg( \frac{q_{i,h} \cdot k_{j,h}^T}{\sqrt{d_k}} \Bigg) $$
dove $d_k$ è la dimensione della singola testa. L'output della testa $h$ per il nodo $i$ aggrega le informazioni pesate:
$$ \text{head}_{i,h} = \sum_{j=1}^{N} \alpha_{i,j,h} \, v_{j,h} $$
I risultati di tutte le $H$ teste vengono poi concatenati e proiettati tramite una matrice di pesi finale $W_O$:
$$ h_i^{Global} = \text{Concat} \big( \text{head}_{i,1}, \dots, \text{head}_{i,H} \big) W_O $$

Per mantenere la scalabilità computazionale su grafi estesi, l'implementazione pratica del modulo globale in GPS utilizza spesso varianti ad *attenzione lineare* (come **Performer** o BigBird), abbattendo la complessità da $\mathcal{O}(N^2)$ a $\mathcal{O}(N)$. Performer, ad esempio, sfrutta una mappa non lineare di kernel (FAVOR+) per approssimare la matrice di softmax, evitando di materializzare esplicitamente la matrice di attenzione globale.

**3. Fusione e Feed-Forward Network (FFN)**:
I due segnali vengono sommati (o combinati) insieme all'input originale del layer tramite connessioni residue (Residual Connection) seguite da Layer Normalization:
$$ \tilde{h}_i = \text{LayerNorm} \big( h_i^{(l)} + h_{i}^{MPNN} + h_{i}^{Global} \big) $$
Successivamente, come nell'architettura originale del Transformer di base (Vaswani et al.), l'embedding viene raffinato passando attraverso una rete completamente connessa a due strati (FFN):
$$ h_i^{(l+1)} = \text{LayerNorm} \big( \tilde{h}_i + \text{FFN}(\tilde{h}_i) \big) $$

*Nota sull'Espressività (PE/SE)*: Affinché il modulo *Global Attention* non tratti tutti i nodi indiscriminatamente come in un set non strutturato privo di posizione spaziale, all'input iniziale del modello (prima del primo layer `GPSConv`) vengono tipicamente pre-sommati o concatenati dei **Positional / Structural Encodings (PE/SE)**. Questo fornisce all'attenzione globale un sistema di coordinate intrinseco per comprendere la topologia del grafo. Un esempio comune è il **Random Walk Structural Encoding (RWSE)**, che per ogni nodo $i$ estrae le probabilità di ritorno in percorsi di lunghezza $1, 2, \dots, k$:
$$ p_i = \big[ P_{i,i}^1, P_{i,i}^2, \dots, P_{i,i}^k \big] $$
dove $P = A D^{-1}$ è la matrice di transizione del random walk. Il vettore $p_i$ viene processato da una rete lineare per produrre un embedding che si somma a $h_i^{(0)}$.
Un'altra variante popolare usa gli autovettori della **Matrice Laplaciana Normalizzata (LapPE)**:
$$ L = I - D^{-1/2} A D^{-1/2} = U \Lambda U^T $$
dove le righe della matrice degli autovettori $U$ fungono da coordinate spettrali per i nodi, preservando la geometria globale del grafo. Questo proietta il potere espressivo della rete strettamente oltre il limite del test 1-WL.

### Vantaggi dell'Architettura GPSConv

1. **Receptive Field Globale Immediato**: Crea una *shortcut* (scorciatoia) diretta tra ogni coppia di nodi fin dal layer 1.
2. **Mitigazione totale dell'Oversquashing**: La separazione del canale globale annulla la necessità di far fluire o comprimere le informazioni a lungo raggio ("long-range dependencies") attraverso i bottleneck morfologici del grafo.
3. **Alto Potere Espressivo**: Combina il meglio di entrambi i mondi, sfruttando la rigorosa espressione strutturale dell'MPNN combinato con i Positional Encodings e arricchendolo con l'astrazione globale dell'attenzione.

Queste caratteristiche rendono il modulo `GPSConv` ideale per task in cui la scoperta di pattern complessi (come anelli di riciclaggio o strutture multilivello di distanziamento dei fondi nelle frodi finanziarie) richiede di analizzare contemporaneamente sia la densità delle interazioni locali, sia le correlazioni sfuggenti tra cluster di account molto distanti all'interno della rete transazionale.

---

## Iperparametri del `GPSConv` in PyTorch Geometric (PyG)

L'implementazione standard del layer `GPSConv` nella libreria PyTorch Geometric prevede i seguenti iperparametri principali per la configurazione del modulo:

| Iperparametro | Tipo | Default | Descrizione |
| :--- | :--- | :--- | :--- |
| **`channels`** | `int` | - | Dimensione delle feature in input e in output di ogni nodo (dimensione dell'embedding $d$). |
| **`conv`** | `Optional[MessagePassing]` | `None` | L'operatore di message passing locale (es. `GCNConv`, `GINConv`, `GATConv`). Se impostato a `None`, il layer esegue solo l'attenzione globale senza estrarre strutture locali. |
| **`heads`** | `int` | `1` | Numero di teste ($H$) per la Multi-Head Attention globale. |
| **`dropout`** | `float` | `0.0` | Probabilità di dropout applicata agli embedding intermedi e ai pesi di attenzione. |
| **`act`** | `str` o `Callable` | `"relu"` | Funzione di attivazione non lineare (es. `"relu"`, `"gelu"`). |
| **`act_kwargs`** | `Dict[str, Any]` | `None` | Argomenti aggiuntivi da passare alla funzione di attivazione. |
| **`norm`** | `str` o `Callable` | `"batch_norm"` | Tipo di normalizzazione applicata, tipicamente `"batch_norm"` o `"layer_norm"`. |
| **`norm_kwargs`** | `Dict[str, Any]` | `None` | Argomenti opzionali per il layer di normalizzazione. |
| **`attn_type`** | `str` | `"multihead"` | Tipo di meccanismo di attenzione globale utilizzato. Opzioni comuni includono `"multihead"` (attenzione quadratica esatta, complessa $\mathcal{O}(N^2)$) o `"performer"` (attenzione lineare approssimata per grafi di grandi dimensioni). |
| **`attn_kwargs`** | `Dict[str, Any]` | `None` | Argomenti addizionali per configurare internamente il layer di attenzione globale. |
