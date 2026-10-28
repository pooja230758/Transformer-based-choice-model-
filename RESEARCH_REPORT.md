# Comprehensive Research Report: Transformer-Based Choice Modeling & Assortment Optimization (Airline Case Study)

**Course / Project**: IIT Kanpur MTH498 — Project Report & Evaluation  
**Author**: Pooja Kumari (Roll No. 230758)  
**Reference Paper**: Peng, Z., Rong, Y., & Zhu, T. (2024). *"Transformer‐based choice model: A tool for assortment optimization evaluation."* *Naval Research Logistics (NRL)*, 71(6).  
**Repository**: [github.com/pooja230758/Transformer-based-choice-model-](https://github.com/pooja230758/Transformer-based-choice-model-) (Derived from [zhupig0501/Transformer-Choice-Model/Airline](https://github.com/zhupig0501/Transformer-Choice-Model/tree/main/Airline))

---

## 1. Literature & Model Comparison

Discrete choice models predict customer selection among a discrete set of alternatives (an *assortment* $S_t = \{1, \dots, |S_t|\}$) or the decision to purchase nothing (the *no-purchase / outside option*, indexed by $0$). In revenue management, understanding substitution patterns (which product a consumer substitutes when their top pick is unavailable or overpriced) is critical for setting optimal assortments and ticket prices.

The paper investigates five choice models with distinct theoretical and operational paradigms:

### 1.1 Multinomial Logit (MNL)
* **Purpose**: Classical benchmark choice model widely adopted in operations research and revenue management due to its closed-form likelihood and polynomial-time assortment optimization.
* **Formulation**: Each alternative $i \in S_t$ has deterministic utility $u_i = w^T x_i + b$. Choice probability is given by:
  $$P(i \mid S_t) = \frac{\exp(u_i)}{\sum_{j \in S_t} \exp(u_j) + \exp(u_0)}$$
  where $u_0$ denotes the outside option utility.
* **Assumptions**: 
  - Utilities follow independent and identically distributed (i.i.d.) Gumbel error distributions.
  - **Independence of Irrelevant Alternatives (IIA)**: The relative odds $\frac{P(i)}{P(j)} = \frac{\exp(u_i)}{\exp(u_j)}$ are strictly independent of any other product in the assortment.
* **Limitations**: Unrealistic substitution patterns (proportional substitution). Cannot capture asymmetric substitution, complementary effects, or customer preference heterogeneity.

### 1.2 Mixed Multinomial Logit (MMNL / Latent Class MNL)
* **Purpose**: Relaxes the IIA restriction by modeling customer heterogeneity through a finite mixture of $K$ latent market segments.
* **Formulation**:
  $$P(i \mid S_t) = \sum_{k=1}^K \alpha_k \frac{\exp(u_{k, i})}{\sum_{j \in S_t} \exp(u_{k, j}) + \exp(u_{k, 0})}, \quad \sum_{k=1}^K \alpha_k = 1$$
* **Assumptions**: Segment-conditional IIA holds within each latent class, but the aggregate population choice violates IIA.
* **Limitations**: Assortment optimization under MMNL is generally NP-hard. Training requires Expectation-Maximization (EM) or Frank-Wolfe / Column Generation (`CG_algo`), which is computationally demanding.

### 1.3 DeepFM (Deep Factorization Machine)
* **Purpose**: Originally proposed by Guo et al. (2017) for recommendation systems and CTR prediction; learns non-linear high-order feature interactions without manual feature engineering.
* **Formulation**: Computes product purchase logit $\hat{y}_i = y_{\text{linear}} + y_{\text{FM}} + y_{\text{DNN}}$, where $y_{\text{FM}}$ captures pairwise inner products of factorized embeddings $\langle v_i, v_j \rangle$, and $y_{\text{DNN}}$ is a multi-layer perceptron. Choice probability is obtained via independent sigmoid classification $\sigma(\hat{y}_i)$, normalized across the assortment.
* **Assumptions**: Treats each alternative's purchase probability independently; ignores assortment-level context and mutual exclusivity during feature extraction.
* **Limitations**: Blind to substitution effects; predicts identical scores for product $i$ regardless of whether the other products in the assortment are high-end or budget flights.

### 1.4 DeepFM-a (DeepFM with Assortment Context)
* **Purpose**: Augments DeepFM by concatenating 11 assortment-level summary features to each product's feature vector (expanding from 17 to 28 input fields).
* **Formulation**: Appends minimum, maximum, and average values of prices, stay duration, and flight trip duration, along with binary indicators `is_cheapest` and `is_most_expensive` in the current assortment.
* **Assumptions**: Assumes hand-crafted summary statistics adequately summarize assortment context.
* **Limitations**: Feature engineering is heuristic. As assortment composition shifts, summary statistics change discontinuously, making optimization non-smooth and requiring heuristic local search (Algorithm 1 swap heuristic).

### 1.5 Transformer-Based Choice Model (The Proposed Simulator)
* **Purpose**: Uses self-attention over the unordered set of candidate alternatives $S_t$ to directly learn context-dependent substitution patterns and cross-product interactions.
* **Formulation**:
  - Feature embedding: Continuous features pass through BatchNorm; categorical features pass through embedding layers to construct representations $H^{(0)} \in \mathbb{R}^{|S_t| \times d}$.
  - Multi-head self-attention:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  - Masked Softmax Cross-Entropy Loss over variable assortment lengths:
    $$P(i \mid S_t) = \text{MaskedSoftmax}_i\left(W_{\text{dense}} H^{(L)}\right)$$
* **Modifications over standard NLP Transformers**:
  1. **No Positional Encoding**: Assortments are sets, not ordered linguistic sequences. Permutation invariance/equivariance is preserved.
  2. **No Autoregressive Decoder**: A simple linear projection + masked softmax replaces the autoregressive decoder to generate joint choice probabilities across all alternatives simultaneously.
* **Limitations**: High computational latency ($O(|S_t|^2)$ attention complexity); requires large transaction volumes; "black box" non-convex structure makes mathematical assortment optimization NP-hard, requiring heuristic evaluation or simulation.

---

## 2. Dataset Understanding (Airline Booking Case Study)

The empirical validation utilizes synthetic airline reservation transactions modeled after Amadeus air ticket booking data:

* **File**: `data_github/transaction_data.csv` (uncompressed from `airline_data.zip`).
* **Volume**: **524,808 product rows** across **30,000 customer assortments**.
* **Assortment Size ($|S_t|$)**: Ranges from **5 to 30 candidate flights**, with an average of **17.49 flights** per assortment (median 18.0).
* **Label Distribution**:
  - Purchase observed (`label = 1`): **21,606 assortments** (~72.0%).
  - Outside option chosen (`label = 0` for all products): **8,394 assortments** (~28.0%).

### Input Feature Breakdown (17 Base Features)
1. **7 Categorical Features**:
   - `officeID` (11 categories): Travel agency office identifier.
   - `depWeekDay` (7 categories, 0–6): Departure day of the week.
   - `OD` (97 categories): Origin-Destination airport pair.
   - `fAirline` (63 categories): First operating airline carrier.
   - `staySaturday` (binary, 0 or 1): Itinerary includes a Saturday night stay.
   - `isContinental` (binary, 0 or 1): Intercontinental vs. domestic flight.
   - `isDomestic` (binary, 0 or 1): Domestic route indicator.
2. **10 Continuous Features** (Min-Max normalized to $[0, 1]$):
   - `stayDurationMinutes`: Length of stay at destination.
   - `totalPrice`: Airfare ticket price (USD).
   - `totalTripDurationMinutes`: Total elapsed flight and layover time.
   - `dtd`: Days to departure (booking window).
   - `nAirlines`: Number of distinct operating airlines in the ticket.
   - `nFlights`: Total flight segments/legs.
   - `outDepTime_sin`, `outDepTime_cos`: Cyclic departure time encoding.
   - `outArrTime_sin`, `outArrTime_cos`: Cyclic arrival time encoding.

---

## 3. Experimental Setup & Methodology

To ensure strict scientific rigor, reproducibility, and prevent data leakage:
* **Seed**: Deterministic seed (`seed = 42`) across Python, NumPy, and PyTorch.
* **Data Partitioning**: Splitting is performed strictly at the **order level** (assortment level) rather than the row level, ensuring no flight options from the same customer search appear in both training and testing.
  - **Train Set**: 4,000 assortments (69,303 product rows).
  - **Validation Set**: 1,000 assortments (17,197 product rows).
  - **Test Set**: 1,000 assortments (17,298 product rows; 721 purchase assortments, 279 outside option assortments).
* **Evaluation Metrics**:
  - **Top-1 Accuracy (%)**: Exact match between the predicted highest-probability alternative and the customer's actual choice.
  - **Top-3 Accuracy (%)**: Whether the chosen flight appears in the model's top 3 recommendations.
  - **Negative Log-Likelihood / Log Loss**: Multi-class cross-entropy on choice sets $-\frac{1}{T}\sum_t \log(p_{t, \text{chosen}})$.
  - **RMSE**: Root mean squared error between predicted choice probabilities and one-hot ground-truth labels.
  - **Training Time (s)**: Wall-clock time required to train the model.
  - **Inference Latency (ms)**: Average time in milliseconds to score a complete assortment.

---

## 4. Empirical Results & Performance Comparison

The complete benchmarking pipeline was executed on the test set. All models were trained and evaluated on identical splits.

### Table 1: Choice Model Performance Comparison

| Model Architecture | Top-1 Accuracy (%) | Top-3 Accuracy (%) | Log Loss (NLL) | RMSE | Training Time (s) | Inference Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MNL** | 41.51% | 74.61% | 1.813 | 0.1837 | **6.30s** | **0.005 ms** |
| **MMNL (3 classes)** | 42.64% | 74.89% | 1.761 | 0.1831 | 14.30s | 0.020 ms |
| **DeepFM** | 32.40% | 62.69% | 2.132 | 0.1970 | 12.41s | 0.029 ms |
| **DeepFM-a** | 34.64% | 66.76% | 2.244 | 0.1973 | 157.76s | 0.047 ms |
| **Transformer** | **59.47%** | **86.68%** | **1.206** | **0.1726** | 236.45s | 1.126 ms |

---

## 5. Extra Experiment: Small vs. Large Assortments

A key theoretical question is how choice models behave as assortment complexity increases. We partitioned the test set into:
* **Small Assortments** ($\le 15$ candidate flights; lower consideration set, less substitution competition).
* **Large Assortments** ($> 15$ candidate flights; dense candidate sets, high substitution, risk of choice overload).

### Table 2: Accuracy Across Assortment Sizes

| Model Architecture | Small Assortments ($\le 15$ flights) Top-1 (%) | Large Assortments ($> 15$ flights) Top-1 (%) | Absolute Degradation | Relative Degradation |
| :--- | :---: | :---: | :---: | :---: |
| **MNL** | 54.77% | 32.79% | -21.98% | -40.1% |
| **MMNL** | 57.60% | 32.79% | -24.81% | -43.1% |
| **DeepFM** | 43.46% | 25.12% | -18.34% | -42.2% |
| **DeepFM-a** | 47.00% | 26.51% | -20.49% | -43.6% |
| **Transformer** | **69.61%** | **52.79%** | **-16.82%** | **-24.2%** |

### Key Observations:
1. All models experience lower Top-1 accuracy as assortment size increases because the baseline random-guess probability drops from $\approx 1/10$ to $\approx 1/25$.
2. **The Transformer demonstrates superior robustness**: its accuracy drops by only **24.2% relatively** (from 69.61% down to 52.79%), whereas MNL and MMNL drop by **over 40% relatively** (down to 32.79%).
3. On large assortments, pairwise substitution effects become critical. Because the Transformer's self-attention layers compute all-to-all attention scores between candidate flights, it successfully detects which flight dominates others in price-schedule trade-offs.

---

## 6. Critical Analysis: Trade-offs & Practical Guidelines

### 6.1 Accuracy vs. Computational Latency
* **MNL & MMNL** provide exceptional inference speed (**0.005 ms - 0.020 ms** per assortment), over **50x to 200x faster** than the Transformer (**1.126 ms**).
* In high-throughput, millisecond-critical environments (such as Global Distribution Systems / Amadeus processing tens of thousands of search requests per second), MNL can easily be embedded in C++ microservices with minimal CPU footprint.
* The Transformer requires matrix operations over $320 \times 88$ tensor representations, creating significant GPU/CPU memory overhead during live web serving.

### 6.2 Interpretability & Regulatory Compliance
* **MNL** features directly interpretable coefficients: $w_{\text{price}}$ indicates the marginal disutility of price, allowing immediate calculation of willingness-to-pay (WTP) and price elasticities ($\epsilon_{ij}$).
* **Transformer** is a non-linear black box. Extracting clear economic elasticities or proving fair pricing compliance to regulatory aviation bodies is non-trivial.

### 6.3 Downstream Assortment Optimization
* Under **MNL**, the revenue-maximizing assortment can be found in polynomial time using the revenue-ordered threshold algorithm (Rusmevichientong et al., 2010).
* Under the **Transformer**, finding the optimal assortment $S^* = \arg\max_{S \subseteq V} \sum_{i \in S} r_i P(i \mid S)$ is an intractable combinatorial optimization problem because every product's probability depends non-linearly on all other items in $S$. Solvers must rely on greedy heuristics or simulated annealing.

---

## 7. Conclusions: When to Use Simpler Models vs. The Transformer

| Operational Scenario | Recommended Model | Primary Justification |
| :--- | :--- | :--- |
| **Real-time online flight pricing & dynamic ranking** | **MNL / MMNL** | Ultra-low latency (<0.02ms), guaranteed polynomial-time assortment optimization, closed-form elasticities. |
| **Offline Assortment Evaluation / Digital Twin Simulator** | **Transformer** | State-of-the-art Top-1 accuracy (59.47%) and Log Loss (1.206); accurately simulates complex customer substitution and choice overload without expensive A/B field tests. |
| **Personalized Click-Through Scoring (Large Catalogues)** | **DeepFM-a** | Scales linearly with product catalogue size; captures categorical feature interactions without $O(N^2)$ self-attention cost. |
| **Market Heterogeneity with Explainable Segments** | **MMNL** | Discovers latent business vs. leisure passenger classes while retaining parametric utility interpretations. |

In summary, the authors' central thesis is validated: **The Transformer model is ideally suited as an offline "Simulator"**—a digital twin of real passenger behavior that allows revenue management teams to safely evaluate, stress-test, and benchmark simpler operational choice models prior to deployment.
