# Hi, I’m Asya 👋

🎓 Dartmouth College ’26 — Computer Science major (Studio Art minor)  
💻 Focus: data systems, AR/VR engineering, machine learning, and computational design  
🌍 Multilingual: Turkish (native), English (fluent), Spanish (intermediate)  

I build **technical systems that turn complex data into actionable outcomes**.  
From high-performance genome sequencing pipelines to AR/VR platforms with real-time interaction, my work combines **rigorous computational methods** with a human-centered approach to problem solving.

---

## 🛠️ Technical Skills
**Programming:** Python, C/C++, C#, Java, R, SQL, JavaScript  
**Systems & Tools:** Linux, GitHub, Bash, Unity, HPC environments, MATLAB  
**Machine Learning:** scikit-learn, pandas, NLP pipelines, clustering/classification  
**Other:** Rendering algorithms, AR/VR SDKs, Maya, Adobe Creative Suite  
## 🔭 Featured Projects

### ⚡ Cloud Tank: AR/VR Systems Research
Developed an **XR platform on Meta Quest 3** using **C#/C++ and Unity**.  
- Implemented **real-time pipelines** for audiovisual data remixing through hand and body tracking.  
- Engineered **projection + headset streaming architecture** for multiple deployment formats.  
- Built a reusable **2D/3D media asset library** with audio-reactive visuals and texture mapping.  
Outcome: validated system via **pilot performances with live feedback** and explored XR as a scalable platform for data-driven environments. :contentReference[oaicite:0]{index=0}

---

### 🧬 Bioinformatics Data Sequencing
Led development of Python + Linux pipelines for **Nanopore long-read genome data** with the University of Groningen.  
- Designed a **mutation-detection and visualization app** for phenotypic virus scanning.  
- Deployed on a **high-performance computing cluster (Hábrók)**, handling terabytes of genomic data.  
- Coordinated integration with CNRS collaborators, aligning software engineering with biological research needs. :contentReference[oaicite:1]{index=1}  
Outcome: produced scalable workflows for interpreting complex biological datasets.  

---

### 📊 Product Review Machine Learning
Built a complete **NLP + clustering pipeline** for **100k+ consumer reviews**.  
- Extracted features with TF-IDF, Count, and Hashing vectors.  
- Trained and compared **Logistic Regression, SVM, and K-Means**, optimizing with stratified 5-fold **GridSearchCV (300+ grids)**.  
- Evaluated with ROC-AUC, F1, silhouette scores, and confusion matrices.  
- Delivered results in **interactive dashboards** tailored for business stakeholders. :contentReference[oaicite:2]{index=2}  
Outcome: translated unstructured data into **evidence-based product segmentation**.

<details>
<summary><b>Sample code: TF-IDF + Logistic Regression for Binary Sentiment</b></summary>

```python
# Reproducible TF-IDF + Logistic Regression for binary sentiment (cutoff = 2/3)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from joblib import dump, load

RNG = 42

def label_by_cutoff(stars, cutoff): 
    return (stars > cutoff).astype(int)

# --- Data ---
# df = pd.read_csv("amazon_reviews.csv")  # columns: 'reviewText', 'overall'
# y = label_by_cutoff(df["overall"], cutoff=2)
# X = df["reviewText"].fillna("")
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=RNG
# )

# --- Pipeline & Search ---
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_df=0.9, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=200, solver="liblinear", random_state=RNG))
])

param_grid = {
    "tfidf__ngram_range": [(1,1), (1,2)],
    "clf__C": [0.1, 1, 10],
    "clf__class_weight": [None, "balanced"],
    "clf__penalty": ["l2"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
grid = GridSearchCV(pipe, param_grid=param_grid, scoring="f1_macro", cv=cv, n_jobs=-1)

# grid.fit(X_train, y_train)

# --- Threshold sweep (maximize macro-F1 on validation) ---
def best_threshold(estimator, X_val, y_val):
    prob = estimator.predict_proba(X_val)[:, 1]
    threshes = np.linspace(0.2, 0.8, 61)
    scores = [(t, f1_score(y_val, (prob >= t).astype(int), average="macro")) for t in threshes]
    t_star, f1_star = max(scores, key=lambda x: x[1])
    return t_star, f1_star

# y_pred = grid.predict(X_test)
# y_prob = grid.predict_proba(X_test)[:,1]
# t_star, f1_star = best_threshold(grid.best_estimator_, X_test, y_test)
# y_pred_tuned = (y_prob >= t_star).astype(int)

# print("Best params:", grid.best_params_)
# print("Macro F1 (0.5):", f1_score(y_test, y_pred, average="macro"))
# print("Macro F1 (tuned):", f1_score(y_test, y_pred_tuned, average="macro"))
# print("ROC-AUC:", roc_auc_score(y_test, y_prob))
# print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_tuned))
# print(classification_report(y_test, y_pred_tuned))

# --- Persist model & quick inference ---
# dump(grid.best_estimator_, "sentiment_model.joblib")

# model = load("sentiment_model.joblib")
# new_texts = pd.Series(["Fast shipping, great quality!", "Arrived broken and support never replied."])
# preds = model.predict(new_texts)
# probs = model.predict_proba(new_texts)[:,1]
# print(list(zip(new_texts.tolist(), preds.tolist(), probs.tolist())))
```
</details>

<details> 
<summary><b>Sample code:Multiclass Rating</b></summary>

```python
# 5-class rating prediction with TF-IDF + LogisticRegression (OvR)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, classification_report

RNG = 42

# df = pd.read_csv("amazon_reviews.csv")  # cols: 'reviewText','overall' (1..5)
# X = df["reviewText"].fillna("")
# y = df["overall"].astype(int)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=RNG
# )

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_df=0.9, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=300, multi_class="ovr", solver="liblinear", random_state=RNG)),
])

param_grid = {
    "tfidf__ngram_range": [(1,1), (1,2)],
    "clf__C": [0.1, 1, 10],
    "clf__class_weight": [None, "balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
grid = GridSearchCV(pipe, param_grid=param_grid, scoring="f1_macro", cv=cv, n_jobs=-1)

# grid.fit(X_train, y_train)
# y_pred = grid.predict(X_test)

# print("Best params:", grid.best_params_)
# print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
# print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
```
</details>

<details> 
<summary><b>Sample code:Clustering via LSA</b></summary>

```python
# Cosine-aware clustering via LSA + Normalizer + Agglomerative (average linkage)
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from joblib import dump, load

RNG = 42

# df = pd.read_csv("amazon_reviews.csv")  # col: 'reviewText'
# docs = df["reviewText"].fillna("")

def lsa_embed(docs, n_components=100):
    pipe = make_pipeline(
        TfidfVectorizer(max_df=0.9, ngram_range=(1,2)),
        TruncatedSVD(n_components=n_components, random_state=RNG),
        Normalizer(copy=False)
    )
    X = pipe.fit_transform(docs)
    return pipe, X

def cluster_and_score(X, k_list=(5, 8, 10, 12, 15)):
    results = []
    for k in k_list:
        labels = AgglomerativeClustering(
            n_clusters=k, linkage="average", metric="cosine"
        ).fit_predict(X)
        score = silhouette_score(X, labels, metric="cosine")
        results.append((k, score, labels))
    results.sort(key=lambda t: t[1], reverse=True)
    return results

# pipe, X_lsa = lsa_embed(docs, n_components=100)
# results = cluster_and_score(X_lsa, k_list=(5,8,10,12,15))
# print("Top-3 (k, silhouette):", [(k, round(s, 4)) for k,s,_ in results[:3]])

# best_k, best_s, best_labels = results[0]
# print("Best k:", best_k, "Silhouette:", best_s)

# Persist the embedding pipeline and labels for downstream analysis/visualization
# dump(pipe, "lsa_text_pipeline.joblib")
# pd.Series(best_labels, name="cluster").to_csv("cluster_labels.csv", index=False)
```
</details>

### ⚙️ TSE Querier (C Systems Project)
- Implemented a **C-based query parser and ranking engine** with abstract data structures.  
- Optimized document scoring with Boolean operators and minimal memory footprint.  
- Built a suite of **unit and regression tests** ensuring zero memory leaks. :contentReference[oaicite:3]{index=3}  
Outcome: demonstrated ability to design **efficient, low-level systems** for high-precision information retrieval.  

---

### 🎮 AR/VR Development Projects
- Created **real-time AR object recognition and spatial mapping** for an interactive environment.  
- Led a 4-person team to develop a multi-part VR experience (ray casting, shaders, hand tracking).  
- Presented to **50+ users at Techigala**, showcasing responsive, cross-platform interaction. :contentReference[oaicite:4]{index=4}  
Outcome: proved capability in **team leadership, rapid prototyping, and XR deployment**.  


---

## 📫 Let’s Connect
- 🌐 [Portfolio](https://journeys.dartmouth.edu/asyaulger/)  
- 💼 [LinkedIn](https://www.linkedin.com/in/asya-ulger-7452a02b2/)  
- ✉️ asya.ulger.26@dartmouth.edu  
