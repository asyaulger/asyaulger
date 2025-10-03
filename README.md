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
<summary><b>Sample code: TF-IDF + Logistic Regression with stratified GridSearchCV</b></summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report

# df = pd.read_csv("amazon_reviews.csv")  # expects columns: 'reviewText', 'overall'
def label_by_cutoff(stars, cutoff): return (stars > cutoff).astype(int)

cutoff = 2
# y = label_by_cutoff(df["overall"], cutoff)
# X = df["reviewText"].fillna("")
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_df=0.9, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=200, solver="liblinear"))
])

param_grid = {
    "tfidf__ngram_range": [(1,1), (1,2)],
    "clf__C": [0.1, 1, 10],
    "clf__class_weight": [None, "balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid=param_grid, scoring="f1_macro", cv=cv, n_jobs=-1)

# grid.fit(X_train, y_train)
# y_pred = grid.predict(X_test)
# y_prob = grid.predict_proba(X_test)[:, 1]

# print("Best params:", grid.best_params_)
# print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
# print("ROC-AUC:", roc_auc_score(y_test, y_prob))
# print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

<details>
<summary><b>Sample code: TF-IDF → TruncatedSVD (LSA) → Agglomerative clustering (cosine) with Silhouette</b></summary>

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# df = pd.read_csv("amazon_reviews.csv")  # expects column: 'reviewText'
# docs = df["reviewText"].fillna("")

lsa_pipe = make_pipeline(
    TfidfVectorizer(max_df=0.9, ngram_range=(1,2)),
    TruncatedSVD(n_components=100, random_state=42),
    Normalizer(copy=False)
)

# X_lsa = lsa_pipe.fit_transform(docs)

def cluster_and_score(X, k_list=(5, 8, 10, 12, 15)):
    results = []
    for k in k_list:
        labels = AgglomerativeClustering(
            n_clusters=k, linkage="average", metric="cosine"
        ).fit_predict(X)
        score = silhouette_score(X, labels, metric="cosine")
        results.append((k, score))
    return sorted(results, key=lambda t: t[1], reverse=True)

# results = cluster_and_score(X_lsa)
# best_k, best_score = results[0]
# print("k, silhouette:", results)
# print("best:", best_k, best_score)

</details>

---

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
