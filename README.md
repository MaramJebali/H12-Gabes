# 🌍 OASIS Gabès - Environmental Intelligence & Ecotourism Platform

**Protect, restore, and showcase Gabès' ecosystem through AI, data science, and community engagement**


---

## 📌 Background

Gabès stands at a critical crossroads. Between historical industrial pollution (phosphogypsum, heavy metals), climate stressors (drought, extreme heat), and increasing pressure on water resources, both the ecosystem and local communities are under severe threat.

*Our mission*: Turn this crisis into an opportunity by leveraging data science, artificial intelligence, and citizen participation.

---

## 🎯 The 3 Pillars of Our Solution

| Pillar | Goal | Technologies |
|--------|------|--------------|
| 🌱 *Ecological Sustainability* | Pollution detection, climate prediction, palm tree health | AI, CNN, RandomForest, Clustering, RAG, Groq LLM |
| 🧭 *Ecotourism* | Make Gabès a sustainable destination | Django, CRUD, Booking system, Local guides |
| 👥 *Community* | Empower farmers, guides, and local authorities | Collaborative platform, alerts, recommendations |

---

## 🧠 AI & Smart Systems Components

### 1. Water Pollution & Heavy Metals (Gulf of Gabès)

*Problem*: Historical phosphogypsum discharge → cadmium, lead, mercury, chromium.

*Solution*:
- Concentration analysis in 4 mollusc species
- *Extinction threshold* calculation for Pinna nobilis (2.48 µg/g Cd)
- Species extinction risk prediction
- Interactive dashboard for decision-makers

*Technologies*: Python, Streamlit, Pandas, NumPy

📁 *Documentation included: documentation.docx*

---

### 2. OASIS AI – 7-Day Climate Prediction (Supervised)

*Problem*: Water and heat stress threatening oasis agriculture.

*Solution*:
- Predicts soil moisture (GWETROOT) and max temperature (T2M_MAX) 7 days ahead
- NASA POWER real data
- RandomForest supervised model
- Alert system (Low / Medium / Critical)
- Explanations & recommendations via *Groq LLM* (llama-3.3-70b)

*Technologies*: Python, Scikit-learn, RandomForest, Groq API, NASA POWER

---

### 3. OASIS AI – Environmental Segmentation (Unsupervised)

*Problem*: Understand hidden climate patterns in Gabès.

*Solution*:
- KMeans clustering on NASA data
- 4 environmental profiles detected:
  - Cluster 0: Balanced conditions
  - Cluster 1: Moderate water stress
  - Cluster 2: High water stress
  - Cluster 3: Unstable conditions
- PCA visualization

*Technologies*: Python, KMeans, PCA, Scikit-learn

---

### 4. Palm Tree Disease Detection (Computer Vision)

*Problem*: Anthracnose and Chimaera threaten palm trees.

*Solution*:
- Fine-tuned CNN (ResNet18)
- Classification: Anthracnose / Chimaera / Healthy
- Interpretability via *GradCam + heatmap*

*Technologies*: PyTorch, ResNet18, GradCam, OpenCV

---

### 5. Advanced RAG + Hybrid AI

*Goal*: Answer environmental questions about Gabès intelligently.

*Pipeline*:
- Structured data (gabes_details.json)
- Embeddings (multilingual-e5-base)
- Hybrid search: FAISS (vector) + BM25 (lexical)
- Re-ranking with cross-encoder (ms-marco-MiniLM-L-6-v2)
- Generation with *Groq LLM* (llama-3.3-70b)

*Use case*: Personalized recommendations for farmers, authorities, investors

---

## 🏛️ Virtual Museum (Unity)

An immersive virtual museum to discover Gabès' biodiversity, history, and environmental challenges.


- Built with *Unity*
- Accessible via browser or standalone app
- Raises public awareness about oasis and coastal preservation

---

## 🧑‍💻 Django Platform (CRUD for Tour Guides & Tourism)

*Features*:
- *Local guide registration* (profile, availability, pricing)
- *Full CRUD* (Create, Read, Update, Delete) for guides
- *Booking system* for tourists to reserve guides
- Filter by season, budget, preferences
- Secure authentication

*Technologies*: Django, SQLite/PostgreSQL, HTML/CSS, Bootstrap
