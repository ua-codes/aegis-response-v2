# 🚨 Disaster Response Management System — v2.0

An end-to-end operational pipeline designed for emergency command centers. This system automates the intake, triage, and deployment of emergency resources by linking satellite/drone visual intelligence directly to regional allocation models and volunteer matching logic.

---

## 🗺️ System Architecture

Our solution breaks down disaster logistics into a unified, cascading multi-model pipeline:

📷 Aerial Payload (Drone/Satellite)
│
▼
┌──────────────────────────────────────┐
│  MODEL 1: Visual Severity Engine     │ ──► EfficientNet-B3 (Fine-Tuned)
│  Tracks damage levels & confidence   │ ──► Calculates initial Risk Assessment
└──────────────────┬───────────────────┘
│ Outputs metrics into
▼
┌──────────────────────────────────────┐
│  MODEL 2: Resource Dispatch Engine   │ ──► LightGBM Multi-Output Regressor
│  Applies NIMS rule-based safety nets │ ──► Allocates SAR, Medical, and Fire teams
└──────────────────┬───────────────────┘
│ Leverages criteria for
▼
┌──────────────────────────────────────┐
│  MODEL 3: Volunteer Skill Matcher    │ ──► TF-IDF + Cosine Similarity
│  Deploys specialized rescue forces   │ ──► Matches field assets to operational gaps
└──────────────────────────────────────┘

---

## ✨ Key Capabilities

* **Intelligent Visual Intake:** Utilizes an upgraded `EfficientNet-B3` deep learning backbone to categorize structural destruction levels instantly.
* **Metadata-Driven Dispatch:** Directly factor in safety-critical dimensions—such as the number of trapped victims—to drive a Multi-Output LightGBM resource allocation engine.
* **National Incident Management System (NIMS) Safety Net:** Incorporates hard rule-based overrides that step in to enforce minimum resource allocations if predictive algorithms under-estimate catastrophic events.
* **Human-in-the-Loop Safeguards:** Flags low-confidence model predictions automatically, routing highly uncertain variables to senior incident commanders for manual sign-off.
* **Semantic Volunteer Coordination:** Ranks emergency responders against real-time disaster demands, dynamically highlighting skill gaps.

---

## 📦 Project Setup & Local Installation

### Prerequisites
Make sure you have Python 3.10+ installed on your system.

### 1. Clone the Repository
```bash
git clone [https://github.com/ua-codes/aegis-response-v2.git](https://github.com/ua-codes/aegis-response-v2.git)
cd aegis-response-v2