# Superconductor Critical Temperature Prediction

A machine learning project predicting the superconducting transition 
temperature (T_c) from elemental composition features, combining 
data-driven modeling with solid-state physics domain expertise.

## Motivation

Superconductivity remains one of the most fascinating phenomena in 
condensed matter physics. Since the discovery of high-T_c cuprate 
superconductors in 1986, the search for room-temperature 
superconductors has intensified. Data-driven approaches can accelerate 
this search by identifying patterns in known superconductors and 
guiding the exploration of new material candidates.

This project demonstrates a complete ML workflow - from exploratory 
data analysis to model deployment - with a strong emphasis on 
**physically grounded interpretation** of both data and model results.

## Dataset

The [Superconductivity Dataset](https://archive.ics.uci.edu/ml/datasets/superconductivty+data) 
from the UCI Machine Learning Repository contains:

- **21,263 superconducting materials**
- **81 features** derived from 8 elemental properties (atomic mass, 
  ionization energy, atomic radius, density, electron affinity, 
  fusion heat, thermal conductivity, valence), aggregated via 
  weighted means, geometric means, ranges, standard deviations, 
  and entropy measures
- **Target variable**: critical temperature T_c (K), ranging from 
  0.00021 K to 185 K

Source: Hamidieh, K. (2018). *A Data-Driven Statistical Model for 
Predicting the Critical Temperature of a Superconductor.* 
[arXiv:1803.10260](https://arxiv.org/abs/1803.10260)

## Project Architecture

The project follows a modular, object-oriented design pattern to 
ensure separation of concerns, reusability, and testability:
superconductor-tc-prediction/ ├── data/ │ ├── raw/ # Original UCI dataset (not tracked) │ └── processed/ # Cleaned data (not tracked) ├── src/ │ ├── data_loader.py # Abstract Base Class + concrete loader │ ├── preprocessor.py # Feature-target split, scaling, CV │ ├── models.py # Strategy Pattern: BaseModel → LR, RF │ ├── evaluator.py # Visualization: predictions, residuals │ └── pipeline.py # Orchestrator (composition) ├── notebooks/ │ ├── 01_EDA.ipynb # Exploratory data analysis │ ├── 02_Modeling.ipynb # Model training & evaluation │ └── 03_Results.ipynb # Summary of findings ├── tests/ │ └── test_pipeline.py # Unit tests ├── requirements.txt └── README.md

## Design Patterns Used

Abstract Base Classes (ABC): BaseDataLoader and BaseModel enforce a consistent interface across implementations.
Strategy Pattern: Multiple model implementations (LinearRegressionModel, RandomForestModel) share a common interface, enabling easy model swapping and comparison.
Composition: The Pipeline class orchestrates all components through object composition, avoiding tight coupling.

# How to Run

## 1. Clone the repository
git clone https://github.com/KKulmus/superconductor-tc-prediction.git
cd superconductor-tc-prediction

## 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

## 3. Install dependencies
pip install -r requirements.txt

## 4. Download data
Download train.csv and unique_m.csv from:
[https://archive.ics.uci.edu/ml/datasets/superconductivty+data](https://archive.ics.uci.edu/ml/datasets/superconductivty+data) 
Place them in data/raw/

## 5. Run notebooks
jupyter lab
## Open notebooks/01_EDA.ipynb, then notebooks/02_Modeling.ipynb

Tech Stack

    Python 3.10+
    pandas: data manipulation
    scikit-learn: ML models, preprocessing, evaluation
    matplotlib / seaborn:  visualization
    Jupyter: interactive analysis

## Author

Dr. Kathrin Kulmus

    PhD in Theoretical Physics
    Data Scientist (StackFuel certified)
    C++ development experience (QAT, Fraunhofer EMI)
    GitHub: @KKulmus


