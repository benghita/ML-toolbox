# ML-Toolbox - A Simplified Machine Learning Package

ML-Toolbox simplifies machine learning for regression and classification tasks, designed for small datasets. It offers a user-friendly Streamlit interface for data preprocessing and model selection.

## Features

- Automated data preprocessing: Handling missing values, encoding, normalization, feature selection, and more.
- Model selection: Training and evaluating regression and classification models.
- Streamlit app for easy interaction.

## Getting Started

### Prerequisites

- Python 3.6+
- Git

### Installation

1. Clone the ML-Toolbox repository:

   ```bash
   git clone https://github.com/benghita/ML-Toolbox.git
   cd ML-Toolbox
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Usage

Run the ML-Toolbox Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

- **ml_toolbox**: Python package.
  - **AutoML.py**: Main class for data preprocessing and model training.
  - **data_preprocessing.py**: Functions for data preprocessing.
  - **models_training.py**: Classes for regression and classification models.
- **app.py**: Streamlit application.
- **.gitignore**: Git configuration.
- **pyvenv.cfg**: Virtual environment config.
- **requirements.txt**: List of required packages.

## Usage

To interact with the ML-Toolbox Streamlit app, visit the following link:

[ML-Toolbox Streamlit App](https://benghita-ml-toolbox.streamlit.app/)

## Customization

Modify `models` and `metrics` dictionaries in `models_training.py` to customize models and metrics.