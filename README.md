# Marsci Simulation

<h1 align="center" id="heading"> An Open Source Method to Generate Data for Marketing</h1>

[![Python][Python.py]][Python-url]

A Python-based simulation framework for generating synthetic marketing and sales data, applying adstock and diminishing returns transformations, and visualizing results for marketing mix modeling (MMM) experiments.

This project can be run locally using Jupyter Notebook or on Google Colab with no local setup required.

---

## Project Structure

```
marsci_simulation/
├── pySimul.ipynb       # Main simulation notebook
├── utils/              # Helper functions for data generation & plotting
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

---

## Running the Simulation

### 1. Run in Google Colab (Recommended)
You can run the notebook directly in Google Colab without installing anything locally:

1. Click this badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mBd3pJVEVQ6wusy7DNQcVvfsOVjsvakJ?usp=sharing).

2. In the first cell, clone the repository:
   ```python
   !git clone https://github.com/Tayerquach/marsci_simulation.git
   %cd marsci_simulation
   ```
3. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
4. Run the rest of the cells in sequence.

---

### 2. Run Locally

#### **Prerequisites**
- Python
- Jupyter Notebook or JupyterLab
- pip package manager

#### **Setup**
```bash
# Clone the repository
git clone https://github.com/Tayerquach/marsci_simulation.git
cd marsci_simulation

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

#### **Run the Notebook [![Jupyter][jupyterlab.ipynb]][jupyterlab-url]**
```bash
jupyter notebook pySimul.ipynb
```
Then execute the cells in order to generate synthetic marketing data and view plots.

---

## Features
- Synthetic baseline sales and ad spend generation
- Campaign scheduling with customizable parameters
- Adstock transformation
- Diminishing returns transformation
- Conversion rate modeling with noise
- Visualization of baseline, spend, transformations, and conversions

---

## Parameters Required in Function:

### Parameters Required in `define_basic_parameters`

| Parameter               | Description                                                                                                                                                                                                                                                                                       | Default value if none provided |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| `years`                 | Number of years to generate data for. Must be a whole number ≥ 1.                                                                                                                                                                                                                                | `5`                            |
| `channels_impressions`  | List of strings for channels using impressions metric (e.g., Facebook, TV).                                                                                                                                                                                                                       | No default provided.           |
| `channels_clicks`       | List of strings for channels using clicks metric (e.g., Search).                                                                                                                                                                                                                                  | No default provided.           |
| `channels_sessions`     | List of strings for channels using sessions metric (e.g., Direct).                                                                                                                                                                                                                                | No default provided.           |
| `frequency_of_campaigns`| Frequency of campaigns (e.g., 2 = every 2 days). Must be whole number ≥ 1.                                                                                                                                                                                                                        | `1`                            |
| `true_cvr`              | List of conversion rates (0–1) in order: impressions channels, clicks channels, sessions channels. Length must match total channels.                                                                                                                                                             | No default provided.           |
| `revenue_per_conv`      | Revenue per conversion. Must be > 0.                                                                                                                                                                                                                                                               | No default provided.           |
| `start_date`            | Dataset start date (`yyyy/mm/dd` or `yyyy-mm-dd`).                                                                                                                                                                                                                                                 | `"2020/1/1"`                   |

---

### Parameters Required in `create_baseline`

| Parameter         | Description                                                                                                                                                                | Default value if none provided |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| `setup_variables` | Dictionary output from `define_basic_parameters`.                                                                                                                          | Required                       |
| `base_p`          | Baseline daily sales without ads.                                                                                                                                          | Required                       |
| `trend_p`         | Growth over the data period.                                                                                                                                               | `2`                            |
| `temp_var`        | Amplitude of temperature variation for seasonality.                                                                                                                        | `8`                            |
| `temp_coef_mean`  | Mean importance of seasonality on sales.                                                                                                                                    | `50000`                        |
| `temp_coef_sd`    | Standard deviation of seasonality effect importance.                                                                                                                        | `5000`                         |
| `error_std`       | Noise in baseline sales.                                                                                                                                                    | `100`                          |

---

### Parameters Required in `create_ads_spend`

| Parameter              | Description                                                                                                                                                                     | Default value if none provided |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| `setup_variables`      | Dictionary output from `define_basic_parameters`.                                                                                                                               | Required                       |
| `campaign_spend_mean`  | Mean spend per campaign.                                                                                                                                                        | Required                       |
| `campaign_spend_std`   | Std deviation of spend per campaign.                                                                                                                                            | Required                       |
| `min_max_proportion`   | Dictionary mapping channel → `(min_prop, max_prop)`. Last channel gets remainder. Proportions must be between 0 and 1.                                                          | Required                       |

---

### Parameters Required in `generate_media`

| Parameter          | Description                                                                                                                                                                  | Default value if none provided |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| `setup_variables`  | Dictionary from `define_basic_parameters`.                                                                                                                                   | Required                       |
| `df_ads_spends`    | DataFrame from `create_ads_spend`.                                                                                                                                            | Required                       |
| `true_cpm`         | Dict mapping channel → CPM for impression channels.                                                                                                                          | Required                       |
| `true_cpc`         | Dict mapping channel → CPC for click/session channels.                                                                                                                        | Required                       |
| `mean_noisy`       | Dict mapping channel → mean noise in CPM/CPC.                                                                                                                                 | `0.0` for missing              |
| `std_noisy`        | Dict mapping channel → std deviation of noise.                                                                                                                                | `0.0` for missing              |
| `sessions_config`  | Dict of session-channel configs: base, trend, weekly_strength, annual_strength, noise_cv, etc.                                                                                | Optional                       |

---

### Parameters Required in `generate_noisy_cvr`

| Parameter         | Description                                                                                                                    | Default value if none provided |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| `setup_variables` | Dictionary from `define_basic_parameters`.                                                                                     | Required                       |
| `df_media`        | DataFrame from `generate_media`.                                                                                                | Required                       |
| `mean_noisy_cvr`  | Dict mapping channel → mean CVR noise.                                                                                          | `0.0` for missing              |
| `std_noisy_cvr`   | Dict mapping channel → std deviation CVR noise.                                                                                 | `0.0` for missing              |

---

### Parameters Required in `pivot_to_mmm_format`

| Parameter         | Description                                                                                          | Default value if none provided |
|-------------------|------------------------------------------------------------------------------------------------------|--------------------------------|
| `setup_variables` | Dictionary from `define_basic_parameters`.                                                           | Required                       |
| `df_ads`          | DataFrame with campaign/channel data from `generate_noisy_cvr`.                                      | Required                       |

---

### Parameters Required in `apply_adstock`

| Parameter           | Description                                                                                             | Default value if none provided |
|---------------------|---------------------------------------------------------------------------------------------------------|--------------------------------|
| `setup_variables`   | Dictionary from `define_basic_parameters`.                                                              | Required                       |
| `df_daily`          | Daily-format DataFrame from `pivot_to_mmm_format`.                                                       | Required                       |
| `true_lambda_decay` | Dict mapping channel → decay rate (0–1).                                                                 | Required                       |

---

### Parameters Required in `apply_diminishing_returns`

| Parameter          | Description                                                                                          | Default value if none provided |
|--------------------|------------------------------------------------------------------------------------------------------|--------------------------------|
| `setup_variables`  | Dictionary from `define_basic_parameters`.                                                           | Required                       |
| `df_adstock`       | DataFrame from `apply_adstock`.                                                                       | Required                       |
| `alpha_saturation` | Float or dict mapping channel → alpha (> 0).                                                          | Required                       |
| `gamma_saturation` | Float or dict mapping channel → gamma (0–1].                                                          | Required                       |
| `x_marginal`       | Optional constant value for S-curve factor.                                                           | Optional                       |

---

### Parameters Required in `calculate_conversions`

| Parameter         | Description                                                                                          | Default value if none provided |
|-------------------|------------------------------------------------------------------------------------------------------|--------------------------------|
| `setup_variables` | Dictionary from `define_basic_parameters`.                                                           | Required                       |
| `df_daily`        | DataFrame after `apply_diminishing_returns`.                                                          | Required                       |

---

### Parameters Required in `generate_final_df`

| Parameter            | Description                                                                                                                                 | Default value if none provided |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| `setup_variables`    | Dictionary from `define_basic_parameters`.                                                                                                  | Required                       |
| `df_daily`           | DataFrame from `calculate_conversions`.                                                                                                     | Required                       |
| `df_baseline`        | DataFrame from `create_baseline`.                                                                                                           | Required                       |
| `self_claim_config`  | Dict mapping channel → proportion adjustment for self-reported metrics (e.g., 0.1 = +10%, -0.05 = -5%).                                    | Optional                       |
| `daily_noise_std`    | Std deviation for daily noise in self-reported metrics (proportion).                                                                        | `0.01`                         |
| `seed`               | Random seed for reproducibility.                                                                                                            | Optional                       |





---

## References

- **Step-by-Step Guide**  
  The simulation process in this project is inspired by the official siMMMulator package, which provides a detailed workflow for generating marketing mix modeling (MMM) data in R.  
  [Read the guide here](https://facebookexperimental.github.io/siMMMulator/)

- **Acknowledgment**  
  This Python implementation is translated from the original R package **siMMMulator**, with modifications to incorporate session-level modeling and conversion tracking. While the core structure and logic remain consistent with the R version, implementation details have been adapted to Python syntax, libraries, and workflow.

---

## Contributing
Contributions are welcome!  
1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/your-feature`)  
3. Commit changes (`git commit -m 'Add feature'`)  
4. Push to branch (`git push origin feature/your-feature`)  
5. Open a Pull Request

---

<!-- MARKDOWN LINKS & IMAGES -->
[Python.py]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/

[jupyterlab.ipynb]: https://shields.io/badge/JupyterLab-Try%20GraphScope%20Now!-F37626?logo=jupyter
[jupyterlab-url]: https://justinbois.github.io/bootcamp/2020_fsri/lessons/l01_welcome.html
