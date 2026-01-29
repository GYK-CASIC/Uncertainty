```markdown
# Uncertainty Research Project

This repository contains the implementation and datasets for evaluating **Uncertainty** and **Uncertainty++** methodologies.

---

## 📂 Project Structure

```text
.
├── datasets/
│   ├── main/             # Main experimental data
│   ├── robustness/       # Robustness evaluation sets
│   └── generalization/   # Generalization testing sets
├── scripts/
│   ├── main.py           # Script to run Uncertainty
│   ├── main++.py         # Script to run Uncertainty++
│   └── hyper_analysis.py # Hyperparameter analysis tool
├── Proxy_LLMs/           # Directory for Proxy Models (User created)
├── requirements.txt      # Python dependencies
└── README.md

```

---

## 🛠 Preparation

### 1. Environment Setup

Install the required packages using the following command:

```bash
pip install -r requirements.txt

```

### 2. Model Download

**Important:** Before running the code, you must download the proxy models and place them in the following path:
`./Proxy_LLMs/`

---

## 🚀 Running the Experiments

All execution scripts are located in the `scripts` directory. Please follow this order:

### 1. Navigate to the scripts folder

```bash
cd scripts

```

### 2. Run Uncertainty

To execute the standard Uncertainty evaluation:

```bash
python main.py

```

### 3. Run Uncertainty++

To execute the enhanced Uncertainty++ evaluation:

```bash
python main++.py

```

---

## 📚 Acknowledgements

We utilize datasets from the **Lastde_Detector** repository. Please refer to the original source for more details:
[https://github.com/TrustMedia-zju/Lastde_Detector](https://github.com/TrustMedia-zju/Lastde_Detector)

## 📄 License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

```
