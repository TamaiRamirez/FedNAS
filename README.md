# Adaptive Federated Neural Architecture Search for Heterogeneous Environments

This repository contains the code developed for the doctoral thesis:

**📄 Adaptive Federated Neural Architecture Search Model for Heterogeneous Environments**  
**👨‍🔬 Author:** Tamai Ramírez Gordillo  
**🎓 Degree:** Doctorado en Informática — Mención de Doctor Internacional  
**🏛 Institution:** Departamento de Tecnología Informática y Computación, Escuela Politécnica Superior, Universidad de Alicante  
**👨‍🏫 Supervisors:** Prof. Higinio Mora Mora, Ph.D. and Prof. Francisco A. Pujol López, Ph.D.  
**📅 Thesis:** 2026  

---

## 🧠 Overview

This thesis propose an adaptive framework for **Federated Neural Architecture Search (FNAS)** in heterogeneous environments. It combines **Federated Learning (FL)** with **Efficient Neural Architecture Search (ENAS)** so that distributed clients can explore neural architectures locally while collaborating through a shared federated optimization process.

The framework is designed for scenarios where data cannot be centralized due to privacy, regulatory, operational, or edge-computing constraints. It explicitly addresses **statistical heterogeneity**, **client variability**, and **architecture-level diversity** by introducing a new aggregation strategy named **Federated Fairness–Performance Aggregation (FedFPA)** and a post-training evaluation metric named **Fairness–Performance Index (FPI)**.

### Key Contributions

- ✅ Client-side ENAS for distributed architecture exploration.
- 🔁 **FedFPA**, a heterogeneity-aware aggregation method that combines FedProx-style local regularization, sample-size awareness, and corrected client improvement.
- 🧩 Block-wise aggregation over a shared ENAS supernet, allowing clients to activate different architectural blocks while aggregating only compatible parameters.
- 📊 **FPI**, an intrinsic post-training metric that jointly evaluates mean macro F1-score, predictive loss, and inter-client fairness.
- 🔒 Privacy-preserving federated setup: clients train on local private data and only exchange model updates or compatible supernet blocks.
- 🧪 Validation on **CIFAR-10**, **CIFAR-100**, and **MNIST** under IID and Non-IID Dirichlet partitions.

---

## 🏗️ Methodology

The framework follows a federated client-server workflow:

1. The server initializes a global ENAS supernet.
2. Each client samples and trains local child architectures using private data.
3. Clients evaluate local progress and estimate the discrepancy between local and global class distributions.
4. Clients send updated active supernet blocks, activation masks, and corrected improvement signals to the server.
5. The server aggregates compatible layer-operation blocks using FedFPA weights.
6. The updated global model is redistributed until convergence or early stopping.

## ⚙️ Installation & Usage

### Prerequisites

- Python 3.10+
- Docker with NVIDIA GPU support
- CUDA-compatible NVIDIA GPU
- TensorFlow
- Flower
- NumPy, SciPy, scikit-learn, pandas, matplotlib

> Exact package versions may depend on the experiment branch or container image used in this repository.

### Installation Steps
Terminal 1:
```bash
# Clone the repository
git https://github.com/TamaiRamirezUA/FedNAS.git
cd FedNAS

# Build the Docker image
docker build --build-arg USERNAME=$(whoami)-docker --build-arg USER_UID=$(id -u) -t flenas:latest .

bash run.sh all flenas flenas:latest
# Include the following line in .bashrc file:
export PATH="$HOME/.local/bin:$PATH"
```

Terminal 2:
```bash
docker commit <flenas container_ID> flenas:latest
```

### Running Federated Experiments

Terminal 1:
```bash
cd FedNAS
# Run the container with GPU support
# Modify run.sh according to your image name, volume paths, and GPU resources
bash run.sh all flenas flenas:latest
flower-superlink --insecure
```

Terminal 2-N:
```bash
# Run the following for every flower node you want to launch
# where i starts in 0 to N-1.

flower-supernode \
    --insecure \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:$((9094 + i)) \
    --node-config "partition-id=i num-partitions=N"
```

Terminal N+1:
```bash
./run_experiments.sh
```

### Configuring Experiments

Before running the experiments, update the configuration files to match the setup you want to evaluate.

#### 1. Update `flwr-enas-nofix/pyproject.toml`

This file defines the main experiment settings, such as the number of federated learning rounds, the number of clients, local training epochs, and the dataset.

Edit the following values:

```toml
num-server-rounds = <number_of_rounds>
num-nodes = <number_of_clients>
local-epochs = <number_of_local_epochs>

dataset = "<huggingface_dataset_id>"
dataset_name = "<dataset_name>"
num-classes = <number_of_classes>
```

Use one of the following dataset configurations:

| Dataset | `dataset` | `dataset_name` | `num-classes` |
|---|---|---:|---:|
| MNIST | `ylecun/mnist` | `mnist` | `10` |
| CIFAR-10 | `uoft-cs/cifar10` | `cifar10` | `10` |
| CIFAR-100 | `uoft-cs/cifar100` | `cifar100` | `100` |
| FEMNIST | `flwrlabs/femnist` | `femnist` | `62` |

#### 2. Update `run_experiments.sh`

This file controls which experiment variants are executed. Each list defines the values used across multiple runs.

```bash
NON_IID_LIST=(false true true true)
ALPHA_LIST=(0.0 1.0 0.5 0.1)
AGG_LIST=("fedavg" "fedprox" "fedfpa" "fedavg")
MU_LIST=(0.0 0.01 0.1 1.0)
BETA_LIST=(0.5 0.5 0.5 0.5)
```

Each position across the lists represents one experiment configuration. For example, the first values from each list are used together, the second values are used together, and so on.

Example:

```bash
NON_IID_LIST=(false true)
ALPHA_LIST=(0.0 0.5)
AGG_LIST=("fedavg" "fedprox")
MU_LIST=(0.0 0.01)
BETA_LIST=(0.5 0.5)
```

This runs two configurations:

1. IID data using FedAvg with `alpha = 0.0`, `mu = 0.0`, and `beta = 0.5`
2. Non-IID data using FedProx with `alpha = 0.5`, `mu = 0.01`, and `beta = 0.5`

##### Supported Options

| Category | Supported Values | Description |
|---|---|---|
| Strategies | `fedavg`<br>`fedprox`<br>`fedfpa` | Aggregation strategies available for the experiments. |
| Datasets | `mnist`<br>`cifar10`<br>`cifar100`<br>`femnist` | Datasets supported by the experiment pipeline. |
| Data Distributions | `NON_IID_LIST` <br> `ALPHA_LIST`| `NON_IID_LIST` is a list of Boolean values that manages for each experiment if the setting is `IID` when `true` or `Non-IID` when `false`. <br> For `Non-IID` experiments, `alpha` in `ALPHA_LIST` controls data heterogeneity. Smaller values create more heterogeneous client datasets, while larger values produce distributions closer to IID. |

---

## 🖼️ Framework Schematic

The thesis framework can be summarized as:

```text
Server initializes global ENAS supernet
        ↓
Clients sample child architectures locally
        ↓
Local training on private client data
        ↓
Client evaluation + corrected improvement estimation
        ↓
Upload active blocks + masks + progress information
        ↓
FedFPA block-wise aggregation at server
        ↓
Updated global supernet redistributed
```

Example child architecture path:

```text
[Conv3x3] → [Conv5x5] → [DepthwiseSeparableConv] → [Skip/Connection] → ... → [Classifier]
```

---

## ✍️ How to Cite

If you use this repository, please cite the doctoral thesis:

```bibtex
@phdthesis{ramirez2026fedfpafnas,
  title     = {Adaptive Federated Neural Architecture Search Model for Heterogeneous Environments},
  author    = {Tamai Ramírez Gordillo},
  school    = {University of Alicante},
  department = {Department of Computer Technology and Computation},
  year      = {2026},
  note      = {Doctoral thesis, International Doctorate Mention}
}
```

Related publications developed during the doctoral period include work on Federated Neural Architecture Search, Distributed Neural Architecture Search, Distributed Machine Learning, and Federated Learning for Industry 5.0.

---

## 🤝 Acknowledgements

This thesis was funded by:

- 🇪🇸 Generalitat Valenciana, ACIF grant for predoctoral research staff hiring: **CIACIF/2022/176**, co-funded by the European Social Fund Plus.
- 🇪🇸 Generalitat Valenciana, BEFPI grant for predoctoral research stays outside the Comunitat Valenciana: **CIBEFP/2024/131**.

---

## 📬 Contact

For technical questions, research collaboration, or issues related to the thesis code:

📧 **Tamai Ramírez Gordillo: tamai.ramirez@ua.es** 
🏫 University of Alicante  
🌐 https://www.ua.es

---

## 📄 License

This work is licensed under a **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)**.

You are free to share this work, provided that you:

- Give appropriate credit to the original authors.
- Do not use the material for commercial purposes.
- Do not distribute modified versions of the material.

For more information, visit: https://creativecommons.org/licenses/by-nc-nd/4.0/