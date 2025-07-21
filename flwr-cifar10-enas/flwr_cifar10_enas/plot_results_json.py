import json
import matplotlib.pyplot as plt
import os
import numpy as np
# Ruta al archivo
results_path = "/workspace/flwr-cifar10-enas/flwr_cifar10_enas/results.json"
output_dir = "/workspace/flwr-cifar10-enas/flwr_cifar10_enas/plots"
os.makedirs(output_dir, exist_ok=True)

# Cargar los resultados
with open(results_path, "r") as f:
    results = json.load(f)

# Inicializar diccionarios para cada métrica
rounds = []
val_acc, test_acc = [], []

# Extraer datos
for rnd_str in sorted(results.keys(), key=lambda x: int(x)):
    rnd = int(rnd_str)
    data = results[rnd_str]

    rounds.append(rnd)

    val_acc.append(data["child_valid_acc"])
    test_acc.append(data["child_test_acc"])

# Plot: Accuracy
plt.figure()
# plt.plot(rounds, train_acc, label="Train Accuracy")
plt.plot(rounds, val_acc, label="Validation Accuracy")
plt.plot(rounds, test_acc, label="Test Accuracy")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Global Evaluation (Accuracy)")
plt.ylim(0,1)
plt.xticks(np.arange(1, 6, 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/accuracy_over_rounds.png")
plt.close()

print(f"Gráficos guardados en: {output_dir}/")