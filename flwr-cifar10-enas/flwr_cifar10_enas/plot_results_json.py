import json
import matplotlib.pyplot as plt
import os
import numpy as np
# Ruta al archivo
results_path = "/workspace/flwr-cifar10-enas/flwr_cifar10_enas/results.json"
output_dir = "/workspace/flwr-cifar10-enas/flwr_cifar10_enas/plots"
os.makedirs(output_dir, exist_ok=True)
""" This code has been developed by Tamai Ramírez Gordillo (GitHub: TamaiRamirezUA)"""
# Cargar los resultados
with open(results_path, "r") as f:
    results = json.load(f)

# Inicializar diccionarios para cada métrica
rounds = []
val_acc, test_acc = [], []
val_loss, test_loss = [], []
# Extraer datos
for rnd_str in sorted(results.keys(), key=lambda x: int(x)):
    rnd = int(rnd_str)
    data = results[rnd_str]

    rounds.append(rnd)

    val_acc.append(data["child_valid_acc"])
    test_acc.append(data["child_test_acc"])
    val_loss.append(data["child_valid_loss"])
    test_loss.append(data["child_test_loss"])

# Plot: Accuracy
plt.figure()
# plt.plot(rounds, train_acc, label="Train Accuracy")
plt.plot(rounds, val_acc, label="Validation Accuracy")
plt.plot(rounds, test_acc, label="Test Accuracy")
plt.plot(rounds, val_loss, label="Validation Loss")
plt.plot(rounds, test_loss, label="Test Loss")
# Línea horizontal punteada en y=1
plt.axhline(y=1, color='r', linestyle='--', linewidth=1.5)
plt.xlabel("Round")
plt.ylabel("Value")
plt.title("Global Evaluation Metrics")
plt.xticks(np.arange(1, 6, 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/global_metrics_over_rounds.png")
plt.close()

print(f"Gráficos guardados en: {output_dir}/")