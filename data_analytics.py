import pandas as pd

# Ruta del archivo parquet
ruta_archivo = "/workspace/flwr-enas-nofix/experiments/20260202_1/results/global_metrics.parquet"
ruta_archivo2 = "/workspace/flwr-enas-nofix/experiments/20260203_1/results/global_metrics.parquet"
# Leer archivo parquet
df = pd.read_parquet(ruta_archivo)
df2 = pd.read_parquet(ruta_archivo2)


# Opcional: mostrar solo las primeras filas
print("\nUltimas 5 filas:")
print(df.tail())
print("*"*80)
print(df2.tail())

