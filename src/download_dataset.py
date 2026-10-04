"""Download Fashion MNIST into the project input folder and a SQLite database.

Loads the Kaggle credentials from the project .env file, downloads the
dataset with kagglehub, copies the CSV files to the input folder and loads
them into a SQLite database in the sql folder.
Run it from a local terminal, not from a Colab kernel.
"""

import os
import platform
import shutil
import sqlite3
import sys

import kagglehub
import pandas as pd
from dotenv import load_dotenv

# stop if running on a remote Linux kernel / parar se estiver em kernel Linux remoto
if platform.system() != "Windows":
    sys.exit("Run this script in a local Windows terminal / Rode este script em um terminal local do Windows")

# project folders, raw strings / pastas do projeto, strings raw
project_dir = r"C:\Users\rafae.RAFAEL_NOTEBOOK\OneDrive\Documentos\Projetos_data_science_IA\github\Convolutional-Neural-Network-Transfer-Learning"
env_path = os.path.join(project_dir, ".env")
input_dir = os.path.join(project_dir, "input")
sql_dir = os.path.join(project_dir, "sql")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(sql_dir, exist_ok=True)

# stop if the .env file is missing / parar se o arquivo .env não existir
if not os.path.exists(env_path):
    sys.exit(f".env not found / .env não encontrado: {env_path}")

# load the credentials / carregar as credenciais
load_dotenv(env_path, override=True)

# confirm without printing the secret / confirmar sem mostrar o segredo
print("KAGGLE_USERNAME set:", "KAGGLE_USERNAME" in os.environ)
print("KAGGLE_KEY set:", "KAGGLE_KEY" in os.environ)
print("KAGGLE_API_TOKEN set:", "KAGGLE_API_TOKEN" in os.environ)

# download the dataset to the kagglehub cache / baixar o dataset para o cache do kagglehub
dataset_path = kagglehub.dataset_download("zalando-research/fashionmnist")
print("Cache path / Caminho do cache:", dataset_path)

# copy only the csv files to the input folder / copiar apenas os csv para a pasta input
for file_name in ["fashion-mnist_train.csv", "fashion-mnist_test.csv"]:
    shutil.copy(os.path.join(dataset_path, file_name), os.path.join(input_dir, file_name))
    print("Copied / Copiado:", file_name)

# list the destination folder with sizes / listar a pasta de destino com tamanhos
for name in sorted(os.listdir(input_dir)):
    print(f"{name}: {os.path.getsize(os.path.join(input_dir, name)) / 1e6:.1f} MB")

# sqlite database path / caminho do banco sqlite
db_path = os.path.join(sql_dir, "fashion_mnist.db")

# csv file and table name pairs / pares de arquivo csv e nome da tabela
tables = {
    "fashion-mnist_train.csv": "fashion_mnist_train",
    "fashion-mnist_test.csv": "fashion_mnist_test",
}

# open the database connection / abrir a conexão com o banco
conn = sqlite3.connect(db_path)

for file_name, table_name in tables.items():
    csv_path = os.path.join(input_dir, file_name)

    # first chunk replaces the table, the others append / primeiro bloco recria a tabela, os demais acrescentam
    write_mode = "replace"
    n_rows = 0

    # read the csv in chunks to keep memory low / ler o csv em blocos para economizar memória
    for chunk in pd.read_csv(csv_path, chunksize=5000):
        chunk.index = range(n_rows, n_rows + len(chunk))
        chunk.to_sql(table_name, conn, if_exists=write_mode, index=True, index_label="id")
        write_mode = "append"
        n_rows += len(chunk)

    # index on the label column for faster queries / índice na coluna label para consultas mais rápidas
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_label ON {table_name} (label)")
    conn.commit()
    print(f"Table / Tabela {table_name}: {n_rows} rows / linhas")

# check the rows per class / conferir as linhas por classe
for table_name in tables.values():
    rows = conn.execute(f"SELECT label, COUNT(*) FROM {table_name} GROUP BY label ORDER BY label").fetchall()
    print(table_name, rows)

# close the connection / fechar a conexão
conn.close()
print(f"Database size / Tamanho do banco: {os.path.getsize(db_path) / 1e6:.1f} MB")