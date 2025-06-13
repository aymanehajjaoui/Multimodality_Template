import os
import re

# Dossier à modifier
model2_path = "model2"

# Liste des noms de variables globales à renommer (à adapter selon ton export)
variables_to_rename = [
    "conv1d_kernel", "conv1d_bias",
    "conv1d_1_kernel", "conv1d_1_bias",
    "conv1d_2_kernel", "conv1d_2_bias",
    "conv1d_3_kernel", "conv1d_3_bias",
    "dense_kernel", "dense_bias",
    "dense_1_kernel", "dense_1_bias",
    "batch_normalization_kernel", "batch_normalization_bias",
    "batch_normalization_1_kernel", "batch_normalization_1_bias",
    "batch_normalization_2_kernel", "batch_normalization_2_bias",
    "batch_normalization_3_kernel", "batch_normalization_3_bias"
]

# Extensions ciblées
target_extensions = [".c", ".h"]

# Appliquer la modification
for root, _, files in os.walk(model2_path):
    for file in files:
        if any(file.endswith(ext) for ext in target_extensions):
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                content = f.read()

            # Remplacer chaque variable par sa version suffixée
            for var in variables_to_rename:
                content = re.sub(rf'\b{var}\b', f"{var}2", content)

            with open(filepath, 'w') as f:
                f.write(content)

print("✅ Renommage terminé dans model2/")
