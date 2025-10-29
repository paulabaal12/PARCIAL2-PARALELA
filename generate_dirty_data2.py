import pandas as pd
import numpy as np
import json

def generate_dirty_dataset(n_rows=10_000_000, output_file='dirty_data.csv'):
    np.random.seed(16)
    
    print(f"Generando dataset con {n_rows:,} filas...")
    
    # Generar datos base
    df = pd.DataFrame({
        'id': range(n_rows),
        'name': np.random.choice(['Juan Perez', 'MARIA LOPEZ', 'pedro gomez', 'Ana Silva', '   Carlos Ruiz   ', 'Luis García'], n_rows),
        'age': np.random.choice([np.nan, *range(18, 80)], n_rows, 
                                p=[0.15] + [0.85/62]*62),  # 15% datos faltantes
        'email': np.random.choice(['juan@gmail.com', 'MARIA@YAHOO.COM', 'pedro@', 
                                   'ana@hotmail.com', 'invalido-email'], n_rows),
        'country': np.random.choice(['Guatemala', 'Gutemala', 'GT', 'guatemala', 
                                     'USA', 'US', 'Gringolandia', 'Mexico', 'Mejico'], n_rows),
        'salary': np.random.normal(50000, 20000, n_rows)
    })
    
    # Añadir duplicados (10%)
    n_duplicates = int(n_rows * 0.10)
    dup_indices = np.random.choice(n_rows, size=n_duplicates, replace=False)
    duplicates = df.iloc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Añadir outliers en salary (1%)
    outlier_indices = np.random.choice(len(df), size=int(len(df)*0.01), replace=False)
    df.loc[outlier_indices, 'salary'] = np.random.choice([0, -5000, 5000000], len(outlier_indices))
    
    # Guardar
    df.to_csv(output_file, index=False)
    
    print(f"  Generated dataset: {output_file}")
    print(f"  Total rows: {len(df):,}")
    
    return output_file

if __name__ == '__main__':
    # Generar el JSON de metadatos
    metadata = {
        "cleaning_config": {
            "name": {"type": "string_normalize", "operation": ["lower", "strip"]},
            "age": {"type": "missing_impute", "strategy": "median"},
            "email": {"type": "string_transform", "operation": ["lower"], "validation": True},
            "country": {"type": "dictionary_replace", "dictionary_name": "country_mapping"},
            "salary": {"type": "outlier_capping", "method": "iqr_fence", "cap_value": 1.5}
        },
        "dictionaries": {
            "country_mapping": {
                "Guatemala": ["Gutemala", "GT", "guate", "miwate", "guatemala"],
                "Estados Unidos": ["USA", "US", "Estados", "Gringolandia"],
                "Mexico": ["MX", "Mejico"]
            }
        }
    }
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    generate_dirty_dataset(n_rows=10_000_000, output_file='dirty_data.csv') # Aumentado a 10M