import pandas as pd
import numpy as np

def generate_dirty_dataset(n_rows=15_000, output_file='dirty_data.csv'):
    np.random.seed(16)
    
    print(f"Generando dataset con {n_rows:,} filas...")
    
    # Generar datos base
    df = pd.DataFrame({
        'id': range(n_rows),
        'name': np.random.choice(['Juan Perez', 'MARIA LOPEZ', 'pedro gomez', 
                                'Ana Silva', '  Carlos Ruiz  '], n_rows),
        'age': np.random.choice([np.nan, *range(18, 80)], n_rows, 
                                p=[0.15] + [0.85/62]*62),  # 15% datos faltantes
        'email': np.random.choice(['juan@gmail.com', 'MARIA@YAHOO.COM', 
                                'pedro@', 'ana@hotmail.com'], n_rows),
        'country': np.random.choice(['Guatemala', 'Gutemala', 'GT', 
                                    'Guatemala', 'guatemala'], n_rows),
        'salary': np.random.normal(50000, 20000, n_rows)
    })
    
    # Añadir duplicados (10%)
    n_duplicates = int(n_rows * 0.10)
    dup_indices = np.random.choice(n_rows, size=n_duplicates, replace=False)
    duplicates = df.iloc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Añadir outliers en salary
    outlier_indices = np.random.choice(len(df), size=int(len(df)*0.01), replace=False)
    df.loc[outlier_indices, 'salary'] = np.random.choice([0, -5000, 5000000], len(outlier_indices))
    
    # Guardar
    df.to_csv(output_file, index=False)
    
    print(f"   Generated dataset: {output_file}")
    print(f"   Total rows: {len(df):,}")
    print(f"   Missing values: {df['age'].isna().sum():,} ({100*df['age'].isna().sum()/len(df):.1f}%)")
    print(f"   Duplicates added: {n_duplicates:,}")
    
    return output_file

if __name__ == '__main__':
    generate_dirty_dataset(n_rows=15_000, output_file='dirty_data.csv')