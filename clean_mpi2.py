from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import sys
import json
import re # Para validación de email

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Función de utilidad para invertir el diccionario de limpieza
def create_reverse_mapping(dictionary):
    reverse_map = {}
    for key, values in dictionary.items():
        for value in values:
            reverse_map[value] = key
    return reverse_map

# ==============================================================================
# LÓGICA DE LIMPIEZA CENTRAL (APLICADA POR CADA WORKER)
# ==============================================================================

def apply_cleaning_rules(df_chunk, config, dictionaries, median_age, salary_bounds):
    """Aplica las reglas de limpieza definidas en el JSON a un chunk."""
    
    # 1. Transformaciones de Cadenas y Reemplazo de Diccionario
    for col, conf in config.items():
        if col in df_chunk.columns:
            if conf['type'] == 'string_normalize':
                for op in conf['operation']:
                    if op == 'lower':
                        df_chunk[col] = df_chunk[col].str.lower()
                    elif op == 'strip':
                        df_chunk[col] = df_chunk[col].str.strip()
                        
            elif conf['type'] == 'dictionary_replace':
                dict_name = conf['dictionary_name']
                reverse_map = create_reverse_mapping(dictionaries[dict_name])
                df_chunk[col] = df_chunk[col].replace(reverse_map)
                
            elif conf['type'] == 'string_transform' and conf.get('validation'):
                # Ejemplo de limpieza intensiva: validar y limpiar emails no válidos
                email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                df_chunk[col] = df_chunk[col].str.lower()
                df_chunk.loc[~df_chunk[col].apply(lambda x: bool(re.match(email_regex, str(x)))), col] = np.nan # Reemplazar inválidos con NaN
                
    # 2. Imputación (usa el valor global calculado por el root)
    if 'age' in config and config['age']['type'] == 'missing_impute' and median_age is not None:
        df_chunk['age'].fillna(median_age, inplace=True)
        
    # 3. Capping de Outliers (usa los límites globales calculados)
    if 'salary' in config and config['salary']['type'] == 'outlier_capping' and salary_bounds is not None:
        lower, upper = salary_bounds
        df_chunk.loc[df_chunk['salary'] < lower, 'salary'] = lower
        df_chunk.loc[df_chunk['salary'] > upper, 'salary'] = upper
        
    return df_chunk

# ==============================================================================
# FUNCIÓN PARALELA (MPI)
# ==============================================================================

def clean_parallel(input_file, metadata_file='metadata.json'):
    
    if rank == 0:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        config = metadata['cleaning_config']
        dictionaries = metadata['dictionaries']
        
        print("="*60)
        print(f"CLEANSTREAM (Parallel with {size} workers) - File: {input_file}")
        print("="*60)
        start_time = time.time()
        
        # Load and Partition
        print(" Loading and distributing data...")
        df = pd.read_csv(input_file)
        
        # Particionar
        chunk_size = len(df) // size
        chunks = []
        for i in range(size):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < size - 1 else len(df)
            chunks.append(df.iloc[start_idx:end_idx].copy())
        
        print(f"  Divided in {size} chunks of ~{chunk_size:,} rows")
    else:
        config, dictionaries, chunks = None, None, None
    
    # Broadcast config y Scatter chunks
    config = comm.bcast(config, root=0)
    dictionaries = comm.bcast(dictionaries, root=0)
    my_chunk = comm.scatter(chunks, root=0)

    # ==================================================
    # FASE 1: CÁLCULO DE ESTADÍSTICAS GLOBALES
    # ==================================================
    
    # 1. Mediana Global (para Imputación de 'age')
    local_sum = my_chunk['age'].sum()
    local_count = my_chunk['age'].count()
    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
    global_count = comm.reduce(local_count, op=MPI.SUM, root=0)
    
    median_age = global_sum / global_count if rank == 0 and global_count > 0 else None
    median_age = comm.bcast(median_age, root=0)
    
    # 2. Límites de Outliers Globales (para 'salary' usando IQR)
    local_values = my_chunk['salary'].dropna().values
    all_values = comm.gather(local_values, root=0)
    
    salary_lower, salary_upper = None, None
    if rank == 0:
        all_salary_values = np.concatenate(all_values)
        Q1 = np.percentile(all_salary_values, 25)
        Q3 = np.percentile(all_salary_values, 75)
        IQR = Q3 - Q1
        salary_lower = Q1 - config['salary']['cap_value'] * IQR
        salary_upper = Q3 + config['salary']['cap_value'] * IQR
        
    salary_lower = comm.bcast(salary_lower, root=0)
    salary_upper = comm.bcast(salary_upper, root=0)
    salary_bounds = (salary_lower, salary_upper)

    # 3. Detección Global de Duplicados (es la parte más lenta de MPI, pero necesaria para la prueba)
    local_hashes = {}
    for idx, row in my_chunk.iterrows():
        # Excluimos 'id' para la detección de duplicados
        row_hash = hash(tuple(row.drop('id', errors='ignore'))) 
        if row_hash not in local_hashes:
            local_hashes[row_hash] = []
        local_hashes[row_hash].append(idx)
    
    all_hashes = comm.gather(local_hashes, root=0)
    
    duplicate_indices = []
    if rank == 0:
        global_hashes = {}
        for worker_hashes in all_hashes:
            for h, indices in worker_hashes.items():
                global_hashes.setdefault(h, []).extend(indices)
        
        for h, indices in global_hashes.items():
            if len(indices) > 1:
                # Marcar los índices duplicados (todos menos el primero)
                duplicate_indices.extend(indices[1:])
                
    duplicate_indices = comm.bcast(duplicate_indices, root=0)
    
    # ==================================================
    # FASE 2: APLICACIÓN DE REGLAS Y CONSOLIDACIÓN
    # ==================================================
    
    # Eliminar duplicados locales
    my_chunk = my_chunk[~my_chunk.index.isin(duplicate_indices)]
    
    # Aplicar todas las reglas de limpieza
    my_chunk = apply_cleaning_rules(my_chunk, config, dictionaries, median_age, salary_bounds)
    
    # Recolectar chunks limpios
    clean_chunks = comm.gather(my_chunk, root=0)
    
    if rank == 0:
        final_df = pd.concat(clean_chunks, ignore_index=True)
        final_df.to_csv('clean_parallel_output.csv', index=False)
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print(f" PARALLEL COMPLETED IN {elapsed:.2f} SECONDS ({size} workers)")
        print(f" Final rows: {len(final_df):,}")
        print("="*60)
        return elapsed
    return None

# ==============================================================================
# FUNCIÓN SECUENCIAL (Para Comparación)
# ==============================================================================

def clean_sequential(input_file, metadata_file='metadata.json'):
    
    if rank != 0:
        return None
        
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    config = metadata['cleaning_config']
    dictionaries = metadata['dictionaries']
    
    print("="*60)
    print("CLEANSTREAM (Sequential Run) - File: dirty_data.csv")
    print("="*60)
    start_time = time.time()
    
    df = pd.read_csv(input_file)
    original_rows = len(df)
    
    # 1. CÁLCULO DE ESTADÍSTICAS
    median_age = df['age'].median()
    
    Q1 = df['salary'].quantile(0.25)
    Q3 = df['salary'].quantile(0.75)
    IQR = Q3 - Q1
    # Usamos el cap_value del JSON
    salary_lower = Q1 - config['salary']['cap_value'] * IQR 
    salary_upper = Q3 + config['salary']['cap_value'] * IQR
    salary_bounds = (salary_lower, salary_upper)
    
    # 2. ELIMINACIÓN DE DUPLICADOS
    # Es mucho más simple y rápido en Pandas secuencial
    df.drop_duplicates(inplace=True, ignore_index=True)
    
    # 3. APLICACIÓN DE REGLAS (Similar a la versión paralela pero sin 'chunk')
    df = apply_cleaning_rules(df, config, dictionaries, median_age, salary_bounds)

    df.to_csv('clean_sequential_output.csv', index=False)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print(f" SEQUENTIAL COMPLETED IN {elapsed:.2f} SECONDS")
    print(f" Final rows: {len(df):,}")
    print("="*60)
    return elapsed
    
# ==============================================================================
# EJECUCIÓN Y REPORTE
# ==============================================================================

# Archivo para almacenar el tiempo secuencial
TIME_FILE = 'sequential_time.txt'

if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'dirty_data.csv'
    
    time_par = None
    time_seq = None
    
    if size == 1:
        # --- MODO 1: Ejecución Secuencial (Establece la base T_s) ---
        # Ejecutado con: mpirun -n 1 python clean_mpi.py dirty_data.csv
        
        # Solo el rank 0 hace la limpieza
        time_seq = clean_sequential(input_file)
        
        # Guarda el tiempo secuencial para futuras comparaciones
        if rank == 0 and time_seq is not None:
            try:
                with open(TIME_FILE, 'w') as f:
                    f.write(str(time_seq))
                print(f"\n[INFO] Tiempo Secuencial guardado en {TIME_FILE}.")
            except Exception as e:
                print(f"[ERROR] No se pudo guardar el tiempo secuencial: {e}")
            
    else:
        # --- MODO 2: Ejecución Paralela (Calcula T_p y el reporte) ---
        # Ejecutado con: mpirun -n N python clean_mpi.py dirty_data.csv (donde N > 1)
        
        time_par = clean_parallel(input_file)
        
        if rank == 0 and time_par is not None:
            
            # Intenta leer el tiempo secuencial guardado
            try:
                with open(TIME_FILE, 'r') as f:
                    time_seq_str = f.read().strip()
                    T_s = float(time_seq_str)
            except FileNotFoundError:
                print("\n[ADVERTENCIA] No se encontró 'sequential_time.txt'. Ejecute primero con 'mpirun -n 1' para establecer el tiempo base.")
                T_s = None
            except Exception as e:
                print(f"\n[ERROR] Error al leer el tiempo secuencial: {e}")
                T_s = None
            
            # Genera el reporte si T_s está disponible
            if T_s is not None:
                T_p = time_par
                speedup = T_s / T_p
                efficiency = speedup / size
                
                print("\n" + "="*60)
                print(" 📊 REPORTE DE EFICIENCIA DEL PARALELISMO")
                print("="*60)
                print(f" Tiempo Secuencial (T_s, n=1): {T_s:.2f}s")
                print(f" Tiempo Paralelo (T_p, n={size}): {T_p:.2f}s")
                print(f" Ganancia (Speedup): {speedup:.2f}x")
                print(f" Eficiencia (S/N): {efficiency:.2f}")
                print("="*60)
            # print("="*60)