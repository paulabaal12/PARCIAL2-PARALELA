from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def clean (input_file):
    if rank == 0:
        print("="*60)
        print(f"CLEANSTREAM (Parallel with {size} workers)")
        print("="*60)
        start_time = time.time()
    
    if rank == 0:
        print(" Loading and distributing data...")
        load_start = time.time()
        df = pd.read_csv(input_file)
        print(f"  Original rows: {len(df):,}")
        
        # Particionar
        chunk_size = len(df) // size
        chunks = []
        for i in range(size):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < size - 1 else len(df)
            chunks.append(df.iloc[start_idx:end_idx].copy())
        
        print(f"   Divided in {size} chunks of ~{chunk_size:,} rows")
        print(f"   Distributed in {time.time()-load_start:.2f}s")
    else:
        chunks = None
    
    # Scatter chunks
    my_chunk = comm.scatter(chunks, root=0)
    
    if rank == 0:
        print(f"\n Analyzing in parallel ({size} workers)...")
    
    # Contar missing values
    local_missing = my_chunk['age'].isna().sum()
    total_missing = comm.reduce(local_missing, op=MPI.SUM, root=0)
    
    # Detectar duplicados locales (hash-based)
    local_hashes = {}
    for idx, row in my_chunk.iterrows():
        row_hash = hash(tuple(row))
        if row_hash not in local_hashes:
            local_hashes[row_hash] = []
        local_hashes[row_hash].append(idx)
    
    all_hashes = comm.gather(local_hashes, root=0)
    
    # Calcular mediana global
    local_sum = my_chunk['age'].sum()
    local_count = my_chunk['age'].count()
    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
    global_count = comm.reduce(local_count, op=MPI.SUM, root=0)
    
    # Calcular estadísticas para outliers
    local_values = my_chunk['salary'].dropna().values
    all_values = comm.gather(local_values, root=0)
    
    if rank == 0:
        # Consolidar duplicados
        global_hashes = {}
        for worker_hashes in all_hashes:
            for h, indices in worker_hashes.items():
                if h not in global_hashes:
                    global_hashes[h] = []
                global_hashes[h].extend(indices)
        
        duplicate_indices = set()
        for h, indices in global_hashes.items():
            if len(indices) > 1:
                duplicate_indices.update(indices[1:])
        
        duplicate_indices = list(duplicate_indices)
        
        # Calcular mediana y límites de outliers
        median_age = global_sum / global_count
        all_salary_values = np.concatenate(all_values)
        Q1 = np.percentile(all_salary_values, 25)
        Q3 = np.percentile(all_salary_values, 75)
        IQR = Q3 - Q1
        salary_lower = Q1 - 1.5 * IQR
        salary_upper = Q3 + 1.5 * IQR
        
        print(f"   Completed analysis")
        print(f"   Missing values: {total_missing:,}")
        print(f"   Duplicates: {len(duplicate_indices):,}")
    else:
        median_age = None
        duplicate_indices = None
        salary_lower = None
        salary_upper = None
    
    # Broadcast decisiones
    median_age = comm.bcast(median_age, root=0)
    duplicate_indices = comm.bcast(duplicate_indices, root=0)
    salary_lower = comm.bcast(salary_lower, root=0)
    salary_upper = comm.bcast(salary_upper, root=0)
    

    if rank == 0:
        print(f"\n Cleaning in parallel ({size} workers)...")
    
    # Cada worker limpia su chunk
    my_chunk['age'].fillna(median_age, inplace=True)
    my_chunk = my_chunk[~my_chunk.index.isin(duplicate_indices)]
    
    # Normalizar
    my_chunk['name'] = my_chunk['name'].str.lower().str.strip()
    my_chunk['email'] = my_chunk['email'].str.lower()
    my_chunk['country'] = my_chunk['country'].replace({
        'Gutemala': 'Guatemala',
        'GT': 'Guatemala',
        'guatemala': 'Guatemala'
    })
    
    # Corregir outliers
    my_chunk.loc[my_chunk['salary'] < salary_lower, 'salary'] = salary_lower
    my_chunk.loc[my_chunk['salary'] > salary_upper, 'salary'] = salary_upper
    

    clean_chunks = comm.gather(my_chunk, root=0)
    
    if rank == 0:
        print(f"   Cleaning completed")
        print("\n Consolidating results...")
        
        final_df = pd.concat(clean_chunks, ignore_index=True)
        final_df.to_csv('clean_cleanstream.csv', index=False)
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print(f" COMPLETED IN {elapsed:.2f} SECONDS")
        print("="*60)
        print(f"Final rows: {len(final_df):,}")
        print(f"Speedup: {size}x workers")
        print("="*60)
        
        return elapsed

if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'dirty_data.csv'
    clean(input_file)