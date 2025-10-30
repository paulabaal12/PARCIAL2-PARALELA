from mpi4py import MPI
import pandas as pd
import numpy as np
import time
import sys
import json
import re

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def is_valid_email(email):
    """Valida si un string tiene formato de correo electrónico."""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, str(email)))


def apply_cleaning_rules(df, config, dictionaries, stats):
    """
    Aplica las reglas de limpieza según la configuración JSON.
    """
    for column, rules in config.items():
        if column not in df.columns:
            continue

        ctype = rules.get("type")

        # --- 1. Missing value imputation ---
        if ctype == "missing_impute":
            strategy = rules.get("strategy", "median")
            if strategy == "median":
                df[column].fillna(stats.get(f"{column}_median", df[column].median()), inplace=True)
            elif strategy == "mean":
                df[column].fillna(stats.get(f"{column}_mean", df[column].mean()), inplace=True)

        # --- 2. String normalization / transformation ---
        elif ctype in ["string_normalize", "string_transform"]:
            ops = rules.get("operation", [])
            for op in ops:
                if op == "lower":
                    df[column] = df[column].astype(str).str.lower()
                elif op == "strip":
                    df[column] = df[column].astype(str).str.strip()

            # Validación (por ejemplo, email)
            if rules.get("validation", False) and column == "email":
                df = df[df[column].apply(is_valid_email)]

        # --- 3. Dictionary replace ---
        elif ctype == "dictionary_replace":
            dict_name = rules.get("dictionary_name")
            if dict_name in dictionaries:
                mapping = dictionaries[dict_name]
                replace_map = {}
                for k, vals in mapping.items():
                    for v in vals:
                        replace_map[v.lower()] = k
                df.loc[:, column] = df[column].astype(str).str.strip().str.lower().replace(replace_map)

        # --- 4. Outlier capping ---
        elif ctype == "outlier_capping":
            method = rules.get("method", "iqr_fence")
            cap_value = rules.get("cap_value", 1.5)
            if method == "iqr_fence" and f"{column}_bounds" in stats:
                lower, upper = stats[f"{column}_bounds"]
                df.loc[df[column] < lower, column] = lower
                df.loc[df[column] > upper, column] = upper

    return df


def clean(input_file, metadata_file="metadata.json"):
    if rank == 0:
        print("="*60)
        print(f"CLEANSTREAM (Parallel with {size} workers)")
        print("="*60)
        start_time = time.time()

    # ========================
    # Cargar configuración
    # ========================
    if rank == 0:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        cleaning_config = metadata.get("cleaning_config", {})
        dictionaries = metadata.get("dictionaries", {})
    else:
        cleaning_config = None
        dictionaries = None

    cleaning_config = comm.bcast(cleaning_config, root=0)
    dictionaries = comm.bcast(dictionaries, root=0)

    # ========================
    # Cargar y distribuir
    # ========================
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

    my_chunk = comm.scatter(chunks, root=0)

    if rank == 0:
        print(f"\n Analyzing in parallel ({size} workers)...")

    if "id" in my_chunk.columns:
        my_chunk["id"] = (
            my_chunk["id"]
            .astype(str)
            .str.replace(r"_dup$", "", regex=True)
            .str.strip()
            .str.lower()
        )


    # ========================
    # Duplicados y estadísticas
    # ========================
    local_missing = 0
    local_stats = {}
    if "age" in my_chunk.columns:
        local_missing = my_chunk["age"].isna().sum()

    total_missing = comm.reduce(local_missing, op=MPI.SUM, root=0)

    # Duplicados locales (hash)
    local_hashes = {}
    for idx, row in my_chunk.iterrows():
        row_hash = hash(tuple(row))
        if row_hash not in local_hashes:
            local_hashes[row_hash] = []
        local_hashes[row_hash].append(idx)

    all_hashes = comm.gather(local_hashes, root=0)

    # Estadísticas para imputación y outliers
    if "age" in my_chunk.columns:
        local_sum = my_chunk["age"].sum()
        local_count = my_chunk["age"].count()
    else:
        local_sum = local_count = 0

    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
    global_count = comm.reduce(local_count, op=MPI.SUM, root=0)

    if "salary" in my_chunk.columns:
        local_values = my_chunk["salary"].dropna().values
    else:
        local_values = np.array([])

    all_values = comm.gather(local_values, root=0)

    # ========================
    # Consolidar estadísticas
    # ========================
    if rank == 0:
        stats = {}
        if global_count > 0:
            stats["age_median"] = global_sum / global_count

        if len(all_values) > 0 and len(np.concatenate(all_values)) > 0:
            all_salary_values = np.concatenate(all_values)
            Q1 = np.percentile(all_salary_values, 25)
            Q3 = np.percentile(all_salary_values, 75)
            IQR = Q3 - Q1
            salary_lower = Q1 - 1.5 * IQR
            salary_upper = Q3 + 1.5 * IQR
            stats["salary_bounds"] = (salary_lower, salary_upper)
        else:
            stats["salary_bounds"] = (None, None)

        # Consolidar duplicados globales
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

        print(f"   Missing values: {total_missing:,}")
        print(f"   Duplicates: {len(duplicate_indices):,}")
    else:
        stats = None
        duplicate_indices = None

    stats = comm.bcast(stats, root=0)
    duplicate_indices = comm.bcast(duplicate_indices, root=0)

    # ========================
    # Limpieza paralela
    # ========================
    if rank == 0:
        print(f"\n Cleaning in parallel ({size} workers)...")

    # Eliminar duplicados
    my_chunk = my_chunk[~my_chunk.index.isin(duplicate_indices)]

    # Aplicar reglas JSON
    my_chunk = apply_cleaning_rules(my_chunk, cleaning_config, dictionaries, stats)

    clean_chunks = comm.gather(my_chunk, root=0)

    # ========================
    # Consolidar resultado
    # ========================
    if rank == 0:
        print("   Cleaning completed")
        print("\n Consolidating results...")

        final_df = pd.concat(clean_chunks, ignore_index=True)
        final_df.to_csv("clean_cleanstream.csv", index=False)

        elapsed = time.time() - start_time

        print("\n" + "="*60)
        print(f" COMPLETED IN {elapsed:.2f} SECONDS")
        print("="*60)
        print(f"Final rows: {len(final_df):,}")
        print(f"Speedup: {size}x workers")
        print("="*60)

        return elapsed


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "dirty_data.csv"
    metadata_file = sys.argv[2] if len(sys.argv) > 2 else "metadata.json"
    clean(input_file, metadata_file)
