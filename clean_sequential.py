import csv
import time
import sys

def clean_sequential(input_file):
    print("="*60)
    print("SEQUENTIAL VERSION")
    print("="*60)
    
    start_time = time.time()
    
    print("\n Loading data...")
    load_start = time.time()
    
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"  Original rows: {len(rows):,}")
    print(f"  Loaded in {time.time()-load_start:.2f}s")
    
    print("\n Analyzing sequentially...")
    analysis_start = time.time()
    
    # Contar missing values
    missing_count = 0
    ages = []
    salaries = []
    
    for row in rows:
        if not row['age'] or row['age'] == '':
            missing_count += 1
        else:
            ages.append(float(row['age']))
        
        if row['salary'] and row['salary'] != '':
            salaries.append(float(row['salary']))
    
    # Calcular mediana 
    ages_sorted = sorted(ages)
    n = len(ages_sorted)
    if n % 2 == 0:
        median_age = (ages_sorted[n//2-1] + ages_sorted[n//2]) / 2
    else:
        median_age = ages_sorted[n//2]
    
    # Calcular cuartiles para outliers 
    salaries_sorted = sorted(salaries)
    n_sal = len(salaries_sorted)
    Q1 = salaries_sorted[n_sal // 4]
    Q3 = salaries_sorted[3 * n_sal // 4]
    IQR = Q3 - Q1
    salary_lower = Q1 - 1.5 * IQR
    salary_upper = Q3 + 1.5 * IQR
    
    # Detectar duplicados 
    print("\n Finding duplicates...")
    duplicate_indices = set()
    for i in range(len(rows)):
        if i % 10000 == 0 and i > 0:
            print(f"  Checked {i:,} rows for duplicates...")
        
        if i in duplicate_indices:
            continue
        
        for j in range(i+1, len(rows)):
            is_duplicate = True
            for key in rows[i].keys():
                if rows[i][key] != rows[j][key]:
                    is_duplicate = False
                    break
            
            if is_duplicate:
                duplicate_indices.add(j)
    
    print(f"\n  Analysis completed in {time.time()-analysis_start:.2f}s")
    print(f"  Missing values: {missing_count:,}")
    print(f"  Duplicates: {len(duplicate_indices):,}")
    
    print("\n Cleaning sequentially (row-by-row)...")
    clean_start = time.time()
    
    cleaned_rows = []
    
    for i, row in enumerate(rows):
        if i % 10000 == 0 and i > 0:
            elapsed = time.time() - clean_start
            rate = i / elapsed
            print(f"  Cleaned {i:,} rows ({rate:.0f} rows/s)")
        
        # Saltar duplicados
        if i in duplicate_indices:
            continue
        
        # Imputar age 
        if not row['age'] or row['age'] == '':
            row['age'] = str(int(median_age))
        
        # Normalizar name 
        name = row['name']
        name = name.lower()
        name = name.strip()
        row['name'] = name
        
        # Normalizar email 
        email = row['email']
        email = email.lower()
        row['email'] = email
        
        # Normalizar country
        country = row['country']
        if country == 'Gutemala' or country == 'GT' or country == 'guatemala':
            row['country'] = 'Guatemala'
        
        # Corregir outliers en salary (UNO POR UNO)
        if row['salary'] and row['salary'] != '':
            salary = float(row['salary'])
            if salary < salary_lower:
                row['salary'] = str(salary_lower)
            elif salary > salary_upper:
                row['salary'] = str(salary_upper)
        
        cleaned_rows.append(row)
    
    print(f"\n  Cleaning completed in {time.time()-clean_start:.2f}s")
    
    print("\n Saving results...")
    
    with open('clean_sequential.csv', 'w', newline='', encoding='utf-8') as f:
        if cleaned_rows:
            writer = csv.DictWriter(f, fieldnames=cleaned_rows[0].keys())
            writer.writeheader()
            writer.writerows(cleaned_rows)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print(f" COMPLETED IN {elapsed:.2f} SECONDS")
    print("="*60)
    print(f"Original rows: {len(rows):,}")
    print(f"Final rows:    {len(cleaned_rows):,}")
    print(f"Removed:       {len(rows) - len(cleaned_rows):,}")
    print("="*60)
    
    return elapsed

if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'dirty_data.csv'
    clean_sequential(input_file)