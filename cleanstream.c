// cleanstream.c
// Prototype of a parallel cleaning pipeline using MPI + OpenMP (educational demo)
// Notes: This is a simplified prototype showing file-offset partitioning, global
// reductions for simple statistics, and per-rank output files. It uses mean/stddev
// for imputations/capping as a fast approximation. For production, replace with
// distributed median/IQR and global dedup by hashing/alltoallv.

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#define LINE_BUF 8192

void str_lower(char *s){ for(; *s; ++s) *s = tolower((unsigned char)*s); }
void trim(char *s){ char *p = s, *q; while(*p && isspace((unsigned char)*p)) p++; if(p!=s) memmove(s,p,strlen(p)+1); q = s + strlen(s) - 1; while(q>=s && isspace((unsigned char)*q)) *q-- = '\0'; }

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(argc < 2){ if(rank==0) fprintf(stderr,"Usage: %s dirty_data.csv\n", argv[0]); MPI_Finalize(); return 1; }
    const char *input = argv[1];

    long long nlines = 0;
    long long *offsets = NULL;
    long long header_end = 0;

    // record start time
    double t_start = MPI_Wtime();

    if(rank==0){
        FILE *f = fopen(input,"r");
        if(!f){ perror("fopen"); MPI_Abort(MPI_COMM_WORLD,1); }
        char buf[LINE_BUF];
        if(!fgets(buf, LINE_BUF, f)){ fclose(f); MPI_Abort(MPI_COMM_WORLD,1); }
        header_end = ftell(f);
        // count lines
        nlines = 0;
        while(fgets(buf, LINE_BUF, f)) nlines++;
        rewind(f);
        fgets(buf, LINE_BUF, f);
        offsets = malloc(sizeof(long long) * (nlines>0 ? nlines : 1));
        if(!offsets && nlines>0){ fprintf(stderr,"malloc offsets failed\n"); MPI_Abort(MPI_COMM_WORLD,1); }
        long long idx = 0; long long pos = ftell(f);
        while(fgets(buf, LINE_BUF, f)){
            offsets[idx++] = pos;
            pos = ftell(f);
        }
        fclose(f);
    }

    MPI_Bcast(&nlines, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    long long per = (nlines>0) ? nlines / size : 0;
    long long rem = (nlines>0) ? nlines % size : 0;
    long long my_start = (nlines>0) ? (rank * per + (rank < rem ? rank : rem)) : 0;
    long long my_count = (nlines>0) ? (per + (rank < rem ? 1 : 0)) : 0;
    long long my_end = my_start + my_count;

    long long my_start_offset = header_end, my_end_offset = -1;
    if(rank==0){
        for(int r=0;r<size;r++){
            long long r_start = r * per + (r < rem ? r : rem);
            long long r_count = per + (r < rem ? 1 : 0);
            long long r_end = r_start + r_count;
            long long start_off = (r_count>0) ? offsets[r_start] : header_end;
            long long end_off = (r_count>0) ? ((r_end < nlines) ? offsets[r_end] : -1) : start_off;
            if(r==0){ my_start_offset = start_off; my_end_offset = end_off; }
            else{
                long long buf2[2] = {start_off, end_off};
                MPI_Send(buf2, 2, MPI_LONG_LONG, r, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        long long buf2[2]; MPI_Recv(buf2,2,MPI_LONG_LONG,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        my_start_offset = buf2[0]; my_end_offset = buf2[1];
    }

    // open and stream my portion
    FILE *f = fopen(input,"r");
    if(!f){ perror("fopen worker"); MPI_Abort(MPI_COMM_WORLD,1); }
    if(fseeko(f, my_start_offset, SEEK_SET) != 0){ /* fallback to rewind */ rewind(f); fseeko(f, my_start_offset, SEEK_SET); }

    char line[LINE_BUF];
    char outname[256]; snprintf(outname, sizeof(outname), "clean_rank_%d.csv", rank);
    FILE *out = fopen(outname,"w"); if(!out){ perror("out fopen"); MPI_Abort(MPI_COMM_WORLD,1); }
    fprintf(out, "id,name,age,email,country,salary\n");

    double local_age_sum = 0.0; long long local_age_count = 0;
    double local_sal_sum = 0.0; double local_sal_sumsq = 0.0; long long local_sal_count = 0;

    while(fgets(line, LINE_BUF, f)){
        long long pos_after = ftell(f);
        if(my_end_offset != -1 && pos_after > my_end_offset) break;
        // simple CSV parse (no quoted commas)
        char tmp[LINE_BUF]; strncpy(tmp, line, LINE_BUF);
        char *p = strtok(tmp, ",\n");
        if(!p) continue;
        char id[128]; strncpy(id, p, sizeof(id)); id[sizeof(id)-1]='\0';
        char name[512] = ""; char age_s[64] = ""; char email[512] = ""; char country[128] = ""; char salary_s[64] = "";
        p = strtok(NULL, ",\n"); if(p) strncpy(name,p,sizeof(name));
        p = strtok(NULL, ",\n"); if(p) strncpy(age_s,p,sizeof(age_s));
        p = strtok(NULL, ",\n"); if(p) strncpy(email,p,sizeof(email));
        p = strtok(NULL, ",\n"); if(p) strncpy(country,p,sizeof(country));
        p = strtok(NULL, ",\n"); if(p) strncpy(salary_s,p,sizeof(salary_s));

        trim(name); str_lower(name); trim(email); str_lower(email); trim(country);

        if(strlen(age_s)>0){ double a = atof(age_s); local_age_sum += a; local_age_count++; }
        if(strlen(salary_s)>0){ double s = atof(salary_s); local_sal_sum += s; local_sal_sumsq += s*s; local_sal_count++; }

        // Write intermediate parsed row (we will apply imputation/capping after global stats)
        fprintf(out, "%s,%s,%s,%s,%s,%s\n", id, name, age_s, email, country, salary_s);
    }

    fclose(out);
    fclose(f);

    double global_age_sum = 0.0; long long global_age_count = 0;
    MPI_Allreduce(&local_age_sum, &global_age_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_age_count, &global_age_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    double global_mean_age = (global_age_count>0) ? (global_age_sum / global_age_count) : 0.0;

    double global_sal_sum = 0.0, global_sal_sumsq = 0.0; long long global_sal_count = 0;
    MPI_Allreduce(&local_sal_sum, &global_sal_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_sal_sumsq, &global_sal_sumsq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_sal_count, &global_sal_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    double sal_mean = (global_sal_count>0) ? (global_sal_sum / global_sal_count) : 0.0;
    double sal_var = (global_sal_count>0) ? (global_sal_sumsq/global_sal_count - sal_mean*sal_mean) : 0.0;
    double sal_std = (sal_var>0) ? sqrt(sal_var) : 0.0;
    double sal_lower = sal_mean - 3.0*sal_std;
    double sal_upper = sal_mean + 3.0*sal_std;

    if(rank==0){
        printf("Global mean age (approx): %.3f\n", global_mean_age);
        printf("Salary mean=%.2f std=%.2f -> bounds [%.2f, %.2f]\n", sal_mean, sal_std, sal_lower, sal_upper);
    }

    // Optional: run a sequential cleaning on rank 0 to produce a baseline time (T_s)
    double seq_elapsed = -1.0;
    long long seq_imputed = 0, seq_capped = 0, seq_rows = 0;
    if(rank == 0){
        double seq_start = MPI_Wtime();
        FILE *fin = fopen(input, "r");
        if(!fin){ perror("fopen seq"); MPI_Abort(MPI_COMM_WORLD,1); }
        char l[LINE_BUF];
        // skip header
        if(!fgets(l, LINE_BUF, fin)){ fclose(fin); MPI_Abort(MPI_COMM_WORLD,1); }
        FILE *fseqout = fopen("clean_sequential_output.csv", "w");
        if(!fseqout){ perror("fopen seqout"); MPI_Abort(MPI_COMM_WORLD,1); }
        fprintf(fseqout, "id,name,age,email,country,salary\n");

        while(fgets(l, LINE_BUF, fin)){
            char tmp[LINE_BUF]; strncpy(tmp, l, LINE_BUF);
            char *p = strtok(tmp, ",\n");
            if(!p) continue;
            char id[128]; strncpy(id, p, sizeof(id)); id[sizeof(id)-1]='\0';
            char name[512] = ""; char age_s[64] = ""; char email[512] = ""; char country[128] = ""; char salary_s[64] = "";
            p = strtok(NULL, ",\n"); if(p) strncpy(name,p,sizeof(name));
            p = strtok(NULL, ",\n"); if(p) strncpy(age_s,p,sizeof(age_s));
            p = strtok(NULL, ",\n"); if(p) strncpy(email,p,sizeof(email));
            p = strtok(NULL, ",\n"); if(p) strncpy(country,p,sizeof(country));
            p = strtok(NULL, ",\n"); if(p) strncpy(salary_s,p,sizeof(salary_s));

            trim(name); str_lower(name); trim(email); str_lower(email); trim(country);

            if(strlen(age_s) == 0){ char buf_age[64]; snprintf(buf_age, sizeof(buf_age), "%.0f", global_mean_age); strncpy(age_s, buf_age, sizeof(age_s)); seq_imputed++; }
            if(strlen(salary_s) > 0){ double s = atof(salary_s); if(s < sal_lower){ s = sal_lower; seq_capped++; } else if(s > sal_upper){ s = sal_upper; seq_capped++; } char buf_sal[64]; snprintf(buf_sal, sizeof(buf_sal), "%.2f", s); strncpy(salary_s, buf_sal, sizeof(salary_s)); }

            fprintf(fseqout, "%s,%s,%s,%s,%s,%s\n", id, name, age_s, email, country, salary_s);
            seq_rows++;
        }
        fclose(fin); fclose(fseqout);
        seq_elapsed = MPI_Wtime() - seq_start;
        printf("Sequential pass done (rows=%lld) in %.3f s\n", seq_rows, seq_elapsed);
    }

    // Second pass: open intermediate per-rank file, apply imputations and capping,
    // count changes and write final per-rank cleaned file.
    char interm_name[256]; snprintf(interm_name, sizeof(interm_name), "clean_rank_%d.csv", rank);
    char final_name[256]; snprintf(final_name, sizeof(final_name), "clean_rank_%d_final.csv", rank);
    FILE *fin = fopen(interm_name, "r");
    if(!fin){ perror("fopen interm"); MPI_Abort(MPI_COMM_WORLD,1); }
    FILE *fout = fopen(final_name, "w");
    if(!fout){ perror("fopen final"); MPI_Abort(MPI_COMM_WORLD,1); }

    // copy header
    if(fgets(line, LINE_BUF, fin)) fprintf(fout, "%s", line);

    long long local_imputed_age = 0;
    long long local_capped_salary = 0;
    long long local_final_rows = 0;

    // process lines
    while(fgets(line, LINE_BUF, fin)){
        char tmp[LINE_BUF]; strncpy(tmp, line, LINE_BUF);
        char *p = strtok(tmp, ",\n");
        if(!p) continue;
        char id[128]; strncpy(id, p, sizeof(id)); id[sizeof(id)-1]='\0';
        char name[512] = ""; char age_s[64] = ""; char email[512] = ""; char country[128] = ""; char salary_s[64] = "";
        p = strtok(NULL, ",\n"); if(p) strncpy(name,p,sizeof(name));
        p = strtok(NULL, ",\n"); if(p) strncpy(age_s,p,sizeof(age_s));
        p = strtok(NULL, ",\n"); if(p) strncpy(email,p,sizeof(email));
        p = strtok(NULL, ",\n"); if(p) strncpy(country,p,sizeof(country));
        p = strtok(NULL, ",\n"); if(p) strncpy(salary_s,p,sizeof(salary_s));

        trim(name); str_lower(name); trim(email); str_lower(email); trim(country);

        // Impute age if missing
        if(strlen(age_s) == 0){
            // write imputed value
            char buf_age[64]; snprintf(buf_age, sizeof(buf_age), "%.0f", global_mean_age);
            strncpy(age_s, buf_age, sizeof(age_s));
            local_imputed_age++;
        }

        // Salary capping
        if(strlen(salary_s) > 0){
            double s = atof(salary_s);
            if(s < sal_lower){ s = sal_lower; local_capped_salary++; }
            else if(s > sal_upper){ s = sal_upper; local_capped_salary++; }
            // write back into salary_s
            char buf_sal[64]; snprintf(buf_sal, sizeof(buf_sal), "%.2f", s);
            strncpy(salary_s, buf_sal, sizeof(salary_s));
        }

        // write final cleaned row
        fprintf(fout, "%s,%s,%s,%s,%s,%s\n", id, name, age_s, email, country, salary_s);
        local_final_rows++;
    }

    fclose(fin);
    fclose(fout);

    // Aggregate stats across ranks
    long long total_input_rows = 0;
    long long total_imputed_age = 0;
    long long total_capped_salary = 0;
    long long total_final_rows = 0;

    // local input rows is the earlier my_count (number of lines assigned)
    long long local_input_rows = my_count;

    MPI_Reduce(&local_input_rows, &total_input_rows, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_imputed_age, &total_imputed_age, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_capped_salary, &total_capped_salary, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_final_rows, &total_final_rows, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // timings: use MPI_Wtime - gather max elapsed across ranks
    double local_elapsed = MPI_Wtime() - t_start; // elapsed per rank
    // Note: we didn't record start with MPI_Wtime earlier; compute approximate by using process CPU start -> instead record start at top

    // We'll compute global_max_elapsed using MPI_Reduce with MAX
    double global_max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &global_max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0){
        // write report
        FILE *rep = fopen("clean_report.txt", "a");
        if(!rep) rep = stdout;
    time_t now = time(NULL);
    char timestr[64] = "";
    struct tm *tm_now = localtime(&now);
    if(tm_now) strftime(timestr, sizeof(timestr), "%Y-%m-%d %H:%M:%S", tm_now);
    fprintf(rep, "===== CleanStream Report (%s) =====\n", timestr);
        fprintf(rep, "Input file: %s\n", input);
        fprintf(rep, "Processes: %d\n", size);
        fprintf(rep, "Total input rows (assigned): %lld\n", total_input_rows);
        fprintf(rep, "Total imputed ages: %lld\n", total_imputed_age);
        fprintf(rep, "Total salary values capped: %lld\n", total_capped_salary);
        fprintf(rep, "Total final rows: %lld\n", total_final_rows);
        fprintf(rep, "Salary bounds used: [%.2f, %.2f]\n", sal_lower, sal_upper);
        fprintf(rep, "Global mean age used for imputation: %.3f\n", global_mean_age);
        fprintf(rep, "Elapsed parallel (approx, max across ranks): %.3f seconds\n", global_max_elapsed);
        if(seq_elapsed >= 0.0){
            fprintf(rep, "Elapsed sequential (rank 0 run): %.3f seconds\n", seq_elapsed);
            double speedup = (seq_elapsed > 0.0) ? (seq_elapsed / global_max_elapsed) : 0.0;
            fprintf(rep, "Speedup (seq/parallel): %.3fx\n", speedup);
            fprintf(rep, "Sequential imputed ages: %lld, capped salary: %lld, rows: %lld\n", seq_imputed, seq_capped, seq_rows);
        }
        fprintf(rep, "Elapsed (approx, max across ranks): %.3f seconds\n", global_max_elapsed);
        fprintf(rep, "=====================================\n\n");
        if(rep != stdout) fclose(rep);
        printf("Wrote clean_report.txt\n");
    }

    MPI_Finalize();
    return 0;
}
