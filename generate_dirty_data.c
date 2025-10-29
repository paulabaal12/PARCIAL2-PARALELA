// generate_dirty_data.c
// Simple CSV generator for dirty data with duplicates and outliers
// Usage: ./generate_dirty_data [n_rows] [output_file]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char *first_names[] = {
    "Juan","María","Pedro","Ana","Carlos","Luis","Sofía","Miguel","Lucía","Andrés",
    "Isabel","Diego","Marta","Jorge","Valeria","Pablo","Camila","Fernando","Elena","Raúl"
};
const char *last_names[] = {
    "Pérez","López","Gómez","Silva","Ruiz","García","Martínez","Rodríguez","Hernández","Fernández",
    "González","Sánchez","Ramírez","Vargas","Morales","Jiménez","Ramos","Castillo","Ortega","Vega"
};

const char *domains[] = {
    "gmail.com","yahoo.com","hotmail.com","example.com","empresa.org","mail.com"
};
const char *countries[] = {
    "Guatemala","Gutemala","GT","guatemala","USA","US","Gringolandia","Mexico","Mejico"
};

int main(int argc, char **argv){
    long n_rows = 10000000; // default 10M
    const char *outfile = "dirty_data.csv";
    if(argc > 1) n_rows = atol(argv[1]);
    if(argc > 2) outfile = argv[2];
    // optional seed as third arg for reproducibility
    unsigned int seed = 16;
    if(argc > 3) seed = (unsigned int)atoi(argv[3]);

    srand(seed);
    FILE *f = fopen(outfile, "w");
    if(!f){ perror("fopen"); return 1; }

    fprintf(f, "id,name,age,email,country,salary\n");

    for(long i=0;i<n_rows;i++){
    int first_i = rand() % (sizeof(first_names)/sizeof(first_names[0]));
    int last_i = rand() % (sizeof(last_names)/sizeof(last_names[0]));
    int country_i = rand() % (sizeof(countries)/sizeof(countries[0]));
        // 15% missing age
        int has_age = (rand() % 100) >= 15;
        int age = 18 + (rand() % 63); // 18..80
        double salary = 50000.0 + ((rand() / (double)RAND_MAX) - 0.5)*40000.0;

        // introduce outliers 1%
        if((rand() % 100) < 1){
            int o = rand() % 3;
            if(o==0) salary = 0;
            else if(o==1) salary = -5000;
            else salary = 5000000;
        }

        // build a variable name: sometimes include middle name, random casing and whitespace
        char namebuf[128];
        if(rand() % 10 == 0){
            // include middle name
            int mid_i = rand() % (sizeof(first_names)/sizeof(first_names[0]));
            snprintf(namebuf, sizeof(namebuf), "%s %s %s", first_names[first_i], first_names[mid_i], last_names[last_i]);
        } else {
            snprintf(namebuf, sizeof(namebuf), "%s %s", first_names[first_i], last_names[last_i]);
        }
        // random case variations
        for(char *p = namebuf; *p; ++p){
            int r = rand() % 100;
            if(r < 10) *p = toupper((unsigned char)*p);
            else if(r < 30) *p = tolower((unsigned char)*p);
            // else keep original (with accents as is)
        }
        // random leading/trailing spaces sometimes
        char name_out[160];
        if(rand() % 20 == 0) snprintf(name_out, sizeof(name_out), "   %s   ", namebuf);
        else snprintf(name_out, sizeof(name_out), "%s", namebuf);
        if(has_age) fprintf(f, "%d,", age);
        else fprintf(f, ",");

        // generate email based on name or random invalid email with small probability
        char emailbuf[128];
        int invalid_email_chance = rand() % 100;
        if(invalid_email_chance < 6){
            // a few invalid emails
            const char *invalids[] = {"pedro@","invalido-email","no-at-symbol.com","@nouser.com"};
            const char *inv = invalids[rand() % (sizeof(invalids)/sizeof(invalids[0]))];
            snprintf(emailbuf, sizeof(emailbuf), "%s", inv);
        } else {
            // build email from name: remove spaces and accent-like chars
            char user[128]; int ui = 0;
            for(char *p = namebuf; *p && ui < (int)sizeof(user)-1; ++p){
                char c = *p;
                if((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')){
                    user[ui++] = tolower((unsigned char)c);
                } else if(c == ' ' || c == '-'){
                    if(rand() % 3 == 0) user[ui++] = '.'; // sometimes dot
                } else {
                    // skip accents and other marks
                }
            }
            user[ui] = '\0';
            if(strlen(user) == 0) snprintf(user, sizeof(user), "user%ld", i);
            int pattern = rand() % 4;
            const char *dom = domains[rand() % (sizeof(domains)/sizeof(domains[0]))];
            if(pattern == 0) snprintf(emailbuf, sizeof(emailbuf), "%s@%s", user, dom);
            else if(pattern == 1) snprintf(emailbuf, sizeof(emailbuf), "%s%d@%s", user, (int)(rand()%1000), dom);
            else if(pattern == 2) snprintf(emailbuf, sizeof(emailbuf), "%c.%s@%s", (char)('a'+(rand()%26)), user, dom);
            else snprintf(emailbuf, sizeof(emailbuf), "%s.%s@%s", user, "mx", dom);
        }

        fprintf(f, "%s,%s,%.2f\n", name_out, emailbuf, countries[country_i], salary);

        // insert duplicate with 10% chance
        if((rand() % 100) < 10){
            // duplicate row - reuse same generated name/email for realism
            fprintf(f, "%ld_dup,%s,", i, name_out);
            if(has_age) fprintf(f, "%d,", age);
            else fprintf(f, ",");
            fprintf(f, "%s,%s,%.2f\n", emailbuf, countries[country_i], salary);
        }
    }

    fclose(f);
    printf("Generated approx %ld rows (plus duplicates) into %s\n", n_rows, outfile);
    return 0;
}
