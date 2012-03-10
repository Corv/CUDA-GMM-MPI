/*
 * readData.cpp
 *
 *  Created on: Nov 4, 2008
 *      Author: Doug Roberts
 *      Modified by: Andrew Pangborn
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

extern "C"
float* readData(char* f, int* ndims, int* nevents);
float* readBIN(char* f, int* ndims, int* nevents);
float* readCSV(char* f, int* ndims, int* nevents);

float* readData(char* f, int* ndims, int* nevents) {
    int length = strlen(f);
    printf("File Extension: %s\n",f+length-3);
    if(strcmp(f+length-3,"bin") == 0) {
        return readBIN(f,ndims,nevents);
    } else {
        return readCSV(f,ndims,nevents);
    }
}

float* readBIN(char* f, int* ndims, int* nevents) {
    FILE* fin = fopen(f,"rb");

    fread(nevents,4,1,fin);
    fread(ndims,4,1,fin);
    int num_elements = (*ndims)*(*nevents);
    printf("Number of rows: %d\n",*nevents);
    printf("Number of cols: %d\n",*ndims);
    float* data = (float*) malloc(sizeof(float)*num_elements);
    fread(data,sizeof(float),num_elements,fin);
    fclose(fin);
    return data;
}

float* readCSV(char* f, int* ndims, int* nevents) {
    string line1;
    ifstream file(f);
    vector<string> lines;
    int num_dims = 0;
    char* temp;
    float* data;

    if (file.is_open()) {
        while(!file.eof()) {
            getline(file, line1);

            if (!line1.empty()) {
                lines.push_back(line1);
            }
        }

        file.close();
    }
    else {
        cout << "Unable to read the file " << f << endl;
        return NULL;
    }
    
    if(lines.size() > 0) {
        line1 = lines[0];
        string line2 (line1.begin(), line1.end());

        temp = strtok((char*)line1.c_str(), ",");

        while(temp != NULL) {
            num_dims++;
            temp = strtok(NULL, ",");
        }

        lines.erase(lines.begin()); // Remove first line, assumed to be header
        int num_events = (int)lines.size();
        int pad_size = 64 - (num_events % 64);
        if(pad_size == 64) pad_size = 0;
        pad_size = 0;

        printf("Number of events in input file: %d\n",num_events);
        printf("Number of padding events added for alignment: %d\n",pad_size);

        // Allocate space for all the FCS data
        data = (float*)malloc(sizeof(float) * num_dims * (num_events+pad_size));
        if(!data){
            printf("Cannot allocate enough memory for FCS data.\n");
            return NULL;
        }

        for (int i = 0; i < num_events; i++) {
            temp = strtok((char*)lines[i].c_str(), ",");

            for (int j = 0; j < num_dims; j++) {
                if(temp == NULL) {
                    free(data);
                    return NULL;
                }
                data[i * num_dims + j] = atof(temp);
                temp = strtok(NULL, ",");
            }
        }

        for(int i = num_events; i < num_events+pad_size; i++) {
            for(int j = 0; j < num_dims; j++) {
                data[i * num_dims + j] = 0.0f;
            }
        }
        num_events += pad_size;

        *ndims = num_dims;
        *nevents = num_events;

        return data;    
    } else {
        return NULL;
    }
    
    
}
