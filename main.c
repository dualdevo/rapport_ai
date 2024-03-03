/*******************************************************
Nom ......... : main.c
Role ........ : Programme principal executant la lecture
                d'une image bitmap
Auteur ...... : Frédéric CHATRIE
Version ..... : V1.1 du 1/2/2021
Licence ..... : /

Compilation :
make veryclean
make
Pour exécuter, tapez : ./all
********************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Bmp2Matrix.h" // Assurez-vous que ce fichier est correctement implémenté.

typedef struct {
    int inputSize;
    int outputSize;
    float** weights;
    float* bias;
} LayerParams;

LayerParams* createLayerParams(int inputSize, int outputSize) {
    LayerParams* lp = (LayerParams*)malloc(sizeof(LayerParams));
    lp->inputSize = inputSize;
    lp->outputSize = outputSize;
    lp->weights = (float**)malloc(outputSize * sizeof(float*));
    for (int i = 0; i < outputSize; ++i) {
        lp->weights[i] = (float*)malloc(inputSize * sizeof(float));
    }
    lp->bias = (float*)malloc(outputSize * sizeof(float));
    return lp;
}

void freeLayerParams(LayerParams* lp) {
    for (int i = 0; i < lp->outputSize; ++i) {
        free(lp->weights[i]);
    }
    free(lp->weights);
    free(lp->bias);
    free(lp);
}

void readWeightsAndBiases(LayerParams* lp, const char* weightsFile, const char* biasesFile) {
    FILE* file = fopen(weightsFile, "r");
    if (!file) {
        perror("Error opening weights file");
        exit(1);
    }
    for (int i = 0; i < lp->outputSize; ++i) {
        for (int j = 0; j < lp->inputSize; ++j) {
            fscanf(file, "%f", &lp->weights[i][j]);
        }
    }
    fclose(file);

    file = fopen(biasesFile, "r");
    if (!file) {
        perror("Error opening biases file");
        exit(1);
    }
    for (int i = 0; i < lp->outputSize; ++i) {
        fscanf(file, "%f", &lp->bias[i]);
    }
    fclose(file);
}

void relu(float* input, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = fmax(0, input[i]);
    }
}

void softmax(float* input, int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i]);
        sum += input[i];
    }
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

void forward(LayerParams* lp, float* input, float* output) {
    for (int i = 0; i < lp->outputSize; ++i) {
        output[i] = lp->bias[i];
        for (int j = 0; j < lp->inputSize; ++j) {
            output[i] += input[j] * lp->weights[i][j];
        }
    }
}

void flatten(BMP* bitmap, float* out) {
    for (int i = 0; i < bitmap->infoHeader.hauteur; i++) {
        for (int j = 0; j < bitmap->infoHeader.largeur; j++) {
            out[i * bitmap->infoHeader.largeur + j] = bitmap->mPixelsGray[i][j];
        }
    }
}

int main() {
    // Le code pour charger une image, initialiser les couches, faire l'inférence et nettoyer.
    // Remplacez "path_to_your_image.bmp" par le chemin vers votre image BMP.
    const char* imagePath = "E:\\pimgtest\\2_0.bmp";
    BMP bitmap;
    FILE* pFile = fopen(imagePath, "rb");
    if (!pFile) {
        printf("Erreur dans la lecture du fichier %s\n", imagePath);
        return 1;
    }

    LireBitmap(pFile, &bitmap);
    fclose(pFile);
    ConvertRGB2Gray(&bitmap);

    float input[784]; // Pour une image 28x28
    flatten(&bitmap, input);
    DesallouerBMP(&bitmap);

    // Assurez-vous que les chemins vers les fichiers de poids et de biais sont corrects
    LayerParams* layer1 = createLayerParams(784, 128);
    LayerParams* layer2 = createLayerParams(128, 64);
    LayerParams* layer3 = createLayerParams(64, 10);
    readWeightsAndBiases(layer1, "layer_weight_dense.txt", "layer_bias_dense.txt");
    readWeightsAndBiases(layer2, "layer_weight_dense_1.txt", "layer_bias_dense_1.txt");
    readWeightsAndBiases(layer3, "layer_weight_dense_2.txt", "layer_bias_dense_2.txt");

    float layer1Output[128], layer2Output[64], finalOutput[10];
    forward(layer1, input, layer1Output);
    relu(layer1Output, 128);
    forward(layer2, layer1Output, layer2Output);
    relu(layer2Output, 64);
    forward(layer3, layer2Output, finalOutput);
    softmax(finalOutput, 10);

    for (int i = 0; i < 10; i++) {
        printf("Classe %d: Probabilité %f\n", i, finalOutput[i]);
    }

    freeLayerParams(layer1);
    freeLayerParams(layer2);
    freeLayerParams(layer3);

    return 0;
}