#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdbool.h>

// --- Neural Network Definitions ---

// A helper struct for storing a 2D index.
typedef struct {
    int i;
    int j;
} Index;

// The neural network structure.
typedef struct {
    int numLayers;
    unsigned short *layerSizes;      // Array of layer sizes
    unsigned char **layers;          // Each layer is an array of unsigned char

    int numSynapses;                 // Equals numLayers - 1
    unsigned char **synapses;        // Each synapse matrix is stored as a 1D array of size (rows*cols)
    unsigned char **synapseActivity;
    unsigned char **longsynapseActivity;
    unsigned char **superlongsynapseActivity;

    int cycles;
    int longmagnitude;
    int superlongmagnitude;
} NeuralNetwork;

// Create and initialize a neural network.
NeuralNetwork *createNeuralNetwork(unsigned short *layerSizes, int numLayers) {
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    if (!nn) { perror("malloc"); exit(1); }
    nn->numLayers = numLayers;
    nn->layerSizes = malloc(numLayers * sizeof(unsigned short));
    if (!nn->layerSizes) { perror("malloc"); exit(1); }
    for (int i = 0; i < numLayers; i++) {
        nn->layerSizes[i] = layerSizes[i];
    }
    // Allocate layers
    nn->layers = malloc(numLayers * sizeof(unsigned char *));
    if (!nn->layers) { perror("malloc"); exit(1); }
    for (int i = 0; i < numLayers; i++) {
        nn->layers[i] = calloc(nn->layerSizes[i], sizeof(unsigned char));
        if (!nn->layers[i]) { perror("calloc"); exit(1); }
    }
    // There are (numLayers-1) synapse matrices.
    nn->numSynapses = numLayers - 1;
    nn->synapses = malloc(nn->numSynapses * sizeof(unsigned char *));
    nn->synapseActivity = malloc(nn->numSynapses * sizeof(unsigned char *));
    nn->longsynapseActivity = malloc(nn->numSynapses * sizeof(unsigned char *));
    nn->superlongsynapseActivity = malloc(nn->numSynapses * sizeof(unsigned char *));
    if (!nn->synapses || !nn->synapseActivity || !nn->longsynapseActivity || !nn->superlongsynapseActivity) {
        perror("malloc");
        exit(1);
    }
    // Initialize each synapse matrix and its activity trackers.
    for (int l = 0; l < nn->numSynapses; l++) {
        int rows = nn->layerSizes[l];
        int cols = nn->layerSizes[l+1];
        int size = rows * cols;
        nn->synapses[l] = malloc(size * sizeof(unsigned char));
        nn->synapseActivity[l] = calloc(size, sizeof(unsigned char));
        nn->longsynapseActivity[l] = calloc(size, sizeof(unsigned char));
        nn->superlongsynapseActivity[l] = calloc(size, sizeof(unsigned char));
        if (!nn->synapses[l] || !nn->synapseActivity[l] ||
            !nn->longsynapseActivity[l] || !nn->superlongsynapseActivity[l]) {
            perror("malloc");
        exit(1);
            }
            // Initialize synapse weights randomly between 120 and 129.
            for (int i = 0; i < size; i++) {
                nn->synapses[l][i] = 120 + (rand() % 10);
            }
    }
    nn->cycles = 1;
    nn->longmagnitude = 0;
    nn->superlongmagnitude = 0;
    return nn;
}

// The forward propagation function.
// It clears all non‐input layers, sets the input layer from combinedInput,
// then propagates through the network.
unsigned char* forward(NeuralNetwork *nn, unsigned char *combinedInput) {
    // Clear all layers except the input layer.
    for (int l = 1; l < nn->numLayers; l++) {
        memset(nn->layers[l], 0, nn->layerSizes[l] * sizeof(unsigned char));
    }
    // Copy input into the first layer.
    memcpy(nn->layers[0], combinedInput, nn->layerSizes[0] * sizeof(unsigned char));

    // Propagate forward.
    for (int l = 0; l < nn->numSynapses; l++) {
        int rows = nn->layerSizes[l];
        int cols = nn->layerSizes[l+1];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                unsigned char weight = nn->synapses[l][i * cols + j];
                if (weight > 128) {
                    nn->layers[l+1][j] += nn->layers[l][i];
                    nn->synapseActivity[l][i * cols + j]++;  // Increment short-term activity.
                }
            }
        }
    }
    // Return the output layer.
    return nn->layers[nn->numLayers - 1];
}

// The learning function adjusts synapse weights based on a reward signal.
void learn(NeuralNetwork *nn, bool reward, int magnitude) {
    nn->longmagnitude += magnitude * (reward ? 1 : -1);

    if (nn->cycles % 3 == 0) {
        nn->superlongmagnitude += nn->longmagnitude;
        magnitude = abs(nn->longmagnitude);
        reward = (nn->longmagnitude > 0);
        // Copy long-term activity into the short-term tracker.
        for (int l = 0; l < nn->numSynapses; l++) {
            int size = nn->layerSizes[l] * nn->layerSizes[l+1];
            memcpy(nn->synapseActivity[l], nn->longsynapseActivity[l], size);
        }
        printf("Long memory is working\n");
    }
    if (nn->cycles % 6 == 0) {
        magnitude = abs(nn->superlongmagnitude);
        reward = (nn->superlongmagnitude > 0);
        for (int l = 0; l < nn->numSynapses; l++) {
            int size = nn->layerSizes[l] * nn->layerSizes[l+1];
            memcpy(nn->synapseActivity[l], nn->superlongsynapseActivity[l], size);
        }
        printf("Super Long memory is working\n");
    }

    // For each synapse layer, update weights.
    for (int l = 0; l < nn->numSynapses; l++) {
        int rows = nn->layerSizes[l];
        int cols = nn->layerSizes[l+1];
        int total = rows * cols;

        // Allocate temporary arrays to record indices.
        Index *activeIndices = malloc(total * sizeof(Index));
        Index *inactiveIndices = malloc(total * sizeof(Index));
        int activeCount = 0, inactiveCount = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (nn->synapseActivity[l][i * cols + j] != 0) {
                    activeIndices[activeCount].i = i;
                    activeIndices[activeCount].j = j;
                    activeCount++;
                } else {
                    inactiveIndices[inactiveCount].i = i;
                    inactiveIndices[inactiveCount].j = j;
                    inactiveCount++;
                }
            }
        }
        // If rewarded, increase weights on active synapses.
        if (reward) {
            if (activeCount > 0) {
                for (int m = 0; m < magnitude; m++) {
                    int index = rand() % activeCount;
                    int i = activeIndices[index].i;
                    int j = activeIndices[index].j;
                    int idx = i * cols + j;
                    if (nn->synapses[l][idx] < 254)
                        nn->synapses[l][idx]++;
                }
            }
        } else { // Punishment: decrease active synapses; reset inactive ones.
            if (activeCount > 0) {
                for (int m = 0; m < magnitude; m++) {
                    int index = rand() % activeCount;
                    int i = activeIndices[index].i;
                    int j = activeIndices[index].j;
                    int idx = i * cols + j;
                    if (nn->synapses[l][idx] > 2)
                        nn->synapses[l][idx]--;
                }
            }
            if (inactiveCount > 0) {
                for (int m = 0; m < magnitude; m++) {
                    int index = rand() % inactiveCount;
                    int i = inactiveIndices[index].i;
                    int j = inactiveIndices[index].j;
                    int idx = i * cols + j;
                    nn->synapses[l][idx] = 129;
                }
            }
        }
        // Accumulate short-term activity into long-term trackers.
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                nn->longsynapseActivity[l][idx] += nn->synapseActivity[l][idx];
                nn->superlongsynapseActivity[l][idx] += nn->synapseActivity[l][idx];
            }
        }
        free(activeIndices);
        free(inactiveIndices);
    }
    // Clear long-term accumulators periodically.
    if (nn->cycles % 3 == 0) {
        for (int l = 0; l < nn->numSynapses; l++) {
            int size = nn->layerSizes[l] * nn->layerSizes[l+1];
            memset(nn->longsynapseActivity[l], 0, size);
        }
        nn->longmagnitude = 0;
    }
    if (nn->cycles % 6 == 0) {
        for (int l = 0; l < nn->numSynapses; l++) {
            int size = nn->layerSizes[l] * nn->layerSizes[l+1];
            memset(nn->superlongsynapseActivity[l], 0, size);
        }
        nn->superlongmagnitude = 0;
    }
    // Clear the short-term activity tracker.
    for (int l = 0; l < nn->numSynapses; l++) {
        int size = nn->layerSizes[l] * nn->layerSizes[l+1];
        memset(nn->synapseActivity[l], 0, size);
    }
    nn->cycles++;
}

// A helper to print the output layer and a sample synapse value.
void printNetwork(NeuralNetwork *nn) {
    int outputLayer = nn->numLayers - 1;
    printf("\nOutput: ");
    for (int i = 0; i < nn->layerSizes[outputLayer]; i++) {
        printf("%d, ", nn->layers[outputLayer][i]);
    }
    if (nn->numSynapses > 0 && nn->layerSizes[outputLayer] > 1) {
        int l = nn->numSynapses - 1;
        int rows = nn->layerSizes[l];
        int cols = nn->layerSizes[l+1];
        if (rows > 1 && cols > 1)
            printf("\nSample synapse value: %d", nn->synapses[l][1 * cols + 1]);
    }
    // Visualize output as characters.
    printf("\n");
    for (int i = 0; i < nn->layerSizes[outputLayer]; i++) {
        printf("%c", nn->layers[outputLayer][i] / 3);
    }
    printf("\n");
}

// Free all dynamically allocated memory for the network.
void freeNeuralNetwork(NeuralNetwork *nn) {
    if (!nn) return;
    for (int i = 0; i < nn->numLayers; i++) {
        free(nn->layers[i]);
    }
    free(nn->layers);
    for (int l = 0; l < nn->numSynapses; l++) {
        free(nn->synapses[l]);
        free(nn->synapseActivity[l]);
        free(nn->longsynapseActivity[l]);
        free(nn->superlongsynapseActivity[l]);
    }
    free(nn->synapses);
    free(nn->synapseActivity);
    free(nn->longsynapseActivity);
    free(nn->superlongsynapseActivity);
    free(nn->layerSizes);
    free(nn);
}

// --- Game and Environment Definitions ---

#define GRID_ROWS 10
#define GRID_COLS 15

char grid[GRID_ROWS][GRID_COLS];
int player1_x = 2, player1_y = 2;
int player2_x = 8, player2_y = 8;
int currentPlayer = 1;

// Initialize the grid, obstacles, and player positions.
void initializeGrid() {
    for (int i = 0; i < GRID_ROWS; i++)
        for (int j = 0; j < GRID_COLS; j++)
            grid[i][j] = '.';

    // Add obstacles (example layout)
    for (int i = 0; i < 5; i++) {
        if (i < GRID_COLS)
            grid[4][i] = '#';
        if ((i+5) < GRID_COLS && 6 < GRID_ROWS)
            grid[6][i+5] = '#';
        if ((i+1) < GRID_ROWS && 10 < GRID_COLS)
            grid[i+1][10] = '#';
        if ((i+3) < GRID_COLS && 8 < GRID_ROWS)
            grid[8][i+3] = '#';
        if ((i+3) < GRID_ROWS && 12 < GRID_COLS)
            grid[i+3][12] = '#';
        if (i < GRID_ROWS && 7 < GRID_COLS)
            grid[i][7] = '#';
        if ((i+1) < GRID_COLS && 1 < GRID_ROWS)
            grid[1][i+1] = '#';
    }
    // Place players
    grid[player1_x][player1_y] = '1';
    grid[player2_x][player2_y] = '2';
}

// Draw the grid to the console.
void drawGrid() {
    // On Unix you might use "clear"; on Windows "cls"
    system("clear");
    for (int i = 0; i < GRID_ROWS; i++) {
        for (int j = 0; j < GRID_COLS; j++)
            printf("%c ", grid[i][j]);
        printf("\n");
    }
    printf("\n");
}

// Toggle the current player.
void switchPlayer() {
    currentPlayer = (currentPlayer == 1) ? 2 : 1;
    printf("Switched to Player %d!\n", currentPlayer);
}

// Move the current player based on a direction character.
bool movePlayer(char direction) {
    int currentX = (currentPlayer == 1) ? player1_x : player2_x;
    int currentY = (currentPlayer == 1) ? player1_y : player2_y;
    int newX = currentX, newY = currentY;

    switch (tolower(direction)) {
        case 'w': newX--; break;
        case 'a': newY--; break;
        case 's': newX++; break;
        case 'd': newY++; break;
        case '5': break;
        default: return false;
    }
    if (newX < 0 || newX >= GRID_ROWS || newY < 0 || newY >= GRID_COLS ||
        grid[newX][newY] == '#' || grid[newX][newY] == '1' || grid[newX][newY] == '2')
        return false;

    // Update grid and player position.
    grid[currentX][currentY] = '.';
    grid[newX][newY] = (currentPlayer == 1) ? '1' : '2';
    if (currentPlayer == 1) {
        player1_x = newX;
        player1_y = newY;
    } else {
        player2_x = newX;
        player2_y = newY;
    }
    return true;
}

// Encode a 3x3 area around (aiX,aiY) into a 9-byte array.
void encodeEnvironment(int aiX, int aiY, unsigned char *envInput) {
    int index = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int x = aiX + i;
            int y = aiY + j;
            if (x < 0 || x >= GRID_ROWS || y < 0 || y >= GRID_COLS) {
                envInput[index++] = 0;
            } else {
                char cell = grid[x][y];
                if (cell == '#')
                    envInput[index++] = 128;
                else if (cell == '1')
                    envInput[index++] = 192;
                else if (cell == '2')
                    envInput[index++] = 128;
                else
                    envInput[index++] = 64;
            }
        }
    }
}

// Interpret the first four output neurons to choose a direction.
char interpretAIOutput(unsigned char *output) {
    if (output[0] > output[1] && output[0] > output[2] && output[0] > output[3])
        return 'w';
    else if (output[1] > output[0] && output[1] > output[2] && output[1] > output[3])
        return 'a';
    else if (output[2] > output[0] && output[2] > output[1] && output[2] > output[3])
        return 's';
    else if (output[3] > output[0] && output[3] > output[1] && output[3] > output[2])
        return 'd';
    else
        return 's';
}

// --- Main Program ---

int main() {
    srand(time(NULL));  // Seed random number generator
    initializeGrid();
    drawGrid();

    // Create the network with layers: 1200 (input), 1000, 1000, 5 (output)
    unsigned short nnLayers[] = {1200, 1000, 1000, 5};
    int numLayers = sizeof(nnLayers) / sizeof(nnLayers[0]);
    NeuralNetwork *nn = createNeuralNetwork(nnLayers, numLayers);

    int gameCycles = 0;
    char aiMove;
    while (true) {
        int reward = 0;
        // Prepare a 1200-byte input vector.
        unsigned char combinedInput[1200] = {0};
        // For example, copy a short text ("s") into the input.
        combinedInput[0] = 's';
        // Encode the environment (using player1's position here) into 9 bytes,
        // and copy it into combinedInput starting at offset 500.
        unsigned char envInput[9];
        encodeEnvironment(player1_x, player1_y, envInput);
        memcpy(&combinedInput[500], envInput, 9);
        gameCycles++;
        unsigned char *output = forward(nn, combinedInput);

        // (In C# there was a check on output[^1] > 255; in C an unsigned char is 0–255.)
        printNetwork(nn);
        switchPlayer();

        // Alternate between AI and hardcoded player move.
        if (gameCycles % 2 == 0) {
            printf("AI move\n");
            aiMove = interpretAIOutput(output);
            printf("AI chooses: %c\n", aiMove);

                if (!movePlayer(aiMove)) {
                    printf("Invalid move!\n");
                    reward -= 500;
                   
                } else {
                    reward += 100;
                    drawGrid();
                }

            learn(nn, reward > 0, abs(reward));
            

        } else {
            printf("Your move\n");
            aiMove = 'w';  // Hardcoded move
        }

       //gameCycles++;

        if (gameCycles > 1000)
            break;
    }

    freeNeuralNetwork(nn);
    return 0;
}
