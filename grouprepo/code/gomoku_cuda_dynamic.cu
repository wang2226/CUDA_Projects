#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <bits/stdc++.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cooperative_groups.h>

using namespace std;

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

// Dimension of board
#define BOARD_SIZE 8
#define DEPTH 2
#define THREADS 768

enum Side {
	AI, Human
};

class Move {
public:
int x, y;

CUDA_CALLABLE_MEMBER Move(int x, int y) {
	this->x = x;
	this->y = y;
}

CUDA_CALLABLE_MEMBER int getX() {
	return this->x;
}
CUDA_CALLABLE_MEMBER int getY() {
	return this->y;
}
CUDA_CALLABLE_MEMBER void setX(int x) {
	this->x = x;
}
CUDA_CALLABLE_MEMBER void setY(int y) {
	this->y = y;
}
};

class Board {
private:
/* Add a stone to the location */
CUDA_CALLABLE_MEMBER void addStone(Side side, int x, int y) {
	if(onBoard(x, y)) {
		taken[x + BOARD_SIZE * y] = 1;
		if(side == AI) {
			ai[x + BOARD_SIZE * y] = 1;
		}
	}
}

CUDA_CALLABLE_MEMBER bool onBoard(int x, int y) {
	return(0 <= x && x < BOARD_SIZE && 0 <= y && y < BOARD_SIZE);
}

public:
char* taken;
char* ai;

// constructor
CUDA_CALLABLE_MEMBER Board(){
	taken = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));
	ai = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));

	// the board 0 for empty
	for (int i = 0; i < BOARD_SIZE; i++) {
		for (int j = 0; j < BOARD_SIZE; j++) {
			taken[i + BOARD_SIZE * j] = 0;
			ai[i + BOARD_SIZE * j] = 0;
		}
	}
}

CUDA_CALLABLE_MEMBER Board(char *ai, char *taken) {
	this->ai = ai;
	this->taken = taken;
}

/* Destructor for the board. */
CUDA_CALLABLE_MEMBER ~Board() {
	free(taken);
	free(ai);
}

/* Returns a copy of this board. */
Board* copy() {
	Board* newBoard = new Board();
	for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
		newBoard->ai[i] = ai[i];
		newBoard->taken[i] = taken[i];
	}
	return newBoard;
}

CUDA_CALLABLE_MEMBER bool occupied(int x, int y) {
	if (onBoard(x, y)) {
		return taken[x + BOARD_SIZE * y] == 1;
	}
	return false;
}

/* Current count of stones. */
CUDA_CALLABLE_MEMBER int countTaken() {
	int count = 0;
	for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
		count += taken[i];
	}
	return count;
}

/* Returns true if there are legal moves left. */
CUDA_CALLABLE_MEMBER bool hasMoves() {
	for (int i = 0; i < BOARD_SIZE; i++) {
		for (int j = 0; j < BOARD_SIZE; j++)
			if (!occupied(i, j)) return true;
	}
	return false;
}

CUDA_CALLABLE_MEMBER char* getAI() {
	return ai;
}

CUDA_CALLABLE_MEMBER char* getTaken() {
	return taken;
}
/*
 * Returns true if the game is finished; false otherwise. The game is finished
 * if neither side has a legal move.
 */
CUDA_CALLABLE_MEMBER bool isDone() {
	if (checkDone(AI) || checkDone(Human) || !hasMoves()) {
		return true;
	}
	return false;
}

CUDA_CALLABLE_MEMBER bool checkDone(Side side) {
	int square[BOARD_SIZE * BOARD_SIZE];

	for (int i = 0; i < BOARD_SIZE; i++) {
		for (int j = 0; j < BOARD_SIZE; j++) {
			if(side == AI)
				square[i + j * BOARD_SIZE] = ai[i + j * BOARD_SIZE];
			else if(side == Human)
				square[i + j * BOARD_SIZE] = taken[i + j * BOARD_SIZE] - ai[i + j * BOARD_SIZE];
		}
	}

	int moves [] = {1,-1,1,1,1,0,0,1};
	for (int i = 0; i < BOARD_SIZE; i++) {
		for (int j = 0; j < BOARD_SIZE; j++) {

			if (square[i + j * BOARD_SIZE] == 0) {
				continue;
			}
			for (int k = 0; k < 8; k += 2) {
				int direction_one = moves[k];
				int direction_two = moves[k + 1];
				int x = i, y = j, count = 0;

				for (int l = 0; l < 5; l++) {
					if (!onBoard(x, y)) {
						break;
					}
					if (square[x + y * BOARD_SIZE] == 0) {
						break;
					}
					x += direction_one;
					y += direction_two;
					count += 1;
				}
				if (count == 5) {
					return true;
				}
			}
		}
	}
	return false;
}

// Returns a list of possible moves for the specified side
vector<Move> getMoves(Board* currentBoard) {
	vector<Move> movesList;
	for (int i = 0; i < BOARD_SIZE; i++) {
		for (int j = 0; j < BOARD_SIZE; j++) {
			if (!currentBoard->occupied(i,j)) {
				/*
				        if (currentBoard->occupied(i + 1,j) || currentBoard->occupied(i - 1,j)
				 || currentBoard->occupied(i,j - 1) || currentBoard->occupied(i,j + 1)
				 || currentBoard->occupied(i + 1,j - 1) || currentBoard->occupied(i + 1,j + 1)
				 || currentBoard->occupied(i - 1,j - 1) || currentBoard->occupied(i - 1,j + 1)) {
				            // If spot is adjacent to an already taken spot, add it to list.
				 */
				Move move(i, j);
				movesList.push_back(move);
			}
		}
	}
	int size = movesList.size();
	for (int i = 0; i < size - 1; i++) {
		int j = i + rand() % (size - i);
		swap(movesList[i], movesList[j]);
	}
	return movesList;
}


/* Modifies the board to reflect the specified move. */
CUDA_CALLABLE_MEMBER void doMove(Move *m, Side side) {
	// A NULL move means pass.
	if (m == NULL)
		return;

	int x = m->getX();
	int y = m->getY();

	addStone(side, x, y);
}

/* Helper function to print the board status */
CUDA_CALLABLE_MEMBER void printBoard() {
	for (int i = 0; i < BOARD_SIZE; i++) {
		printf("%2d: ", i);
		for (int j = 0; j < BOARD_SIZE; j++) {
			if(ai[i * BOARD_SIZE + j] == 1)
				printf("%c ", 'X');
			else if(taken[i * BOARD_SIZE + j] == 1)
				printf("%c ", 'O');
			else
				printf("%c ", '-');
		}
		printf("\n");
	}
	printf("\n");
}
};


class Node {
private:
Board* board;
Move* move;
// the side that made the move leading to this node
Side side;
// our side - maximizing side
Side evaluater;
int alpha;
int beta;
// Node* parent;

public:
CUDA_CALLABLE_MEMBER Node(Move* move, Side side, Side evaluater, Board* board) {
	this->board = board;
	this->move = move;
	this->side = side;
	this->evaluater = evaluater;
	this->alpha = INT_MIN;
	this->beta = INT_MAX;
	// this->parent = NULL;
}

CUDA_CALLABLE_MEMBER ~Node() {
	delete this->board;
}
CUDA_CALLABLE_MEMBER Board* getBoard() {
	return this->board;
}
CUDA_CALLABLE_MEMBER Move* getMove() {
	return this->move;
}
CUDA_CALLABLE_MEMBER Side getSide() {
	return this->side;
}
// CUDA_CALLABLE_MEMBER Node* getParent() { return this->parent; }
// CUDA_CALLABLE_MEMBER void setParent(Node* node) { this->parent = node; }
CUDA_CALLABLE_MEMBER int getAlpha() {
	return this->alpha;
}
CUDA_CALLABLE_MEMBER int getBeta() {
	return this->beta;
}
CUDA_CALLABLE_MEMBER void setAlpha(int alpha) {
	this->alpha = alpha;
}
CUDA_CALLABLE_MEMBER void setBeta(int beta) {
	this->beta = beta;
}
};

// Copy data from host to device
template <class T> void CopyData(T* input, unsigned int N, unsigned int dsize, T** d_in) {
	// Allocate pinned memory on host (for faster HtoD copy)
	T* h_in_pinned = NULL;
	checkCudaErrors(cudaMallocHost((void**) &h_in_pinned, N * dsize));
	assert(h_in_pinned);
	memcpy(h_in_pinned, input, N * dsize);

	// copy data
	checkCudaErrors(cudaMalloc((void**) d_in, N * dsize));
	checkCudaErrors(cudaMemcpy(*d_in, h_in_pinned, N * dsize, cudaMemcpyHostToDevice));
}

// __device__ bool occupied(char* taken, int x, int y) {
CUDA_CALLABLE_MEMBER bool occupied(char* taken, int x, int y) {
	if (0 <= x && x < BOARD_SIZE && 0 <= y && y < BOARD_SIZE) {
		return taken[x + BOARD_SIZE * y] == 1;
	}
	return false;
}

/*           Gomoku Modle Evaluation Table
 *        Modle Name        Modle        Value
 *        Live Four         ?AAAA?       300000
 *        Dead Four A        AAAA?       2500
 *        Dead Four B        AAA?A       3000
 *        Dead Four C        AA?AA       2600
 *        Live Three        ??AAA??      3000
 *        Dead Three A       AAA??       500
 *        Dead Three B      ?A?AA?       800
 *        Dead Three C       A??AA       600
 *        Dead Three D       A?A?A       550
 *        Live Two          ???AA???     650
 *        Dead Two A         AA???       150
 *        Dead Two B      	??A?A??      250
 *        Dead Two C        ?A??A?       200
 */

// d:direction v:Relative order value of p（base p=0） p:current position
// __device__ int getLine(char* ai, char* taken, int x, int y, int d, int v) {
CUDA_CALLABLE_MEMBER int getLine(char* ai, char* taken, int x, int y, int d, int v) {

	switch (d) {
	case 1:
		x = x + v;
		break;
	case 2:
		x = x + v;
		y = y + v;
		break;
	case 3:
		y = y + v;
		break;
	case 4:
		x = x - v;
		y = y + v;
		break;
	case 5:
		x = x - v;
		break;
	case 6:
		x = x - v;
		y = y - v;
		break;
	case 7:
		y = y - v;
		break;
	case 8:
		x = x + v;
		y = y - v;
	}

	if (!(0 <= x && x < BOARD_SIZE && 0 <= y && y < BOARD_SIZE)) return -1;

	// 0 for empty, 1 for AI, 2 for Human
	if(!occupied(taken, x, y)) return 0;
	if(ai[x + BOARD_SIZE * y] == 1) return 1;
	return 2;
}

// calculate score based current model
// __device__ int getScore(char* ai, char* taken, int x, int y, Side evaluater, Side currentPlyer) {
CUDA_CALLABLE_MEMBER int getScore(char* ai, char* taken, int x, int y, Side evaluater, Side currentPlyer) {
	int value = 0;
	int numoftwo = 0;

	// code for both side
	int me = evaluater == AI ? 1 : 2;
	int plyer = currentPlyer == AI ? 1 : 2;

	// 8 direction
	for (int i = 1; i <= 8; i++) {
		// Live Four: 01111*, * stand for current empty,
		// 0 stand for other empty,the same as below
		if (getLine(ai, taken, x, y, i, -1) == plyer && getLine(ai, taken, x, y, i, -2) == plyer
		    && getLine(ai, taken, x, y, i, -3) == plyer && getLine(ai, taken, x, y, i, -4) == plyer
		    && getLine(ai, taken, x, y, i, -5) == 0) {

			value += 300000;
			if(me != plyer) {
				value -= 500;
			}
			continue;
		}

		// Dead Four A: 21111*
		if (getLine(ai, taken, x, y, i, -1) == plyer && getLine(ai, taken, x, y, i, -2) == plyer
		    && getLine(ai, taken, x, y, i, -3) == plyer && getLine(ai, taken, x, y, i, -4) == plyer
		    && (getLine(ai, taken, x, y, i, -5) == 3 - plyer || getLine(ai, taken, x, y, i, -5) == -1)) {

			value += 250000;
			if(me != plyer) {
				value -= 500;
			}
			continue;
		}

		// Dead Four B: 111*1
		if (getLine(ai, taken, x, y, i, -1) == plyer && getLine(ai, taken, x, y, i, -2) == plyer
		    && getLine(ai, taken, x, y, i, -3) == plyer && getLine(ai, taken, x, y, i, 1) == plyer) {

			value += 240000;
			if(me != plyer) {
				value -= 500;
			}
			continue;
		}

		// Dead Four C: 11*11
		if (getLine(ai, taken, x, y, i, -1) == plyer && getLine(ai, taken, x, y, i, -2) == plyer
		    && getLine(ai, taken, x, y, i, 1) == plyer && getLine(ai, taken, x, y, i, 2) == plyer) {

			value += 230000;
			if(me != plyer) {
				value -= 500;
			}
			continue;
		}

		// Live Three, near 3 position: 111*0
		if (getLine(ai, taken, x, y, i, -1) == plyer && getLine(ai, taken, x, y, i, -2) == plyer
		    && getLine(ai, taken, x, y, i, -3) == plyer) {

			if (getLine(ai, taken, x, y, i, 1) == 0) {
				value += 750;
				if (getLine(ai, taken, x, y, i, -4) == 0) {
					value += 3150;
					if(me != plyer) {
						value -= 300;
					}
				}
			}
			if ((getLine(ai, taken, x, y, i, 1) == 3 - plyer || getLine(ai, taken, x, y, i, 1) == -1)
			    && getLine(ai, taken, x, y, i, -4) == 0) {
				value += 500;
			}
			continue;
		}

		// Live Three, away 3 position: 1110*
		if (getLine(ai, taken, x, y, i, -1) == 0 && getLine(ai, taken, x, y, i, -2) == plyer
		    && getLine(ai, taken, x, y, i, -3) == plyer && getLine(ai, taken, x, y, i, -4) == plyer) {
			value += 350;
			continue;
		}

		// Dead Three: 11*1
		if (getLine(ai, taken, x, y, i, -1) == plyer && getLine(ai, taken, x, y, i, -2) == plyer
		    && getLine(ai, taken, x, y, i, 1) == plyer) {
			value += 600;
			if (getLine(ai, taken, x, y, i, -3) == 0 && getLine(ai, taken, x, y, i, 2) == 0) {
				value += 3150;
				continue;
			}
			if ((getLine(ai, taken, x, y, i, -3) == 3 - plyer||getLine(ai, taken, x, y, i, -3) == -1) && (getLine(ai, taken, x, y, i, 2) == 3 - plyer||getLine(ai, taken, x, y, i, 2) == -1)) {
				continue;
			} else {
				value += 700;
				continue;
			}
		}

		// Number of Live Two
		if (getLine(ai, taken, x, y, i, -1) == plyer && getLine(ai, taken, x, y, i, -2) == plyer
		    && getLine(ai, taken, x, y, i, -3) != 3 - plyer && getLine(ai, taken, x, y, i, 1) != 3 - plyer) {
			numoftwo++;
		}

		// Other gomoku modle
		int numOfplyer = 0;
		// ++++* +++*+ ++*++ +*+++ *++++
		for (int k = -4; k <= 0; k++) {
			int temp = 0;

			for (int l = 0; l <= 4; l++) {
				if (getLine(ai, taken, x, y, i, k + l) == plyer) {
					temp++;
				} else
				if (getLine(ai, taken, x, y, i, k + l) == 3 - plyer || getLine(ai, taken, x, y, i, k + l) == -1) {
					temp = 0;
					break;
				}
			}
			numOfplyer += temp;
		}
		value += numOfplyer * 15;
		if (numOfplyer != 0) {
			//
		}
	}

	if(numoftwo >= 2) {
		value += 3000;
		if(me != plyer) {
			value -= 100;
		}
	}
	//assert(value != 0);
	return value;
}

__device__ void doMove(char* ai, char* taken, int x, int y, Side side) {
	if(0 <= x && x < BOARD_SIZE && 0 <= y && y < BOARD_SIZE) {
		taken[x + BOARD_SIZE * y] = 1;
		if(side == AI) {
			ai[x + BOARD_SIZE * y] = 1;
		}
	}
}

__global__ void cudaSearchKernel(int leaf_x, int leaf_y, char* ai, char* taken, int* alpha, int* beta, Side side, Side evaluater, int depth) {
	if (depth == 0) {
		*alpha = getScore(ai, taken, leaf_x, leaf_y, evaluater, side);
		*beta = *alpha;
		return;
	}
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

	while (index < BOARD_SIZE * BOARD_SIZE) {
		int x = index % BOARD_SIZE;
		int y = index / BOARD_SIZE;
		Side oppositeSide = side == AI ? Human : AI;

		if (!occupied(taken, x, y)) {
			char *new_ai;
			char *new_taken;

			new_ai = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));
			new_taken = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));

			for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
				new_ai[i] = ai[i];
				new_taken[i] = taken[i];
			}
			doMove(new_ai, new_taken, x, y, oppositeSide);

			int *new_alpha = (int *) malloc(sizeof(int));
			int *new_beta = (int *) malloc(sizeof(int));
			*new_alpha = *alpha;
			*new_beta = *beta;

			// search child
			cudaSearchKernel<<<1, 32>>>(x, y, new_ai, new_taken, new_alpha, new_beta, oppositeSide, evaluater, depth - 1);
			cudaDeviceSynchronize();

			if (side == evaluater) {
				atomicMin(beta, *new_alpha);
			} else {
				atomicMax(alpha, *new_beta);
			}

			free(new_alpha);
			free(new_beta);
			free(new_ai);
			free(new_taken);

			if (*alpha >= *beta) {
				return;
			}
		}
		index += blockDim.x * gridDim.x;
	}
}

__global__ void cudaTreeKernel(Move* moves, char* ai, char* taken, int* values, Side side,
                               Side evaluater, int alpha, int beta, int depth) {
	// only one thread does high-level tasks
	if (threadIdx.x == 0) {
		// make one new node per block
		Move* move = new Move(moves[blockIdx.x].getX(), moves[blockIdx.x].getY());

		char* new_ai;
		char* new_taken;

		new_ai = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));
		new_taken = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));

		for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
			new_ai[i] = ai[i];
			new_taken[i] = taken[i];
		}

		// Board* newBoard = new Board(new_ai, new_taken);
		doMove(new_ai, new_taken, move->getX(), move->getY(), side);
		// Node* node = new Node(move, side, evaluater, newBoard);

		int *new_alpha = (int *) malloc(sizeof(int));
		int *new_beta = (int *) malloc(sizeof(int));
		*new_alpha = alpha;
		*new_beta = beta;

		cudaSearchKernel<<<1, 32>>>(move->getX(), move->getY(), new_ai, new_taken, new_alpha, new_beta, side, evaluater, depth);
		cudaDeviceSynchronize();

		// update the values we care about - if the parent node is a maximizing node,
		// it cares about the child alpha values
		if (side == evaluater) {
			values[blockIdx.x] = *new_beta;
		} else {
			values[blockIdx.x] = *new_alpha;
		}

		//delete move;
		free(new_ai);
		free(new_taken);
		free(new_alpha);
		free(new_beta);
	}
}

class parallelTreeGPU {
public:
Node* root;

protected:
Side evaluater;

public:
parallelTreeGPU(Board* board, Side evaluater) {
	// this is our side
	this->evaluater = evaluater;
	root = new Node(NULL, evaluater == AI ? Human : AI, evaluater, board);
}

Node* getRoot() {
	return root;
}

// PVS
Move* search(Node* startingNode, int depth) {
	if (depth == 0) {
		Move* move = startingNode->getMove();
		Side side = startingNode->getSide();
		// int value = startingNode->getBoard()->getScore(*move, evaluater, side);
		int value = (startingNode->getBoard()->getAI(), startingNode->getBoard()->getTaken(),
		             move->getX(), move->getY(), evaluater, side);

		startingNode->setAlpha(value);
		startingNode->setBeta(value);
		return NULL;
	}

	Board* board = startingNode->getBoard();
	Side oppositeSide = startingNode->getSide() == AI ? Human : AI;
	vector<Move> moves = board->getMoves(board);
	// printf("size = %d\n", moves.size());

	if (moves.size() == 0) {
		return NULL;
	}

	/* CPU search the first child node */
	Move* move = new Move(moves[0].getX(), moves[0].getY());
	Board* newBoard = board->copy();
	newBoard->doMove(move, oppositeSide);
	// change child player each recursion
	Node* child = new Node(move, oppositeSide, evaluater, newBoard);

	// pass alpha and beta values down
	child->setAlpha(startingNode->getAlpha());
	child->setBeta(startingNode->getBeta());

	// search child
	// Move* best = search(child, depth - 1);
	search(child, depth - 1);

	// array to store the values of interest of the children
	int* values;
	values = (int *)calloc(moves.size(), sizeof(int));

	if (startingNode->getSide() == evaluater) {
		startingNode->setBeta(min(startingNode->getBeta(), child->getAlpha()));
		values[0] = child->getAlpha();
	} else {
		startingNode->setAlpha(max(startingNode->getAlpha(), child->getBeta()));
		values[0] = child->getBeta();
	}

	delete child;

	/* GPU search the rest of the child nodes */
	int numMoves = moves.size() - 1;
	Move* dev_moves;
	Move* moves_ptr = &moves[1];
	char* dev_taken;
	char* dev_ai;
	int* dev_values;

	CopyData(board->ai, BOARD_SIZE * BOARD_SIZE, sizeof(char), &dev_ai);
	CopyData(board->taken, BOARD_SIZE * BOARD_SIZE, sizeof(char), &dev_taken);
	CopyData(moves_ptr, numMoves, sizeof(Move), &dev_moves);

	checkCudaErrors(cudaMalloc((void **) &dev_values, numMoves * sizeof(int)));
	checkCudaErrors(cudaMemset(dev_values, 0, numMoves * sizeof(int)));

	// call kernel to search the rest of the children in parallel
	// GPU execution parameters
	// 1 thread block per move
	// THREADS working on the same move

	unsigned int blocks = numMoves;
	unsigned int threads = THREADS;
	unsigned int shared = threads * sizeof(float);

	dim3 dimGrid(blocks, 1, 1);
	dim3 dimBlock(threads, 1, 1);
	cudaTreeKernel<<<dimGrid, dimBlock, shared>>>(dev_moves, dev_ai, dev_taken, dev_values, oppositeSide,
	                                              evaluater, startingNode->getAlpha(), startingNode->getBeta(), depth - 1);

	// copy remaining child values into host array
	checkCudaErrors(cudaMemcpy(values + 1, dev_values, numMoves * sizeof(int), cudaMemcpyDeviceToHost));

	// find the best move
	int index = 0;
	if (startingNode->getSide() == evaluater) {
		int best = INT_MAX;

		for (int i = 0; i <= numMoves; i++) {
			if (values[i] < best) {
				best = values[i];
				index = i;
			}
		}
		startingNode->setBeta(best);
	} else {
		int best = INT_MIN;

		for (int i = 0; i <= numMoves; i++) {
			if (values[i] > best) {
				best = values[i];
				index = i;
			}
		}
		startingNode->setAlpha(best);
	}

	cudaFree(dev_values);
	cudaFree(dev_taken);
	cudaFree(dev_ai);
	cudaFree(dev_moves);

	Move* curMove = new Move(moves[index].getX(), moves[index].getY());
	return curMove;
}
};

class Player {
private:
Side side;
Side opponent;
Board* board;

public:
Player(Side side) {
	this->side = side;
	this->opponent = side == AI ? Human : AI;
	this->board = new Board();
}

Move* doMove(Move* opponentsMove) {
	board->doMove(opponentsMove, opponent);

	if (!board->hasMoves()) {
		return NULL;
	}

	parallelTreeGPU* tree = new parallelTreeGPU(board, side);
	Move* moveToMake = tree->search(tree->getRoot(), DEPTH);
	board->doMove(moveToMake, side);
	return moveToMake;
}
};
int main() {
	// Run game on GPU here
	Board* board = new Board();
	Player* player1 = new Player(AI);
	Player* player2 = new Player(Human);

	Side turn = AI;
	string winner;

Move* m = NULL;

	// timers
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsedTime;

	cout << endl << "Starting GPU DP game..." << endl;
	checkCudaErrors(cudaEventRecord(start, 0));


    while (!board->isDone()) {
        // get the current player's move
        if (turn == AI) {
            m = player1->doMove(m);
        //printf("taken count = %d\n", board->countTaken());
            assert(m != NULL);
        }
        else {
            m = player2->doMove(m);
            assert(m != NULL);
        }

        // make move once it is determiend to be legal
        board->doMove(m, turn);

        // switch players
        if (turn == AI) {
            turn = Human;
        }
        else {
            turn = AI;
        }
    }
    board->printBoard();


/*
	Move* m = new Move(0,0);
	board->doMove(m, Human);
	Move* n = player1->doMove(m);
	board->doMove(n, AI);
	board->printBoard();
*/

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	if(turn == AI)
		winner = "O";
	else
		winner = "X";


	cout << "GPU Dynamic Parallelism Game completed." << endl << " elapsedTime = " << elapsedTime/1000 << " s" <<endl;
	cout << winner << " won! " << "Board Size: " << BOARD_SIZE << " * " << BOARD_SIZE << " Depth : "<< DEPTH <<endl;

	return 0;
}
