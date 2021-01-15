#include <iostream>
#include <vector>
#include <set>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "common.hpp"
#include <helper_cuda.h>
#include <stdio.h>
using namespace std;

// Dimension of board
#define dimension 9

// Define the total depth and the sequential depth to be searched
#define totalDepth 3
#define sequentialDepth 2

// Structs
typedef struct {
	int x;
	int y;
} move_t;

typedef struct {
	int score;
	bool gameOver;
} score_t;

typedef struct {
	int movesNumOne;
	int movesNumTwo;
	move_t* movesOne;
	move_t* movesTwo;
} playedMoves_t;


class Board {
public:
int board[dimension * dimension];         // the board 0 for empty, 1 for black and 2 for white
__host__ __device__ Board(){         // constructor
	for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			board[i * dimension + j] = 0;
		}
	}
}

__host__ __device__ Board(const Board &toBeCopied){         // Copy constructor
	for (int i = 0; i < dimension * dimension; i++) {
		board[i] = toBeCopied.board[i];
	}
}


/*
 * Helper function to print the board status
 */
__host__ __device__ void printBoard() {
	for (int i = 0; i < dimension; i++) {
		printf("%d: ", i);
		for (int j = 0; j < dimension; j++) {
			printf("%d ", board[i * dimension + j]);
		}
		printf("\n");
	}
	printf("\n");
}

/*
 * Add a stone of 'color' to the location
 */
__host__ __device__ void addStone(int color, int location_x, int location_y){
	board[location_x * dimension + location_y] = color;
}

/*
 * Remove the stone at given location
 */
 __host__ __device__  void removeStone(int location_x, int location_y){
	board[location_x * dimension + location_y] = 0;
}

/*
 * Getter
 */
__host__ __device__ int getElement(int location_x, int location_y){
	return board[location_x * dimension + location_y];
}

/*
 * Setter
 */
__host__ __device__ void setBoard(int tempBoard[]){
	for (int i = 0; i < dimension*dimension; i++){
		board[i] = tempBoard[i];
	}
}

/*
 * Get the number of possible moves at this board state
 */
__host__ __device__ int getPossibleMovesCount(){
	int sum = 0;
	for (int i = 0; i < dimension; i++){
		for (int j = 0; j < dimension; j++){
			if (board[i * dimension + j] == 0){
				sum++;
			}
		}
	}
	return sum;
}
/*
 * Get a list of current possible moves. Possible moves are defined to be those positions (i,j) such that any adjacent cells, including diagonals have a stone placed
 */
 __host__ __device__ move_t* getPossibleMoves(playedMoves_t played, int* sum){
	Board tempBoard;
	tempBoard.setBoard(board);
	for (int oneMoves = 0; oneMoves < played.movesNumOne; oneMoves++){
		tempBoard.addStone(1, played.movesOne[oneMoves].x, played.movesOne[oneMoves].y);
	}
	for (int twoMoves = 0; twoMoves < played.movesNumTwo; twoMoves++){
		tempBoard.addStone(2, played.movesTwo[twoMoves].x, played.movesTwo[twoMoves].y);
	}
	int moveCount = tempBoard.getPossibleMovesCount();
	*sum = moveCount;
	move_t* moves = (move_t*) malloc(sizeof(move_t) * tempBoard.getPossibleMovesCount());
	int index = 0;

	for (int i = 0; i < dimension; i++){
		for (int j = 0; j < dimension; j++){
			if (tempBoard.getElement(i, j) == 0){
				moves[index].x = i;
				moves[index].y = j;
				index++;
			}
		}
	}
	
	return moves;
}


/*
 * Function that gives a score based on the number of consecutive stones, whether they have open ends and whose turn it currently is. To be called by the other heuristic functions to calculate score
 */
 __host__ __device__ int shapeScore(int countConsecutive, int openEnds, bool playersTurn){


	if (openEnds == 0 && countConsecutive < 5) {
		return 0;
	}
	switch(countConsecutive) {
	case 4:
		switch (openEnds) {
		case 1:
			if (playersTurn) {
				return 100000000;
			} else{
				return 50;
			}
		case 2:
			if (playersTurn) {
				return 100000000;
			} else{
				return 500000;
			}
		}
	case 3:
		switch (openEnds) {
		case 1:
			if (playersTurn) {
				return 7;
			} else{
				return 5;
			}
		case 2:
			if (playersTurn) {
				return 10000;
			} else{
				return 50;
			}
		}
	case 2:
		switch (openEnds) {
		case 1:
			return 3;
		case 2:
			return 5;
		}
	case 1:
		switch (openEnds) {
		case 1:
			return 1;
		case 2:
			return 2;
		}
	default:
		return 200000000;
	}
	return 0;
}

/*
 * Using the shapreScore function, calculate the score for the player 'evaluateFor' and at column 'columnNum' given that the current player is 'currentPlayer'
 */
 __host__ __device__ score_t calculateScoreVertical(int currentPlayer, int evaluateFor, int columnNum){
	int score = 0;
	int openEnds = 0;
	int consecutiveCount = 0;
	bool gameOver = false;

	for (int i = 0; i < dimension; i++) {
		if (board[i * dimension + columnNum] == evaluateFor) {        // If the color is the one we are counting, increment
			consecutiveCount++;
		} else if (board[i * dimension + columnNum] == 0 && consecutiveCount > 0) {        // If the cell is empty and theres been more than one consecutive, increment openends and restart counters
			openEnds++;
			score += shapeScore(consecutiveCount, openEnds, evaluateFor == currentPlayer);
			consecutiveCount = 0;
			openEnds = 1;
		} else if (board[i * dimension + columnNum] == 0) {        // If the cell is empty and no consecutive
			openEnds = 1;
		} else if (consecutiveCount > 0) {        //  If there's been more than one consecutive but a dead end
			score += shapeScore(consecutiveCount, openEnds, evaluateFor == currentPlayer);
			consecutiveCount = 0;
			openEnds = 0;
		} else {         // Just the opposite color with no consecutive
			openEnds=0;
		}
		if (consecutiveCount >= 5) {
			gameOver = true;
		}
	}
	if (consecutiveCount > 0) {        // Account for consecutive ending at the last cell
		score += shapeScore(consecutiveCount, openEnds, evaluateFor == currentPlayer);
	}

	score_t ret = {score, gameOver};
	return ret;

}

/*
 * Using the shapreScore function, calculate the score for the player 'evaluateFor' and at row 'rowNum' given that the current player is 'currentPlayer'
 */
 __host__ __device__ score_t calculateScoreHorizontal(int currentPlayer, int evaluateFor, int rowNum){
	int score = 0;
	int openEnds = 0;
	int consecutiveCount = 0;
	bool gameOver = false;

	for (int i = 0; i < dimension; i++) {
		if (board[rowNum * dimension + i] == evaluateFor) {        // If the color is the one we are counting, increment
			consecutiveCount++;
		} else if (board[rowNum * dimension + i] == 0 && consecutiveCount > 0) {        // If the cell is empty and theres been more than one consecutive, increment openends and restart counters
			openEnds++;
			score += shapeScore(consecutiveCount, openEnds, evaluateFor == currentPlayer);
			consecutiveCount = 0;
			openEnds = 1;
		} else if (board[rowNum * dimension + i] == 0) {        // If the cell is empty and no consecutive
			openEnds = 1;
		} else if (consecutiveCount > 0) {        //  If there's been more than one consecutive but a dead end
			score += shapeScore(consecutiveCount, openEnds, evaluateFor == currentPlayer);
			consecutiveCount = 0;
			openEnds = 0;
		} else {         // Just the opposite color with no consecutive
			openEnds=0;
		}
		if (consecutiveCount >= 5) {
			gameOver = true;
		}
	}
	if (consecutiveCount > 0) {        // Account for consecutive ending at the last cell
		score += shapeScore(consecutiveCount, openEnds, evaluateFor == currentPlayer);
	}

	score_t ret = {score, gameOver};
	return ret;
}


/*
 * Using the shapreScore function, calculate the score for the player 'evaluateFor' and at row 'rowNum' given that the current player is 'currentPlayer' at the diagonal
 */
__host__ __device__ score_t calculateScoreDiagonalLR(int currentPlayer, int evaluateFor){

	int score = 0;
	int openEnds = 0;
	int consecutiveCount = 0;
	bool gameOver = false;

	for (int i = 0; i < 2 * dimension - 1; i++) {
		consecutiveCount = 0;
		openEnds = 0;

		int z = (i < dimension) ? 0 : i - dimension + 1;
		for (int j = z; j <= i - z; j++) {
			if (board[j * dimension + (i - j)] == evaluateFor) {        // If the color is the one we are counting, increment
				consecutiveCount++;
			} else if (board[j * dimension + (i - j)] == 0 && consecutiveCount > 0) {        // If the cell is empty and theres been more than one consecutive, increment openends and restart counters
				openEnds++;
				score += shapeScore(consecutiveCount, openEnds, evaluateFor == currentPlayer);
				consecutiveCount = 0;
				openEnds = 1;
			} else if (board[j * dimension + (i - j)] == 0) {        // If the cell is empty and no consecutive
				openEnds = 1;
			} else if (consecutiveCount > 0) {        //  If there's been more than one consecutive but a dead end
				score += shapeScore(consecutiveCount, openEnds, evaluateFor == currentPlayer);
				consecutiveCount = 0;
				openEnds = 0;
			} else {         // Just the opposite color with no consecutive
				openEnds=0;
			}
			if (consecutiveCount >= 5) {
				gameOver = true;
			}
		}
		if (consecutiveCount > 0) {        // Account for consecutive ending at the last cell
			score += shapeScore(consecutiveCount, openEnds, evaluateFor == currentPlayer);
		}
	}

	score_t ret = {score, gameOver};
	return ret;
}

/*
 * Using the shapreScore function, calculate the score for the player 'evaluateFor' and at row 'rowNum' given that the current player is 'currentPlayer' at the diagonal
 */
__host__ __device__ score_t  calculateScoreDiagonalRL(int currentPlayer, int evaluateFor){

	int score = 0;
	int openEnds = 0;
	int consecutiveCount = 0;
	bool gameOver = false;

	for (int i = 0; i < 2 * dimension - 1; i++) {
		consecutiveCount = 0;
		openEnds = 0;

		int z = (i < dimension) ? 0 : i - dimension + 1;
		for (int j = z; j <= i - z; j++) {
			if (board[j * dimension + (dimension-1-i+j)] == evaluateFor) {        // If the color is the one we are counting, increment
				consecutiveCount++;
			} else if (board[j * dimension + (dimension-1-i+j)] == 0 && consecutiveCount > 0) {        // If the cell is empty and theres been more than one consecutive, increment openends and restart counters
				openEnds++;
				score += shapeScore(consecutiveCount, openEnds, evaluateFor == currentPlayer);
				consecutiveCount = 0;
				openEnds = 1;
			} else if (board[j * dimension + (dimension-1-i+j)] == 0) {        // If the cell is empty and no consecutive
				openEnds = 1;
			} else if (consecutiveCount > 0) {        //  If there's been more than one consecutive but a dead end
				score += shapeScore(consecutiveCount, openEnds, evaluateFor == currentPlayer);
				consecutiveCount = 0;
				openEnds = 0;
			} else {         // Just the opposite color with no consecutive
				openEnds=0;
			}
			if (consecutiveCount >= 5) {
				gameOver = true;
			}
		}
		if (consecutiveCount > 0) {        // Account for consecutive ending at the last cell
			score += shapeScore(consecutiveCount, openEnds, evaluateFor == currentPlayer);
		}
	}
	score_t ret = {score, gameOver};
	return ret;
}

/*
 * Calcuate the whole board score for player 'evaluateFor' given that the current player is 'currentPlayer'
 */
 __host__ __device__ score_t calculateBoardScorePlayer(int currentPlayer, int evaluateFor){
	int score = 0;
	bool gameOver = false;
	for (int i = 0; i < dimension; i++) {
		score_t horizontal = calculateScoreHorizontal(currentPlayer, evaluateFor, i);
		score_t vertical = calculateScoreVertical(currentPlayer, evaluateFor, i);
		score += horizontal.score + vertical.score;
		gameOver = gameOver || horizontal.gameOver || vertical.gameOver;
	}
	score_t diagonalLR = calculateScoreDiagonalLR(currentPlayer, evaluateFor);
	score_t diagonalRL = calculateScoreDiagonalRL(currentPlayer, evaluateFor);
	score += diagonalLR.score + diagonalRL.score;
	gameOver = gameOver || diagonalLR.gameOver || diagonalRL.gameOver;
	score_t ret = {score, gameOver};
	return ret;
}
/*
 * Calculate the whole board score for the current player. Defined to be the difference between the score evaluated for the current player and the score evaluated for the opponent
 */
 __host__ __device__ score_t calculateBoardScoreTotal(playedMoves_t played, int currentPlayer){
	Board tempBoard;
	tempBoard.setBoard(board);
    for (int oneMoves = 0; oneMoves < played.movesNumOne; oneMoves++){
	 	tempBoard.addStone(1, played.movesOne[oneMoves].x, played.movesOne[oneMoves].y);
	}
	for (int twoMoves = 0; twoMoves < played.movesNumTwo; twoMoves++){
		tempBoard.addStone(2, played.movesTwo[twoMoves].x, played.movesTwo[twoMoves].y);
	}
	score_t cur = tempBoard.calculateBoardScorePlayer(currentPlayer, currentPlayer);
	score_t other = tempBoard.calculateBoardScorePlayer(currentPlayer, (currentPlayer == 1) ? 2 : 1);
	score_t ret = {cur.score - other.score, cur.gameOver || other.gameOver};
	// score_t ret = {0, false};

	return ret;
}
};

__global__ void minimaxKernel(Board* device_currentBoard, move_t* device_move_array, int n, move_t* device_bestMove, int* score, int depth, bool maximizingPlayer, int currentPlayer);
__global__ void minimaxKernelSeqPar(Board* device_boards, int* device_scores, int leavesCount, int parDepth, int currentPlayer, bool maximizingPlayer);



class Minimax {
private:
Board board;
public:
__host__ __device__ Minimax(Board miniMaxBoard){
	board = miniMaxBoard;
}
	
/*
 * Given some set of old moves, and a new move, make a new playedMoves object, add to that and return
 */
__host__ __device__ playedMoves_t updatePlayedMoves(playedMoves_t oldPlayed, move_t nextMove, int currentPlayer){
	playedMoves_t newPlayed = {oldPlayed.movesNumOne, oldPlayed.movesNumTwo, NULL, NULL};
	// Allocate memory, copy stuff and return
	if (currentPlayer == 1){
		newPlayed.movesNumOne = oldPlayed.movesNumOne+1;
		newPlayed.movesOne = (move_t*) malloc(sizeof(move_t) * newPlayed.movesNumOne);
		newPlayed.movesTwo = (move_t*) malloc(sizeof(move_t) * newPlayed.movesNumTwo);
		for (int i = 0; i < oldPlayed.movesNumOne; i++){
			newPlayed.movesOne[i].x = oldPlayed.movesOne[i].x;
			newPlayed.movesOne[i].y = oldPlayed.movesOne[i].y;
		}
		for (int j = 0; j < oldPlayed.movesNumTwo; j++){
			newPlayed.movesTwo[j].x = oldPlayed.movesTwo[j].x;
			newPlayed.movesTwo[j].y = oldPlayed.movesTwo[j].y;
		}
		newPlayed.movesOne[newPlayed.movesNumOne - 1].x = nextMove.x;
		newPlayed.movesOne[newPlayed.movesNumOne - 1].y = nextMove.y;
	} else {
		newPlayed.movesNumTwo = oldPlayed.movesNumTwo+1;
		newPlayed.movesOne = (move_t*) malloc(sizeof(move_t) * newPlayed.movesNumOne);
		newPlayed.movesTwo = (move_t*) malloc(sizeof(move_t) * newPlayed.movesNumTwo);
		for (int i = 0; i < oldPlayed.movesNumOne; i++){
			newPlayed.movesOne[i].x = oldPlayed.movesOne[i].x;
			newPlayed.movesOne[i].y = oldPlayed.movesOne[i].y;
		}
		for (int j = 0; j < oldPlayed.movesNumTwo; j++){
			newPlayed.movesTwo[j].x = oldPlayed.movesTwo[j].x;
			newPlayed.movesTwo[j].y = oldPlayed.movesTwo[j].y;
		}
		newPlayed.movesTwo[newPlayed.movesNumTwo - 1].x = nextMove.x;
		newPlayed.movesTwo[newPlayed.movesNumTwo - 1].y = nextMove.y;
	}
	return newPlayed;
}


/*
 * Do minimax with alpha-beta pruning from the initial board state with playedMoves_t 'played'
 */
__host__ __device__ int doMinimaxAB(playedMoves_t played, int depth, bool maximizingPlayer, int currentPlayer, int maximizer, int alpha, int beta){
	score_t status = board.calculateBoardScoreTotal(played, maximizer);
	// At a terminal node, return the evaluation
	if (depth == 0 || status.gameOver) {
		return status.score;
	}
	int movesCount;
	move_t* moves = board.getPossibleMoves(played, &movesCount);

	// recursively call minimax for each child move
	if (maximizingPlayer) {
		int maxEval = INT_MIN;
		for (int i = 0; i < movesCount; i++) {
			move_t move = moves[i];
			playedMoves_t newPlayed = updatePlayedMoves(played, move, currentPlayer);
			int eval = doMinimaxAB(newPlayed, depth - 1, false, (currentPlayer == 1) ? 2 : 1, maximizer, alpha, beta);
			maxEval = max(maxEval, eval);
			alpha = max(alpha, eval);
			if (beta <= alpha) {
				break;
			}
			free(newPlayed.movesOne);
			free(newPlayed.movesTwo);
		}
		free(moves);
		return maxEval;
	} else{
		int minEval = INT_MAX;
		for (int i = 0; i < movesCount; i++) {
			move_t move = moves[i];
			playedMoves_t newPlayed = updatePlayedMoves(played, move, currentPlayer);
			int eval = doMinimaxAB(newPlayed, depth - 1, true, (currentPlayer == 1) ? 2 : 1,maximizer, alpha, beta);
			minEval= min(minEval, eval);
			beta = min(beta, eval);
			if (beta <= alpha) {
				break;
			}
			free(newPlayed.movesOne);
			free(newPlayed.movesTwo);
		}
		free(moves);
		return minEval;
	}
}




// Get all the leaf boards at the sequential depth using the recursive helper function
Board* getLeafBoards(int seqDepth, int currentPlayer, int leavesCount){
	Board* ret = (Board*) malloc(sizeof(Board) * leavesCount);
	/* cout << leavesCount << endl; */
	getLeafBoardsHelper(ret, board, seqDepth, currentPlayer, 0);
	return ret;
}

// Recursively add the terminal boards to the 'boards' list
int getLeafBoardsHelper(Board* boards, Board currentBoard, int seqDepth, int currentPlayer, int curIndex){
	if (seqDepth == 0){	
		boards[curIndex] = currentBoard;
		return curIndex+1;
	} else {
		Board tempBoard = currentBoard;
		int movesCount;
		playedMoves_t played = {0, 0, NULL, NULL};
		move_t* moves = tempBoard.getPossibleMoves(played, &movesCount);
		for (int i = 0; i < movesCount; i++){
			move_t move = moves[i];
			tempBoard.addStone(currentPlayer, move.x, move.y);
			curIndex = getLeafBoardsHelper(boards, tempBoard, seqDepth-1, (currentPlayer == 1) ? 2 : 1, curIndex);
			tempBoard.removeStone(move.x, move.y);
		}
		return curIndex;
	}
}

// Gather the result from the GPU computation in the CPU by tracing down the game tree
pair<int, int> getMinimaxSeqParAfterGPU(int* scores, playedMoves_t played, int depth, bool maximizingPlayer, int currentPlayer, int maximizer, int curIndex){
	if (depth == 0) {
		// At a terminal node, instead of calculating the score, just look up on the scores array
		int ret = scores[curIndex];
		return make_pair(ret, curIndex + 1);
	}
	int movesCount;
	move_t* moves = board.getPossibleMoves(played, &movesCount);

	if (maximizingPlayer) {
		int maxEval = INT_MIN;
		for (int i = 0; i < movesCount; i++) {
			move_t move = moves[i];
			playedMoves_t newPlayed = updatePlayedMoves(played, move, currentPlayer);
			pair<int, int> result = getMinimaxSeqParAfterGPU(scores, newPlayed, depth - 1, false, (currentPlayer == 1) ? 2 : 1, maximizer, curIndex);	
			int eval = result.first;
			curIndex = result.second;
			maxEval = max(maxEval, eval);
			free(newPlayed.movesOne);
			free(newPlayed.movesTwo);
		}
		free(moves);
		return make_pair(maxEval, curIndex);
	} else{
		int minEval = INT_MAX;
		for (int i = 0; i < movesCount; i++) {
			move_t move = moves[i];
			playedMoves_t newPlayed = updatePlayedMoves(played, move, currentPlayer);
			pair<int, int> result = getMinimaxSeqParAfterGPU(scores, newPlayed, depth - 1, true, (currentPlayer == 1) ? 2 : 1, maximizer, curIndex);
			int eval = result.first;
			curIndex = result.second;
			minEval= min(minEval, eval);
			free(newPlayed.movesOne);
			free(newPlayed.movesTwo);
		}
		free(moves);
		return make_pair(minEval, curIndex);
	}
}


// Get the best move by gathering the result from the GPU computation in the CPU by tracing down the game tree
move_t retrieveBestMoveAfterGPU(int depth, bool maximizingPlayer, int* host_scores, int currentPlayer){
	
	playedMoves_t played = {0, 0, NULL, NULL};
	int movesCount;
	move_t* moves = board.getPossibleMoves(played, &movesCount);
	printf("Size of options: %d\n", movesCount);
	/* for (int i = 0; i<movesCount; i++) { */
	/* 	cout << moves[i].x << ", " << moves[i].y << endl; */
	/* } */
	int bestVal = INT_MIN;
	move_t bestMove = moves[0];
	int curIndex = 0;
	for (int i = 0; i < movesCount; i++) {
		move_t move = moves[i];
		playedMoves_t newPlayed = updatePlayedMoves(played, move, currentPlayer);
		// Call the getMinimaxSeqParAfterGPU function to retrieve the results
		pair<int, int> result = getMinimaxSeqParAfterGPU(host_scores, newPlayed, depth - 1, !maximizingPlayer, (currentPlayer == 1) ? 2 : 1, currentPlayer, curIndex);
		int eval = result.first;
		curIndex = result.second;
		// cout << curIndex << endl;
		/* printf("%d, %d Score: %d\n", move.x, move.y, eval); */
		if (eval > bestVal) {
			bestVal = eval;
			bestMove = move;
		}
		free(newPlayed.movesOne);
		free(newPlayed.movesTwo);
	}
	printf("Best score: %d\n", bestVal);
	printf("Best move %d : %d\n", bestMove.x, bestMove.y);
	free(moves);
	return bestMove;
}

// On the GPU get the best move using sequential parallel GPU implementation
move_t* getBestMoveABSeqPar(int seqDepth, int depth, bool maximizingPlayer, int currentPlayer){
	int parDepth = depth - seqDepth;
	int curOptions = board.getPossibleMovesCount();
	int leavesCount = curOptions;
	for (int i = 0; i < seqDepth - 1; i++){
		leavesCount *= (curOptions - (i+1));
	}
	/* cout << "leavesCount: " << leavesCount << endl; */
	Board* host_boards = getLeafBoards(seqDepth, currentPlayer, leavesCount);
	int* host_scores = (int*) malloc(sizeof(int) * leavesCount);

	Board* device_boards;
	int* device_scores;
	
	// Copy over the data to device
	/* printf("%ld\n", sizeof(Board)*leavesCount); */
	if (cudaMalloc(&device_boards, sizeof(Board) * leavesCount) != cudaSuccess) {
		fprintf(stderr, "Error: %s cudaMalloc at line %d in function %s\n", cudaGetErrorString(cudaGetLastError()), (__LINE__), (__func__));
	}
	cudaError_t err = cudaMemcpy(device_boards, host_boards, sizeof(Board)*leavesCount, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		printf("Error: %s cudaMemcpy at line %d\n", cudaGetErrorString(err), (__LINE__));
	}

	// checkCudaErrors(cudaMalloc((void**) &device_scores, sizeof(int) * leavesCount));
	if (cudaMalloc(&device_scores, sizeof(int) * leavesCount) != cudaSuccess) {
		fprintf(stderr, "Error: cudaMalloc at line %d in function %s\n", (__LINE__), (__func__));
	}

	int currentPlayerGPU = currentPlayer;
	if (seqDepth % 2 == 1){
		currentPlayerGPU = (currentPlayer == 1) ? 2 : 1;
	}
	// Declare thread block size etc. and launch threads
	int num_threads_per_block = 128;
	int num_blocks =  (leavesCount + num_threads_per_block - 1) / num_threads_per_block;
	minimaxKernelSeqPar<<<num_blocks, num_threads_per_block>>>(device_boards, device_scores, leavesCount, parDepth, currentPlayerGPU, seqDepth % 2 == 0);
	// cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("Error: %s kernel launch at line %d\n", cudaGetErrorString(err), (__LINE__));
	}
	move_t* host_bestMove = (move_t*) malloc(sizeof(move_t));
	// get_result_gpu(device_scores, host_scores, leavesCount);
	err = cudaMemcpy(host_scores, device_scores, sizeof(int) * leavesCount, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		printf("Error: %s cudaMemcpy launch at line %d\n", cudaGetErrorString(err), (__LINE__));
	}
	*host_bestMove =  retrieveBestMoveAfterGPU(seqDepth, maximizingPlayer, host_scores, currentPlayer);
	free(host_boards);
	free(host_scores);
	cudaFree(device_boards);
	cudaFree(device_scores);
	return host_bestMove;
}
};

// GPU kernel to do parallel minimax search
__global__ void minimaxKernelSeqPar(Board* device_boards, int* device_scores, int leavesCount, int parDepth, int currentPlayer, bool maximizingPlayer){
	int tid  = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < leavesCount){
		Minimax minimax(device_boards[tid]);
		playedMoves_t played = {0, 0, NULL, NULL};
		int maximizer = currentPlayer;
		if (!maximizingPlayer){ 
			maximizer = (currentPlayer == 1) ? 2 : 1;
		}
		int score = minimax.doMinimaxAB(played, parDepth, maximizingPlayer, currentPlayer, maximizer, INT_MIN, INT_MAX);
		// int score = 0;
		device_scores[tid] = score;
		// printf("%d : Score: %d\n", tid, score);
	}
}



int main() {

	cudaDeviceSetLimit(cudaLimitStackSize, 30000);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 100000000);
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	/* printf("%ld, %ld\n", free, total); */
	
	Board board;
	board.addStone(1, 0, 0);
	// board.addStone(1, 2,3); 
	// board.addStone(1, 1,4); 
	// board.addStone(2, 0,4); 
	// board.addStone(1, 4,4); 
	// board.addStone(2, 4,0); 
	// board.addStone(2, 4, 1); 
	board.printBoard();

	uint64_t start_t;
	uint64_t end_t;
	InitTSC();


	Minimax minimax(board);
	start_t = ReadTSC();
	move_t* device_bestMove = minimax.getBestMoveABSeqPar(sequentialDepth, totalDepth, true, 2);
	end_t = ReadTSC();
	board.printBoard();
	printf("Time to run Minimax is %g\n", ElapsedTime(end_t - start_t));
	cout << "Best move: " << (*device_bestMove).x << ", " << (*device_bestMove).y << endl;

    	/* free(device_bestMove); */
	return 0;
}
