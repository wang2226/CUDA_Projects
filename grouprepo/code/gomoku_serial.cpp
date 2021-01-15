#include <iostream>
#include <vector>
#include <set>
#include <bits/stdc++.h>
#include "common.hpp"
#include <stdio.h>

using namespace std;


// Dimension of board
#define dimension 9

// Depth to search 
#define totalDepth 3

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
Board(){         // constructor
	for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			board[i * dimension + j] = 0;
		}
	}
}

Board(const Board &toBeCopied){         // Copy constructor
	for (int i = 0; i < dimension * dimension; i++) {
		board[i] = toBeCopied.board[i];
	}
}


/*
 * Helper function to print the board status
 */
void printBoard() {
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
void addStone(int color, int location_x, int location_y){
	board[location_x * dimension + location_y] = color;
}

/*
 * Remove the stone at given location
 */
void removeStone(int location_x, int location_y){
	board[location_x * dimension + location_y] = 0;
}

/*
 * Getter
 */
int getElement(int location_x, int location_y){
	return board[location_x * dimension + location_y];
}

/*
 * Setter
 */
void setBoard(int tempBoard[]){
	for (int i = 0; i < dimension*dimension; i++){
		board[i] = tempBoard[i];
	}
}

/* 
 * Get number of possible moves at this board state
 */
int getPossibleMovesCount(){
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
 * Get a list of current possible moves. Possible moves are defined to be those positions (i,j) that are not occupied
 */
move_t* getPossibleMoves(playedMoves_t played, int* sum){
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
int shapeScore(int countConsecutive, int openEnds, bool playersTurn){


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
score_t calculateScoreVertical(int currentPlayer, int evaluateFor, int columnNum){
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
score_t calculateScoreHorizontal(int currentPlayer, int evaluateFor, int rowNum){
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
score_t calculateScoreDiagonalLR(int currentPlayer, int evaluateFor){

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
score_t  calculateScoreDiagonalRL(int currentPlayer, int evaluateFor){

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
score_t calculateBoardScorePlayer(int currentPlayer, int evaluateFor){
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
score_t calculateBoardScoreTotal(playedMoves_t played, int currentPlayer){
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

	return ret;
}
};


class Minimax {
private:
Board board;
public:
  Minimax(Board miniMaxBoard){
	board = miniMaxBoard;
}
	
/*
 * Given some set of old played moves, make a new playedMoves_t object and add to this list
 */
playedMoves_t updatePlayedMoves(playedMoves_t oldPlayed, move_t nextMove, int currentPlayer){
	playedMoves_t newPlayed = {oldPlayed.movesNumOne, oldPlayed.movesNumTwo, NULL, NULL};
	// Allocate memory and add accordingly
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
 * Minimax function without alpha-beta pruning from the intial board with played moves 'played'
 */
int doMinimax(playedMoves_t played, int depth, bool maximizingPlayer, int currentPlayer, int maximizer){
	score_t status = board.calculateBoardScoreTotal(played, maximizer);
	// At a terminal node return
	if (depth == 0 || status.gameOver) {
		return status.score;
	}
	int movesCount;
	// For each possible move, recursively call minimax
	move_t* moves = board.getPossibleMoves(played, &movesCount);

	if (maximizingPlayer) {
		int maxEval = INT_MIN;
		for (int i = 0; i < movesCount; i++) {
			move_t move = moves[i];
			playedMoves_t newPlayed = updatePlayedMoves(played, move, currentPlayer);
			int eval = doMinimax(newPlayed, depth - 1, false, (currentPlayer == 1) ? 2 : 1, maximizer);
			maxEval = max(maxEval, eval);
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
			int eval = doMinimax(newPlayed, depth - 1, true, (currentPlayer == 1) ? 2 : 1,maximizer);
			minEval= min(minEval, eval);
			free(newPlayed.movesOne);
			free(newPlayed.movesTwo);
		}
		free(moves);
		return minEval;
	}
}



/*
 * Minimax function with alpha-beta pruning from the intial board with played moves 'played'
 */
int doMinimaxAB(playedMoves_t played, int depth, bool maximizingPlayer, int currentPlayer, int maximizer, int alpha, int beta){
	score_t status = board.calculateBoardScoreTotal(played, maximizer);
	// At a terminal node return
	if (depth == 0 || status.gameOver) {
		return status.score;
	}
	int movesCount;
	// For each possible move, recursively call minimax
	move_t* moves = board.getPossibleMoves(played, &movesCount);

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


/*
 * From the initial board state get the best move without alpha-beta pruning
 */
move_t getBestMove(int depth, bool maximizingPlayer, int currentPlayer){
	playedMoves_t played = {0, 0, NULL, NULL};
	int movesCount;
	move_t* moves = board.getPossibleMoves(played, &movesCount);
	printf("Size of options: %d\n", movesCount);
	for (int i = 0; i<movesCount; i++) {
		cout << moves[i].x << ", " << moves[i].y << endl;
	}
	int bestVal = INT_MIN;
	move_t bestMove = moves[0];
	for (int i = 0; i < movesCount; i++) {
		move_t move = moves[i];
		playedMoves_t newPlayed = updatePlayedMoves(played, move, currentPlayer);
		int eval = doMinimax(newPlayed, depth - 1, !maximizingPlayer, (currentPlayer == 1) ? 2 : 1, currentPlayer);
		printf("%d, %d Score: %d\n", move.x, move.y, eval);
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


/*
 * From the initial board state get the best move with alpha-beta pruning
 */
move_t getBestMoveAB(int depth, bool maximizingPlayer, int currentPlayer){
	playedMoves_t played = {0, 0, NULL, NULL};
	int movesCount;
	move_t* moves = board.getPossibleMoves(played, &movesCount);
	printf("Size of options: %d\n", movesCount);
	/* for (int i = 0; i<movesCount; i++) { */
	/* 	cout << moves[i].x << ", " << moves[i].y << endl; */
	/* } */
	int bestVal = INT_MIN;
	move_t bestMove = moves[0];
	for (int i = 0; i < movesCount; i++) {
		move_t move = moves[i];
		playedMoves_t newPlayed = updatePlayedMoves(played, move, currentPlayer);
		int eval = doMinimaxAB(newPlayed, depth - 1, !maximizingPlayer, (currentPlayer == 1) ? 2 : 1, currentPlayer, INT_MIN, INT_MAX);
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
};


int main() {
	// Delclare a new board, add a stone and calculate time to get the next move
	Board board;
	board.addStone(1, 0, 0);
	board.printBoard();

	uint64_t start_t;
	uint64_t end_t;
	InitTSC();

	Minimax minimax(board);
	start_t = ReadTSC();
	move_t device_bestMove = minimax.getBestMoveAB(totalDepth, true, 2);
	end_t = ReadTSC();
	board.printBoard();
	printf("Time to run Minimax is %g\n", ElapsedTime(end_t - start_t));
	cout << "Best move: " << device_bestMove.x << ", " << device_bestMove.y << endl; 
	return 0;
}
