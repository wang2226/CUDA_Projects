#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <bits/stdc++.h>
#include <stdio.h>
#include <time.h>

using namespace std;

// Dimension of board
#define BOARD_SIZE 8
#define DEPTH 2

enum Side {
	AI, Human
};

class Move {
public:
int x, y;

Move(int x, int y) {
	this->x = x;
	this->y = y;
}

int getX() {
	return this->x;
}
int getY() {
	return this->y;
}
void setX(int x) {
	this->x = x;
}
void setY(int y) {
	this->y = y;
}
};

class Board {
private:
/* Add a stone to the location */
void addStone(Side side, int x, int y) {
	if(onBoard(x, y)) {
		taken[x + BOARD_SIZE * y] = 1;
		if(side == AI) {
			ai[x + BOARD_SIZE * y] = 1;
		}
	}
}

bool onBoard(int x, int y) {
	return(0 <= x && x < BOARD_SIZE && 0 <= y && y < BOARD_SIZE);
}

public:
char* taken;
char* ai;

// constructor
Board(){
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

Board(char *ai, char *taken) {
	this->ai = ai;
	this->taken = taken;
}

/* Destructor for the board. */
~Board() {
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

bool occupied(int x, int y) {
	if (onBoard(x, y)) {
		return taken[x + BOARD_SIZE * y] == 1;
	}
	return false;
}

/* Current count of stones. */
int countTaken() {
	int count = 0;
	for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
		count += taken[i];
	}
	return count;
}

// 0 for empty, 1 for AI, 2 for Human
int getPlayerCode(int x, int y) {
	if(!occupied(x, y))
		return 0;
	if(ai[x + BOARD_SIZE * y] == 1)
		return 1;
	return 2;
}

/* Returns true if there are legal moves left. */
bool hasMoves() {
	for (int i = 0; i < BOARD_SIZE; i++) {
		for (int j = 0; j < BOARD_SIZE; j++)
			if (!occupied(i, j)) return true;
	}
	return false;
}

/*
 * Returns true if the game is finished; false otherwise. The game is finished
 * if neither side has a legal move.
 */
bool isDone() {
	if (checkDone(AI) || checkDone(Human) || !hasMoves()) {
		return true;
	}
	return false;
}

bool checkDone(Side side) {
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
int getLine(Move m, int d, int v) {
	int x = m.getX();
	int y = m.getY();

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

	if (!onBoard(x, y)) return -1;

	return getPlayerCode(x, y);
}

// calculate score based current model
int getScore(Move p, Side evaluater, Side currentPlyer) {
	int value = 0;
	int numoftwo = 0;

	// code for both side
	int me = evaluater == AI ? 1 : 2;
	int plyer = currentPlyer == AI ? 1 : 2;

	// 8 direction
	for (int i = 1; i <= 8; i++) {
		// Live Four: 01111*, * stand for current empty,
		// 0 stand for other empty,the same as below
		if (getLine(p, i, -1) == plyer && getLine(p, i, -2) == plyer
		    && getLine(p, i, -3) == plyer && getLine(p, i, -4) == plyer
		    && getLine(p, i, -5) == 0) {

			value += 300000;
			if(me != plyer) {
				value -= 500;
			}
			continue;
		}

		// Dead Four A: 21111*
		if (getLine(p, i, -1) == plyer && getLine(p, i, -2) == plyer
		    && getLine(p, i, -3) == plyer && getLine(p, i, -4) == plyer
		    && (getLine(p, i, -5) == 3 - plyer || getLine(p, i, -5) == -1)) {

			value += 250000;
			if(me != plyer) {
				value -= 500;
			}
			continue;
		}

		// Dead Four B: 111*1
		if (getLine(p, i, -1) == plyer && getLine(p, i, -2) == plyer
		    && getLine(p, i, -3) == plyer && getLine(p, i, 1) == plyer) {

			value += 240000;
			if(me != plyer) {
				value -= 500;
			}
			continue;
		}

		// Dead Four C: 11*11
		if (getLine(p, i, -1) == plyer && getLine(p, i, -2) == plyer
		    && getLine(p, i, 1) == plyer && getLine(p, i, 2) == plyer) {

			value += 230000;
			if(me != plyer) {
				value -= 500;
			}
			continue;
		}

		// Live Three, near 3 position: 111*0
		if (getLine(p, i, -1) == plyer && getLine(p, i, -2) == plyer
		    && getLine(p, i, -3) == plyer) {

			if (getLine(p, i, 1) == 0) {
				value += 750;
				if (getLine(p, i, -4) == 0) {
					value += 3150;
					if(me != plyer) {
						value -= 300;
					}
				}
			}
			if ((getLine(p, i, 1) == 3 - plyer || getLine(p, i, 1) == -1)
			    && getLine(p, i, -4) == 0) {
				value += 500;
			}
			continue;
		}

		// Live Three, away 3 position: 1110*
		if (getLine(p, i, -1) == 0 && getLine(p, i, -2) == plyer
		    && getLine(p, i, -3) == plyer && getLine(p, i, -4) == plyer) {
			value += 350;
			continue;
		}

		// Dead Three: 11*1
		if (getLine(p, i, -1) == plyer && getLine(p, i, -2) == plyer
		    && getLine(p, i, 1) == plyer) {
			value += 600;
			if (getLine(p, i, -3) == 0 && getLine(p, i, 2) == 0) {
				value += 3150;
				continue;
			}
			if ((getLine(p, i, -3) == 3 - plyer||getLine(p, i, -3) == -1) && (getLine(p, i, 2) == 3 - plyer||getLine(p, i, 2) == -1)) {
				continue;
			} else {
				value += 700;
				continue;
			}
		}

		// Number of Live Two
		if (getLine(p, i, -1) == plyer && getLine(p, i, -2) == plyer
		    && getLine(p, i, -3) != 3 - plyer && getLine(p, i, 1) != 3 - plyer) {
			numoftwo++;
		}

		// Other gomoku modle
		int numOfplyer = 0;
		// ++++* +++*+ ++*++ +*+++ *++++
		for (int k = -4; k <= 0; k++) {
			int temp = 0;

			for (int l = 0; l <= 4; l++) {
				if (getLine(p, i, k + l) == plyer) {
					temp++;
				} else
				if (getLine(p, i, k + l) == 3 - plyer
				    || getLine(p, i, k + l) == -1) {
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

/* Modifies the board to reflect the specified move. */
void doMove(Move *m, Side side) {
	// A NULL move means pass.
	if (m == NULL)
		return;

	int x = m->getX();
	int y = m->getY();

	addStone(side, x, y);
}

/* Helper function to print the board status */
void printBoard() {
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
Node(Move* move, Side side, Side evaluater, Board* board) {
	this->board = board;
	this->move = move;
	this->side = side;
	this->evaluater = evaluater;
	this->alpha = INT_MIN;
	this->beta = INT_MAX;
	// this->parent = NULL;
}

~Node() {
	delete this->board;
}
Board* getBoard() {
	return this->board;
}
Move* getMove() {
	return this->move;
}
Side getSide() {
	return this->side;
}
// Node* getParent() { return this->parent; }
// void setParent(Node* node) { this->parent = node; }
int getAlpha() {
	return this->alpha;
}
int getBeta() {
	return this->beta;
}
void setAlpha(int alpha) {
	this->alpha = alpha;
}
void setBeta(int beta) {
	this->beta = beta;
}
};

class DecisionTree {

private:
Node *root;

protected:
Side evaluater;

public:
DecisionTree(Board *board, Side evaluater) {
	// this is our side
	this->evaluater = evaluater;
	root = new Node(NULL, evaluater == AI ? Human : AI, evaluater, board);
}

~DecisionTree() {
	// free some stuff
}

Move * findBestMove(int depth) {
	Board* board = root->getBoard();
	vector<Move> moves = board->getMoves(board);
	Node* best = NULL;

	#pragma omp for schedule(guided)
	for (unsigned int i = 0; i < moves.size(); i++) {
		Move* move = new Move(moves[i].getX(), moves[i].getY());
		Board* newBoard = board->copy();
		newBoard->doMove(move, evaluater);
		Node* child = new Node(move, evaluater, evaluater, newBoard);

		// pass alpha and beta values down
		child->setAlpha(root->getAlpha());
		child->setBeta(root->getBeta());

		// search child
		search(child, depth - 1);

		if (best == NULL || child->getBeta() > best->getBeta()) {
			best = child;
		}
	}
	return best->getMove();
}

void search(Node *startingNode, int depth) {
	if (depth == 0) {
		Move* move = startingNode->getMove();
		Side side = startingNode->getSide();
		int value = startingNode->getBoard()->getScore(*move, evaluater, side);

		startingNode->setAlpha(value);
		startingNode->setBeta(value);
		return;
	}

	Board *board = startingNode->getBoard();
	Side oppositeSide = startingNode->getSide() == AI ? Human : AI;
	// vector<Move> moves = board->getMoves(oppositeSide);
	vector<Move> moves = board->getMoves(board);

	#pragma omp for schedule(guided)
	for (unsigned int i = 0; i < moves.size(); i++) {
		// create the next child
		Move *move = new Move(moves[i].getX(), moves[i].getY());
		Board *newBoard = board->copy();
		newBoard->doMove(move, oppositeSide);
		Node *child = new Node(move, oppositeSide, evaluater, newBoard);

		// pass alpha and beta values down
		child->setAlpha(startingNode->getAlpha());
		child->setBeta(startingNode->getBeta());

		// search child
		search(child, depth - 1);

		if (startingNode->getSide() == evaluater) {
			startingNode->setBeta(min(startingNode->getBeta(), child->getAlpha()));
		} else {
			startingNode->setAlpha(max(startingNode->getAlpha(), child->getBeta()));
		}

		delete child;

		if (startingNode->getAlpha() > startingNode->getBeta()) {
			return;
		}
	}
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

	DecisionTree *tree = new DecisionTree(board, side);
	Move *moveToMake = tree->findBestMove(DEPTH);
	board->doMove(moveToMake, side);
	return moveToMake;
}
};
int main() {
	// Run game on CPU here
	Board* board = new Board();
	Player* player1 = new Player(AI);
	Player* player2 = new Player(Human);

	Side turn = AI;
	string winner;
	// Move* m = new Move(6,6);
	Move* m = NULL;

	// timers
	clock_t start,end;
	start = clock();

	cout << endl << "Starting CPU game..." << endl;

	while (!board->isDone()) {
		// for(int i = 0; i < 225; i++){
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
	end = clock();
	float elapsedTime = (double(end-start)/CLOCKS_PER_SEC) * 1000;

	board->printBoard();

	if(turn == AI)
		winner = "O";
	else
		winner = "X";

	cout << "CPU Game completed." << endl << " elapsedTime = " << elapsedTime << " ms" <<endl;
	cout << winner << " won! " << "Board Size: " << BOARD_SIZE << " * " << BOARD_SIZE << " Depth : "<< DEPTH <<endl;

	return 0;
}
