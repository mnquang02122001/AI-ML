# AI-Machine Learning(CSS188 Project)
## Project: Multi-Agent Search (HMQ)
### Introduction:

First, play a game of classic Pacman by running the following command: python pacman.py
and using the arrow keys to move. Now, run the provided [ReflexAgent] in [multiAgents.py]: python pacman.py -p ReflexAgent
Note that it plays quite poorly even on simple layouts: python pacman.py -p ReflexAgent -l testClassic

### Question 1: Reflex Agent
Improve the [ReflexAgent] in [multiAgents.py] to play respectably. The provided reflex agent code provides some helpful examples of methods that query the [GameState] for information. A capable reflex agent will have to consider both food locations and ghost locations to perform well. Your agent should easily and reliably clear the [testClassic] layout: python pacman.py -p ReflexAgent -l testClassic
Try out your reflex agent on the default mediumClassic layout with one ghost or two (and animation off to speed up the display): python pacman.py --frameTime 0 -p ReflexAgent -k 1(2)
Grading: We will run your agent on the openClassic layout 10 times. You will receive 0 points if your agent times out, or never wins. You will receive 1 point if your agent wins at least 5 times, or 2 points if your agent wins all 10 games. You will receive an addition 1 point if your agent’s average score is greater than 500, or 2 points if it is greater than 1000. You can try your agent out under these conditions with: python autograder.py -q q1
To run it without graphics, use: python autograder.py -q q1 --no-graphics

### Question 2:Minimax
Now you will write an adversarial search agent in the provided [MinimaxAgent] class stub in [multiAgents.py]. Your minimax agent should work with any number of ghosts, so you’ll have to write an algorithm that is slightly more general than what you’ve previously seen in lecture. In particular, your minimax tree will have multiple min layers (one for each ghost) for every max layer.

Grading: We will be checking your code to determine whether it explores the correct number of game states. This is the only reliable way to detect some very subtle bugs in implementations of minimax. As a result, the autograder will be very picky about how many times you call GameState.generateSuccessor. If you call it any more or less than necessary, the autograder will complain. To test and debug your code, run: python autograder.py -q q2

### Question 3: Alpha-Beta Pruning
Make a new agent that uses alpha-beta pruning to more efficiently explore the minimax tree, in [AlphaBetaAgent].
To test and debug your code, run: python autograder.py -q q3

### Question 4: Expectimax
Minimax and alpha-beta are great, but they both assume that you are playing against an adversary who makes optimal decisions. As anyone who has ever won tic-tac-toe can tell you, this is not always the case. In this question you will implement the [ExpectimaxAgent], which is useful for modeling probabilistic behavior of agents who may make suboptimal choices.
You can debug your implementation on small the game trees using the command: python autograder.py -q q4

### Question 5: Evaluation Function
Write a better evaluation function for pacman in the provided function [betterEvaluationFunction]. The evaluation function should evaluate states, rather than actions like your reflex agent evaluation function did
To test and debug your code, run: python autograder.py -q q5
