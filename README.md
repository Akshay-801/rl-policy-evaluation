# POLICY EVALUATION

## AIM
To develop a Python program to evaluate a given policy for Jack's Car Rental problem by calculating the state-value function iteratively to maximize cumulative rewards.

## PROBLEM STATEMENT
We are assigned the task of creating a Reinforcement Learning agent to evaluate policies for Jack's Car Rental problem. Jack manages two car rental locations. Every day, some number of customers arrive at each location to rent cars, and some number of rented cars are returned. Renting out a car yields a reward of $10. Overnight, Jack can move cars between the two locations to better meet the demand the next day. Moving a car costs $2.

The environment has a maximum capacity of 20 cars at each location and a maximum of 5 cars can be moved overnight. The requests and returns at each location follow a Poisson distribution. The agent must evaluate different policies dictating how many cars to transfer between locations overnight to maximize the expected discounted return.

## POLICY EVALUATION FUNCTION
To evaluate a policy $\pi$, we use the Bellman Expectation Equation for the state-value function $V(s)$. The algorithm iteratively updates the value of each state until the changes fall below a small threshold $\theta$.

![alt text](./images/image.png)

## OUTPUT:

### Policy 1 (No Movement Policy):
--- EVALUATING POLICY 1 (No Movement) ---
Iteration 1 | Max Delta: 99.4583
Iteration 2 | Max Delta: 78.1130
Iteration 3 | Max Delta: 63.8115
Iteration 4 | Max Delta: 54.0203
Iteration 5 | Max Delta: 46.5412
...
Iteration 15 | Max Delta: 4.8732

### Policy 2 (Simple Rebalancing Policy):
--- EVALUATING POLICY 2 (Simple Rebalancing) ---
Iteration 1 | Max Delta: 99.4583
Iteration 2 | Max Delta: 78.8921
Iteration 3 | Max Delta: 65.5123
Iteration 4 | Max Delta: 56.4190
Iteration 5 | Max Delta: 49.3302
...
Iteration 16 | Max Delta: 4.7188

### Comparing Policies:
----- Policy Comparison -----
Policy 1 (No Movement) Average State Value      : 398.24
Policy 2 (Simple Rebalancing) Average State Value : 421.57

Comparison Result State-by-State:
Policy 2 is better on average, but not in all individual states.

## RESULT:
Therefore, policies are compared successfully using the iterative policy evaluation function for Jack's Car Rental MDP. The evaluation demonstrates that proactively rebalancing the cars between locations yields a significantly higher expected state-value over time compared to taking no action.
