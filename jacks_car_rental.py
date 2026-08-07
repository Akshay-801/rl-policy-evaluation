import math
import numpy as np


def poisson_probs(lmbda, max_k=11):
    probs = [math.exp(-lmbda) * (lmbda ** k) / math.factorial(k) for k in range(max_k)]
    tail = 1.0 - sum(probs)
    probs[-1] += tail
    return probs


def expected_return(state, action, V, poisson_rental_0, poisson_rental_1,
                    poisson_return_0, poisson_return_1,
                    max_cars=20, rental_reward=10, move_cost=2,
                    gamma=0.9):
    
    n0, n1 = state
    
    moved = int(action)
    n0_after = min(n0 - moved, max_cars)
    n1_after = min(n1 + moved, max_cars)
    reward = -move_cost * abs(moved)

    value = 0.0

    max_rental = len(poisson_rental_0)
    max_return = len(poisson_return_0)

    for req0 in range(max_rental):
        p_req0 = poisson_rental_0[req0]
        for req1 in range(max_rental):
            p_req1 = poisson_rental_1[req1]

            real_rent0 = min(n0_after, req0)
            real_rent1 = min(n1_after, req1)
            reward_rent = (real_rent0 + real_rent1) * rental_reward

            n0_post_rent = n0_after - real_rent0
            n1_post_rent = n1_after - real_rent1

            for ret0 in range(max_return):
                p_ret0 = poisson_return_0[ret0]
                for ret1 in range(max_return):
                    p_ret1 = poisson_return_1[ret1]

                    p = p_req0 * p_req1 * p_ret0 * p_ret1

                    n0_end = min(n0_post_rent + ret0, max_cars)
                    n1_end = min(n1_post_rent + ret1, max_cars)

                    total_reward = reward + reward_rent
                    value += p * (total_reward + gamma * V[n0_end, n1_end])

    return value


def policy_evaluation(policy, V_init=None, theta=1e-3, gamma=0.9,
                      max_cars=20, max_move=5,
                      rental_lambdas=(3,4), return_lambdas=(3,2)):
    max_poisson = 11
    pr_r0 = poisson_probs(rental_lambdas[0], max_poisson)
    pr_r1 = poisson_probs(rental_lambdas[1], max_poisson)
    pr_ret0 = poisson_probs(return_lambdas[0], max_poisson)
    pr_ret1 = poisson_probs(return_lambdas[1], max_poisson)

    V = np.zeros((max_cars + 1, max_cars + 1)) if V_init is None else V_init.copy()

    iteration = 0
    while True:
        delta = 0.0
        iteration += 1
        for i in range(max_cars + 1):
            for j in range(max_cars + 1):
                s = (i, j)
                a = policy(i, j)
                v = V[i, j]
                V[i, j] = expected_return(s, a, V, pr_r0, pr_r1, pr_ret0, pr_ret1,
                                           max_cars=max_cars, gamma=gamma)
                delta = max(delta, abs(v - V[i, j]))
        print(f"Iteration {iteration} | Max Delta: {delta:.6f}")
        if delta < theta:
            break

    return V


def policy_no_movement(i, j):
    return 0


def policy_simple_balance(i, j, max_move=5):
    move = int(round((j - i) / 2.0))
    move = max(-max_move, min(max_move, move))
    return move


def main():
    print("--- EVALUATING POLICY 1 (No Movement) ---")
    V1 = policy_evaluation(lambda i, j: policy_no_movement(i, j), theta=1e-3, gamma=0.9)
    print("Done. Average State Value:", np.mean(V1))

    print("--- EVALUATING POLICY 2 (Simple Balancing) ---")
    V2 = policy_evaluation(lambda i, j: policy_simple_balance(i, j), theta=1e-3, gamma=0.9)
    print("Done. Average State Value:", np.mean(V2))

    print("--- Comparison ---")
    print("Policy 1 Average:", np.round(np.mean(V1), 2))
    print("Policy 2 Average:", np.round(np.mean(V2), 2))


if __name__ == '__main__':
    main()
