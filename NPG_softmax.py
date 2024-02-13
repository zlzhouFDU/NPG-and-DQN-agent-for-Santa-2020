from scipy.stats import beta
import numpy as np

np.random.seed(123)

def D_softmax_ent_agent_3(observation, configuration):
    global policy_params, num_bandit, baseline, last_action, total_reward, last_action_probs \
        , learning_rate, entropy_coefficient, baseline_decay, reward_sums, post_a, post_b, gradient \
        , gamma, reward_rate, count_bay, count_soft, baseline_r
    num_bandit = configuration["banditCount"]

    if observation.step == 0:

        policy_params = np.zeros(num_bandit)
        baseline = np.zeros(num_bandit)
        total_reward = 0
        learning_rate = 0.001
        entropy_coefficient = 0.05
        baseline_decay = 0.9
        gamma = 0.97
        post_a = np.ones(num_bandit)
        post_b = np.ones(num_bandit)
        # post_a = np.random.uniform(a, b, size=num_bandit)
        # post_b = np.random.uniform(a, b, size=num_bandit)

        baseline_r = 0.0

        count_soft = 0
        count_bay = 0

        gradient = np.zeros(num_bandit)
        reward_rate = 0
    else:
        opp_index = (observation.agentIndex + 1) % len(observation.lastActions)
        opp_last = observation.lastActions[opp_index]

        reward = observation.reward - total_reward
        total_reward = observation.reward
        reward = reward

        if opp_last == last_action:  # and observation.step < 1500:
            reward += baseline[last_action]

            # post_a[last_action] += reward + (1 - observation.step / 2000)
            # post_b[last_action] += (1 - reward)

            post_a[last_action] += reward + (2 - observation.step / 2000)
            post_b[last_action] += (2 - reward)
        else:
            post_a[last_action] += reward + (1 - observation.step / 2000)
            post_b[last_action] += (1 - reward)

        # reward = baseline[opp_last]
        #
        # post_a[opp_last] += reward + (2 - observation.step / 2000)
        # post_b[opp_last] += (2 - reward)

        moment = gradient
        gradient = np.zeros(num_bandit)
        gradient = (baseline - np.mean(baseline)) / (1 - gamma) / np.std(baseline + 1e-10) * last_action_probs
        # gradient[last_action] = (reward-baseline[last_action]) / (1 - gamma)
        # gradient[last_action] = (baseline[last_action]) / (1 - gamma) #* (1 - last_action_probs[last_action])
        # gradient[last_action] = (reward - baseline[last_action]) / (1 - gamma)
        # gradient[last_action] = 1
        # gradient = gradient - last_action_probs
        # gradient = gradient * baseline / (1 - gamma)

        # 添加熵正则化项
        entropy_term = -entropy_coefficient * np.sum(last_action_probs * np.log(last_action_probs + 1e-10))

        # 更新策略参数
        policy_params += learning_rate * (gradient + entropy_term) + 0.01 * learning_rate * moment
        # policy_params = policy_params * (1-alpha) + (learning_rate * (gradient + entropy_term) + 0.09 * learning_rate * moment) * alpha

        # 更新基线估计
        baseline = post_a / (post_a + post_b).astype(float) + beta.std(post_a, post_b) * 3
        baseline_r = baseline_decay * baseline_r + (1 - baseline_decay) * reward

    # 计算Softmax策略
    action_probs = np.exp(policy_params) / np.sum(np.exp(policy_params))

    # 概率缩减为97%
    if observation.step > 300:

        action_probs[last_action] = 0.97 * action_probs[last_action]

        action_probs[opp_last] = 0.97 * action_probs[opp_last]

        action_probs = action_probs / np.sum(action_probs)

    # 选择动作
    chosen_action = np.random.choice(num_bandit, p=action_probs)

    last_action = chosen_action

    last_action_probs = action_probs

    if observation.step % 400 == 0 and observation.step > 0:
        learning_rate *= 1.2

    if observation.step > 0 and (total_reward / observation.step) < reward_rate:
        if observation.step > 100:
            chosen_action = int(np.argmax(baseline))

            last_action = chosen_action

        entropy_coefficient *= 0.9

        reward_rate = total_reward / observation.step


    elif observation.step > 0:
        chosen_action = int(np.argmax(action_probs))

        last_action = chosen_action


        reward_rate = total_reward / observation.step
    return chosen_action