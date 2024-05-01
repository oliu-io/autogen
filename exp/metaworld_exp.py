# %% [markdown]
# # Learning Code as Policy for Metaworld
#

# %%
# Run experiment

# %%
import llfbench
import random
import os
import pickle
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import autogen.trace as trace
from autogen.trace.trace_ops import ExceptionNode, TraceExecutionError


def parse_obs(obs):
    """Parse the observation string into a dictionary of lists of floats."""
    import json

    obs = json.loads(obs)
    for key in obs:
        obs[key] = obs[key].replace("[", "").replace("]", "").split()
        obs[key] = [float(i) for i in obs[key]]
    return obs


def control_offset(obs, action):
    """
    Turn relative control to absolute control
    """
    offset = obs["observation"]["hand_pos"]
    action = [action[0] + offset[0], action[1] + offset[1], action[2] + offset[2], action[3]]
    return action


class TracedEnv:
    def __init__(self, env_name, seed=0, feedback_type="a", relative=True):
        random.seed(seed)
        np.random.seed(seed)
        self.env = llfbench.make(env_name, feedback_type=feedback_type)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.relative = relative
        self.obs = None

    @trace.trace_op()
    def reset(self):
        """
        Reset the environment and return the initial observation and info.
        """
        obs, info = self.env.reset()
        obs["observation"] = parse_obs(obs["observation"])
        self.obs = obs
        return obs, info

    def step(self, action):
        # We trace the step function; however, we do not trace the generation of
        # reward and info since we want to test the ability of agent to learn
        # from language feedback alone. We cannot simply apply trace_op, as the
        # information would be leaked. Below is a hack to prevent raward and
        # info from being leaked.
        try:  # Not traced
            control = control_offset(self.obs, action.data) if self.relative else action.data
            next_obs, reward, termination, truncation, info = self.env.step(control)
            next_obs["observation"] = parse_obs(next_obs["observation"])
            self.obs = next_obs
        except (
            Exception
        ) as e:  # Since we are not using trace_op, we need to handle exceptions maually as trace_op does internally.
            e_node = ExceptionNode(
                e,
                inputs={"action": action},
                description="[exception] The operator step raises an exception.",
                name="exception_step",
            )
            raise TraceExecutionError(e_node)

        # This is a way to hack trace_op to prevent reward and info from being traced.
        @trace.trace_op()
        def step(action):
            """
            Take action in the environment and return the next observation, reward, termination, truncation, and info.
            """
            return next_obs, None, termination, truncation, {}

        next_obs, _, termination, truncation, _ = step(action)  # traced
        return next_obs, reward, termination, truncation, info


def rollout(env, horizon, controller):
    """Rollout a controller in an env for horizon steps."""
    traj = dict(observation=[], action=[], reward=[], termination=[], truncation=[], success=[], input=[])

    # Initialize the environment
    obs, info = env.reset()
    traj["observation"].append(obs)

    # Rollout
    for t in range(horizon):
        error = None
        try:  # traced
            controller_input = obs["observation"]
            action = controller(controller_input)
            next_obs, reward, termination, truncation, info = env.step(action)
        except trace.TraceExecutionError as e:
            error = e
            break

        if error is None:  # log
            traj["observation"].append(next_obs)
            traj["action"].append(action)
            traj["reward"].append(reward)
            traj["termination"].append(termination)
            traj["truncation"].append(truncation)
            traj["success"].append(info["success"])
            if termination or truncation or info["success"]:
                break
            obs = next_obs
    return traj, error


def evaluate(env, horizon, policy, n_episodes=10):
    """Evaluate a policy in an env for horizon steps."""
    returns = []
    successes = []
    episode_lens = []
    for _ in range(n_episodes):
        traj, error = rollout(env, horizon, policy)
        assert error is None, "Error in rollout."
        sum_of_rewards = sum(traj["reward"])
        success = sum(traj["success"]) > 0
        episode_len = len(traj["reward"])
        returns.append(sum_of_rewards)
        successes.append(success)
        episode_lens.append(episode_len)
    print(f"Average return: {np.mean(returns)} (std: {np.std(returns)})")
    print(f"Average success: {np.mean(success)} (std: {np.std(success)})")
    print(f"Average episode length: {np.mean(episode_lens)} (std: {np.std(episode_lens)})")
    return returns, successes, episode_lens


# %% Evaluate Expert policy


def evaluate_expert(env_name, horizon, n_episodes, seed=0):
    env = TracedEnv(env_name, seed=seed, relative=False)

    def expert_policy(obs, env):
        action = env.env.expert_action
        if action is None:
            action = env.env.action_space.sample()
        return action

    print("Expert policy")
    returns, successes, episode_lens = evaluate(
        env, horizon, lambda obs: expert_policy(obs, env), n_episodes=n_episodes
    )


# %%


# Optimization experiment
def optimize_policy(
    env_name,
    horizon,
    n_episodes=10,
    n_optimization_steps=100,
    seed=0,
    relative=True,
    feedback_type=("hp", "hn"),
    logdir="./exp_results",
    mask=None,
    verbose=False,
    provide_reward=False,
):
    writer = SummaryWriter(logdir)

    # Define the variable
    @trace.trace_op(trainable=True)
    def controller(obs):
        """
        A feedback controller that computes the action based on the observation.
        """
        return [0, 0, 0, 0]

    optimizer = trace.optimizers.FunctionOptimizer(controller.parameters(), memory_size=0)

    print("Optimization Starts")
    log = dict(returns=[], successes=[], episode_lens=[])
    for i in range(n_optimization_steps):
        env = TracedEnv(env_name, seed=seed, feedback_type=feedback_type, relative=relative)  # fix init condition
        # Rollout and collect feedback
        traj, error = rollout(env, horizon, controller)

        # Compute feedback and logging
        if error is None:
            # Provide feedback to the last observation
            feedback = f"Success: {traj['success'][-1]}"
            if provide_reward:
                feedback += f"Rewards: {sum(traj['reward'])}"
            target = traj["observation"][-1]["observation"]
            # Logging Evaluate the current policy (Data not used in training)
            returns, successes, episode_lens = evaluate(env, horizon, controller, n_episodes=n_episodes)
            log["returns"].append(returns)
            log["successes"].append(successes)
            log["episode_lens"].append(episode_lens)
            writer.add_scalar("Rollout/return", sum(traj["reward"]), i)
            writer.add_scalar("Rollout/success", traj["success"][-1], i)
            writer.add_scalar("Evaluation/return", sum(returns), i)
            writer.add_scalar("Evaluation/success", sum(successes), i)
        else:  # Self debugging
            feedback = str(error)
            target = error.exception_node

        # Set the objective based on MW env
        optimizer.objective = (
            optimizer.default_objective
            + " Hint: "
            + traj["observation"][0]["instruction"].data.replace("absolute", "relative")
        )
        if env_name == "llf-metaworld-pick-place-v2":
            optimizer.objective = (
                "The goal of the task is to pick up a puck and put it to a goal position." + optimizer.objective
            )

        # Optimization step
        optimizer.zero_feedback()
        optimizer.backward(target, feedback)
        optimizer.step(verbose=verbose, mask=mask)

        print(f"Iteration: {i}")
        print(f"Feedback: {feedback}")
        print(f"Parameter:\n {controller.parameter.data}")

    returns, successes, episode_lens = evaluate(env, horizon, controller, n_episodes=n_episodes)
    log["returns"].append(returns)
    log["successes"].append(successes)
    log["episode_lens"].append(episode_lens)

    print("returns", [sum(r) for r in log["returns"]])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="llf-metaworld-pick-place-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--n_optimization_steps", type=int, default=100)
    parser.add_argument("--relative", type=bool, default=True)
    parser.add_argument("--feedback_type", type=str, default="a")
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--provide_reward", action="store_true")
    parser.add_argument("--logdir", type=str, default="./exp_results")
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()

    # logging
    logdir = args.logdir + f"/{args.env_name}"
    logdir += (
        f"/seed_{args.seed}_feedback_type_{args.feedback_type}_mask_{args.mask}_reward_{args.provide_reward}"
        + args.note
    )
    logdir += datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(logdir, exist_ok=True)
    pickle.dump(args, open(os.path.join(logdir, "args.pkl"), "wb"))  # Save the arguments for reproducibility

    evaluate_expert(args.env_name, args.horizon, args.n_episodes, seed=args.seed)
    optimize_policy(
        args.env_name,
        args.horizon,
        args.n_episodes,
        n_optimization_steps=args.n_optimization_steps,
        seed=args.seed,
        relative=args.relative,
        feedback_type=args.feedback_type,
        logdir=logdir,
        mask=args.mask,
    )
