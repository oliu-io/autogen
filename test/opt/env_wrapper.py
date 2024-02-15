import autogen
import gym
import llfbench
from textwrap import dedent, indent

from typing import Any, Dict, List, Optional, Union, Tuple


class LLFBenchUserAgent(autogen.UserProxyAgent):
    def __init__(self, llm_config, env_name="llf-poem-Haiku-v0"):
        self.env = llfbench.make(env_name, instruction_type='b', feedback_type='a')
        super().__init__(
            name="UserAgent",
            system_message="Not used",
            llm_config=llm_config,
            max_consecutive_auto_reply=5,
            human_input_mode='NEVER',
            code_execution_config=False
        )
        self.register_reply(autogen.ConversableAgent, LLFBenchUserAgent._generate_llfbench_reply)

        # we need to append rewards in history (and see how well it works)
        self.info_history = []
        self.reward_history = []
        self.obs_history = []

    def get_task_description(self):
        pass

    def get_starting_message(self):
        # should do two things:
        # 1. initial observation from env.reset()
        # 2. communicate additional high-level information about the environment that could be missing from task spec
        #    such as reward range, action format, etc.?
        obs, info = self.env.reset()
        # initialize these
        self.obs_history = [obs]
        self.info_history = [info]
        self.reward_history = []
        return obs['instruction']

    def verbalize(self, next_obs, feedback, reward):
        message = f"""Score: {reward}\n\n"""
        message += f"Feedback: {feedback}\n\n"
        if next_obs is not None:
            message += f"Instruction: {next_obs}\n\n"
        return message

    def _generate_llfbench_reply(self, messages: Optional[List[Dict]] = None,
                                 sender: Optional[autogen.Agent] = None,
                                 config: Optional[Any] = None) -> Tuple[bool, str]:
        # This returns final, reply
        # so this is not to indicate that the whole conversation should be over...

        # return True, ""
        # otherwise, return False, message
        message = messages[-1]
        next_obs, reward, terminated, truncated, info = self.env.step(message['content'])
        success = info['success']
        message = self.verbalize(next_obs['observation'], next_obs['feedback'], reward)
        # by appending TERMINATE to the message
        # we can stop the entire conversation
        if success:
            message += '\n\nTERMINATE'

        self.info_history.append(info)
        self.obs_history.append(next_obs)
        self.reward_history.append(reward)

        return True, message
