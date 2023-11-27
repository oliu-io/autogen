import autogen
import gym
import verbal_gym
from textwrap import dedent, indent

from typing import Any, Dict, List, Optional, Union, Tuple


class VerbalGymUserAgent(autogen.UserProxyAgent):
    def __init__(self, llm_config, env_name="verbal-poem-Haiku-v0"):
        self.env = verbal_gym.make(env_name, instruction_type='b', feedback_type='a')
        super().__init__(
            name="UserAgent",
            system_message="Not used",
            llm_config=llm_config,
            max_consecutive_auto_reply=5,
            human_input_mode='NEVER'
        )
        self.register_reply(autogen.ConversableAgent, VerbalGymUserAgent._generate_verbal_gym_reply)
        filtered_reply_func_list = []
        # we remove the function call to send the message to openai
        for tup in self._reply_func_list:
            if tup['reply_func'] == autogen.ConversableAgent.generate_oai_reply:
                continue
            filtered_reply_func_list.append(tup)
        self._reply_func_list = filtered_reply_func_list

    def get_starting_message(self):
        # should do two things:
        # 1. initial observation from env.reset()
        # 2. communicate additional high-level information about the environment that could be missing from task spec
        #    such as reward range, action format, etc.?
        obs, info = self.env.reset()
        return obs['instruction']

    def verbalize(self, next_obs, feedback, reward):
        message = f"""Score: {reward}\n\n"""
        message += f"Feedback: {feedback}\n\n"
        if next_obs is not None:
            message += f"Instruction: {next_obs}\n\n"
        return message

    def _generate_verbal_gym_reply(self, messages: Optional[List[Dict]] = None,
                                   sender: Optional[autogen.Agent] = None,
                                   config: Optional[Any] = None) -> Tuple[bool, str]:
        # This returns final, reply
        # so this is not to indicate that the whole conversation should be over...

        # return True, ""
        # otherwise, return False, message
        message = messages[-1]
        next_obs, reward, terminated, truncated, next_info = self.env.step(message['content'])
        success = next_info['success']
        message = self.verbalize(next_obs['observation'], next_obs['feedback'], reward)
        # by appending TERMINATE to the message
        # we can stop the entire conversation
        if success:
            message += '\n\nTERMINATE'
        return True, message
