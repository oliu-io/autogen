import logging
import random
import re
import sys
from dataclasses import dataclass, field
import trace
from typing import Dict, List, Optional, Union, Tuple
import warnings


from autogen.code_utils import content_str
from autogen.agentchat.agent import Agent
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.graph_utils import check_graph_validity, invert_disallowed_to_allowed


from autogen.trace.trace import node, trace_class
from autogen.trace.nodes import Node, MessageNode
from autogen.trace.trace_ops import trace_op
import copy
from autogen.agentchat.groupchat import GroupChat as _GroupChat
from autogen.agentchat.groupchat import GroupChatManager as _GroupChatManager

logger = logging.getLogger(__name__)


class NoEligibleSpeakerException(Exception):
    """Exception raised for early termination of a GroupChat."""

    def __init__(self, message="No eligible speakers."):
        self.message = message
        super().__init__(self.message)


@dataclass
class GroupChat(_GroupChat):
    @property
    def messages(self):  # return data
        return [m.data for m in self.__messages]
        # raise AttributeError

    @messages.setter
    def messages(self, messages):  # set data
        self.__messages = [node(m) for m in messages]

    def reset(self):
        self.__messages.clear()

    def append(self, message: Node[Dict], speaker_node: Agent):
        # set the name to speaker's name if the role is not function
        # if the role is tool, it is OK to modify the name
        @trace_op(
            "[GroupChat.append.process_message] Process a message before appending it to the group chat.",
            unpack_input=False,
        )
        def process_message(message: Node[Dict], speaker: Node[Agent]):
            message, speaker = copy.deepcopy(message.data), speaker.data
            # Old code
            if message["role"] != "function":
                message["name"] = speaker.name  # this depends on the speaker
                message["content"] = content_str(message["content"])
            return message  # This would be wrapped as a MessageNode

        message = process_message(message, speaker_node)
        assert isinstance(message, MessageNode)
        self.__messages.append(message)

    @trace_op("[GroupChat.agent_by_name] Returns the agent with a given name.", unpack_input=False)
    def agent_by_name(self, name: Node[str]) -> Agent:
        """Returns the agent with a given name."""
        return self.agents[self.agent_names.index(name)]

    @trace_op("[GroupChat.manual_select_speaker] Manually select the next speaker.", unpack_input=False)
    def manual_select_speaker(self, agents: Optional[List[Node[Agent]]] = None) -> Union[Node[Agent], None]:
        return super().manual_select_speaker(agents.data)

    @trace_op("[GroupChat.next_agent] Select the next speaker.", unpack_input=False)
    def next_agent(self, agent: Node[Agent], agents: Optional[List[Node[Agent]]] = None) -> Node[Agent]:
        return super().next_agent(agent.data, agents.data)

    @trace_op("[GroupChat.random_select_speaker] Randomly select the next speaker.", unpack_input=False)
    def random_select_speaker(self, agents: Optional[List[Node[Agent]]] = None) -> Union[Node[Agent], None]:
        return super().random_select_speaker(agents.data)

    @trace_op(
        "[GroupChat._prepare_and_select_agents.select_speaker_message] Prompt the user to select the next speaker.",
        unpack_input=False,
    )
    def select_speaker_message(self, agents: Node[Node[List[Agent]]]):
        return {"role": "system", "content": self.select_speaker_prompt([a for a in agents.data])}

    @trace_op(
        "[GroupChat.select_speaker_msg] Return the system message for selecting the next speaker.", unpack_input=False
    )
    def select_speaker_msg(self, agents: Optional[Node[List[Agent]]] = None) -> str:
        if agents is not None:
            agents = agents.data
        return super().select_speaker_msg(agents)

    def _prepare_and_select_agents(
        self, last_speaker: Node[Agent]
    ) -> Tuple[Optional[Node[Agent]], List[Agent], Optional[List[Node[Dict]]]]:
        @trace_op(
            "[GroupChat._prepare_and_select_agents.get_graph_eligible_agents] Get the eligible agents.",
            unpack_input=False,
        )
        def get_graph_eligible_agents(last_speaker):
            last_speaker = last_speaker.data

            if self.speaker_selection_method.lower() not in self._VALID_SPEAKER_SELECTION_METHODS:
                raise ValueError(
                    f"GroupChat speaker_selection_method is set to '{self.speaker_selection_method}'. "
                    f"It should be one of {self._VALID_SPEAKER_SELECTION_METHODS} (case insensitive). "
                )

            # If provided a list, make sure the agent is in the list
            allow_repeat_speaker = (
                self.allow_repeat_speaker
                if isinstance(self.allow_repeat_speaker, bool) or self.allow_repeat_speaker is None
                else last_speaker in self.allow_repeat_speaker
            )

            agents = self.agents
            n_agents = len(agents)
            # Warn if GroupChat is underpopulated
            if n_agents < 2:
                raise ValueError(
                    f"GroupChat is underpopulated with {n_agents} agents. "
                    "Please add more agents to the GroupChat or use direct communication instead."
                )
            elif n_agents == 2 and self.speaker_selection_method.lower() != "round_robin" and allow_repeat_speaker:
                logger.warning(
                    f"GroupChat is underpopulated with {n_agents} agents. "
                    "Consider setting speaker_selection_method to 'round_robin' or allow_repeat_speaker to False, "
                    "or use direct communication, unless repeated speaker is desired."
                )

            if (
                self.func_call_filter
                and self.messages
                and ("function_call" in self.messages[-1] or "tool_calls" in self.messages[-1])
            ):
                funcs = []
                if "function_call" in self.messages[-1]:
                    funcs += [self.messages[-1]["function_call"]["name"]]
                if "tool_calls" in self.messages[-1]:
                    funcs += [
                        tool["function"]["name"]
                        for tool in self.messages[-1]["tool_calls"]
                        if tool["type"] == "function"
                    ]

                # find agents with the right function_map which contains the function name
                agents = [agent for agent in self.agents if agent.can_execute_function(funcs)]
                if len(agents) == 1:
                    # only one agent can execute the function
                    return agents[0], agents, None
                elif not agents:
                    # find all the agents with function_map
                    agents = [agent for agent in self.agents if agent.function_map]
                    if len(agents) == 1:
                        return agents[0], agents, None
                    elif not agents:
                        raise ValueError(
                            f"No agent can execute the function {', '.join(funcs)}. "
                            "Please check the function_map of the agents."
                        )
            # remove the last speaker from the list to avoid selecting the same speaker if allow_repeat_speaker is False
            agents = [agent for agent in agents if agent != last_speaker] if allow_repeat_speaker is False else agents

            # Filter agents with allowed_speaker_transitions_dict

            is_last_speaker_in_group = last_speaker in self.agents

            # this condition means last_speaker is a sink in the graph, then no agents are eligible
            if last_speaker not in self.allowed_speaker_transitions_dict and is_last_speaker_in_group:
                raise NoEligibleSpeakerException(
                    f"Last speaker {last_speaker.name} is not in the allowed_speaker_transitions_dict."
                )
            # last_speaker is not in the group, so all agents are eligible
            elif last_speaker not in self.allowed_speaker_transitions_dict and not is_last_speaker_in_group:
                graph_eligible_agents = []
            else:
                # Extract agent names from the list of agents
                graph_eligible_agents = [
                    agent for agent in agents if agent in self.allowed_speaker_transitions_dict[last_speaker]
                ]

            # If there is only one eligible agent, just return it to avoid the speaker selection prompt
            if len(graph_eligible_agents) == 1:
                return graph_eligible_agents[0], graph_eligible_agents, None

            # If there are no eligible agents, return None, which means all agents will be taken into consideration in the next step
            if len(graph_eligible_agents) == 0:
                graph_eligible_agents = None
            return graph_eligible_agents

        graph_eligible_agents = get_graph_eligible_agents(last_speaker)

        ####################################################################################
        # Use the selected speaker selection method
        select_speaker_messages = None
        if self.speaker_selection_method.lower() == "manual":
            selected_agent = self.manual_select_speaker(graph_eligible_agents)
        elif self.speaker_selection_method.lower() == "round_robin":
            selected_agent = self.next_agent(last_speaker, graph_eligible_agents)
        elif self.speaker_selection_method.lower() == "random":
            selected_agent = self.random_select_speaker(graph_eligible_agents)
        else:
            selected_agent = None
            select_speaker_messages = self.__messages.copy()  # list[Node[Dict]]
            # If last message is a tool call or function call, blank the call so the api doesn't throw

            @trace_op(
                "[GroupChat._prepare_and_select_agents.process_speaker_message] Process the speaker message.",
                unpack_input=False,
            )
            def process_speaker_message(message: Node[Dict]):
                if message.data.get("function_call", False):
                    return dict(message.data, function_call=None)
                if message.data.get("tool_calls", False):
                    return dict(message.data, tool_calls=None)
                return message.data  # bypass

            select_speaker_messages[-1] = process_speaker_message(select_speaker_messages[-1])
            self.select_speaker_message(graph_eligible_agents)
            select_speaker_messages = select_speaker_messages + [self.select_speaker_message(graph_eligible_agents)]

        assert isinstance(selected_agent, Node) or selected_agent is None
        ####################################################################################
        # TODO selected_agent needs to be traced properly
        return selected_agent, graph_eligible_agents, select_speaker_messages

    def select_speaker(self, last_speaker_node: Node[Agent], selector: ConversableAgent) -> Agent:
        """Select the next speaker."""

        selected_agent_node, agents, messages = self._prepare_and_select_agents(last_speaker_node)
        if selected_agent_node is not None:
            return selected_agent_node

        # auto speaker selection
        selector.update_system_message(self.select_speaker_msg(agents))
        final, name = selector.generate_oai_reply(messages)

        @trace_op("[GroupChat._finalize_speaker] Finalize the speaker selection.", unpack_input=False)
        def _finalize_speaker(last_speaker: Node[Agent], final: Node[bool], name: Node[str], agents: Node[List[Agent]]):
            return node(self._finalize_speaker(last_speaker.data, final.data, name.data, agents.data))

        return _finalize_speaker(last_speaker_node, final, name, agents)

    async def a_select_speaker(self, last_speaker: Agent, selector: ConversableAgent) -> Agent:
        raise NotImplementedError


@trace_class
class GroupChatManager(_GroupChatManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self._groupchat, GroupChat), "Need to use the GroupChat class in trace."
        self.register_reply(
            Agent, GroupChatManager.run_chat, config=self._groupchat, reset_config=GroupChat.reset
        )  # XXX override

    # We implement this method to operate with Nodes
    def run_chat(
        self,
        messages: Optional[List[Node[Dict]]] = None,
        sender: Optional[Agent] = None,
        config: Optional[GroupChat] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Run a group chat."""
        if messages is None:
            messages = self.__oai_messages[sender]  # XXX Node[Dict]
        message = messages[-1]

        speaker_node = node(sender)  # XXX Convert to Node

        groupchat = config
        if self.client_cache is not None:
            warnings.warn("Client cache is not supported in trace.")
            for a in groupchat.agents:
                a.previous_cache = a.client_cache
                a.client_cache = self.client_cache
        for i in range(groupchat.max_round):
            # XXX Check types
            assert isinstance(speaker_node, Node), speaker_node
            assert isinstance(speaker_node.data, ConversableAgent)
            assert isinstance(message, Node), message

            groupchat.append(message, speaker_node)  # append a MessageNode to groupchat
            # broadcast the message to all agents except the speaker
            for agent in groupchat.agents:

                @trace_op(
                    "[GroupChatManager.broadcast_message] Broadcast a message to all agents except the speaker.",
                    unpack_input=False,
                )
                def broadcast_message(message: Node[dict], agent: Agent, speaker_node: Node[Agent]):
                    if agent != speaker_node.data:
                        self.send(message, agent, request_reply=False, silent=True)

                broadcast_message(message, agent, speaker_node)
            if self._is_termination_msg(message.data) or i == groupchat.max_round - 1:  # XXX Not traced
                # The conversation is over or it's the last round
                break
            try:
                # select the next speaker
                speaker_node = groupchat.select_speaker(speaker_node, self)
                reply = speaker_node.call("generate_reply", sender=self)
            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker_node = groupchat.agent_by_name(node(groupchat.admin_name))
                    reply = speaker_node.call("generate_reply", sender=self)
                else:
                    # admin agent is not found in the participants
                    raise
            except NoEligibleSpeakerException:
                # No eligible speaker, terminate the conversation
                break

            # assert isinstance(reply, Node) or reply is None, reply
            if reply is None:
                # no reply is generated, exit the chat
                break

            # check for "clear history" phrase in reply and activate clear history function if found
            if (
                groupchat.enable_clear_history
                and isinstance(reply, dict)
                and "CLEAR HISTORY" in reply["content"].upper()
            ):
                raise NotImplementedError('Writing to reply["content"] is not supported.')  # TODO
                reply["content"] = self.clear_agents_history(reply["content"], groupchat)
            # The speaker sends the message without requesting a reply
            speaker_node.call("send", reply, self, request_reply=False)

            # @trace_op('[GroupChatManager.get_last_message] Get the last message.')
            def get_last_message(speaker):
                last_message_node = self.last_message_node(speaker.data)
                return MessageNode(
                    last_message_node.data,
                    inputs=[last_message_node, speaker],
                    description="[GroupChatManager.get_last_message] Get the last message.",
                )
                # return self.last_message_node(speaker.data)

            message = get_last_message(speaker_node)

        if self.client_cache is not None:
            warnings.warn("Client cache is not supported in trace.")
            for a in groupchat.agents:
                a.client_cache = a.previous_cache
                a.previous_cache = None
        return node(True), None

    async def a_run_chat(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[GroupChat] = None,
    ):
        raise NotImplementedError
