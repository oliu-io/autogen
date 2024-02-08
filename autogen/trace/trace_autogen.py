from typing import Optional, List, Dict, Callable, Union, Type, Any, Tuple
from autogen.trace.nodes import MessageNode, Node, ParameterNode, GRAPH, USED_NODES, NAME_SCOPES, node
from autogen.trace.utils import for_all_methods
from autogen.agentchat.agent import Agent
from autogen.trace.trace_operators import trace_operator

from autogen.agentchat.conversable_agent import ConversableAgent
import inspect
from dill.source import getsource
from collections import defaultdict

# Here we implement wrapper of Autogen ConversableAgent class

class agent_scope():
    """ This is a context manager that can be used to add the agent's name to the
    scope NAME_SCOPES when a method of an agent is called. This is to track
    which agents create nodes."""
    def __init__(self, agent_name):
        self.agent_name = agent_name

    def __enter__(self):
        NAME_SCOPES.append(self.agent_name)

    def __exit__(self, exc_type, exc_value, traceback):
        NAME_SCOPES.pop()

@for_all_methods
def trace_agent_scope(fun):
    """ This is a decorator that can be applied on all methods of a
    ConversableAgent. For a decorated agent, the agent's name is added to the
    scope NAME_SCOPES when a method of an agent is called. This is to track
    which agents create nodes."""
    def wrapper(self, *args, **kwargs):
        assert isinstance(self, ConversableAgent)
        with agent_scope(self.name):
            output = fun(self, *args, **kwargs)
        return output
    return wrapper


def trace_ConversableAgent(AgentCls):
    """ Return a decorated Agent class who communicates with MessageNode type message, which can be traced."""

    # make all the messages the Node type
    assert issubclass(AgentCls, ConversableAgent)

    @trace_agent_scope
    class TracedAgent(AgentCls):

        def __init__(self, *args, **kwargs):
            self.__oai_system_message = None
            self.__oai_messages = defaultdict(list)  # dict of list of Nodes
            super().__init__(*args, **kwargs)

        # We override the self._oai_system_message. Interally, the system prompt
        # is stored in self.__oai_system_message as a list of ParameterNodes.
        # TODO Add other parameters
        @property
        def _oai_system_message(self):
            return [self.__oai_system_message.data]  # XXX Not sure why _oai_system_message in Autogen is always a list of length 1

        @_oai_system_message.setter
        def _oai_system_message(self, value):  # This is the parameter
            assert isinstance(value, list)
            assert len(value) == 1  # XXX Not sure why _oai_system_message in Autogen is always a list of length 1
            with agent_scope(self.name):  # NOTE setters are not covered by trace_agent_scope
                self.__oai_system_message = ParameterNode(value[0], description='[Parameter] System message of the agent.')

        @property
        def parameters(self):  # Return a list of ParameterNodes
            return [self.__oai_system_message]

        #### Wrap the output as a Node.
        def generate_init_message(self, **context) -> Union[str, Dict]:
            return node(super().generate_init_message(**context))

        #### Modify self.send to use the MessageNode type
        def send(
            self,
            message: Union[Dict, str, Node],  # TODO Should we restrict message to be a Node type?
            recipient: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
        ):
            assert message is not None  # self.send is called in either self.initiate_chat or self.receive. In both cases, message is not None.
            super().send(node(message), recipient, request_reply, silent)

        async def a_send(
            self,
            message: Union[Dict, str],
            recipient: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
        ) -> bool:
            raise NotImplementedError

        #### Modify self._append_oai_message.
        # We override the self._oai_messages and implement it as a property based on an internal attribute self.__oai_messages.
        # self.__oai_messages is a dict of list of Node, whereas self._oai_messages is a copy of self.__oai_messages and is a dict of list of dict.
        # In this way, we can keep the original codes of methods that read from self._oai_messages.
        # For methods that write into self._oai_messages, we override them below to write into self.__oai_messages directly
        # Lastly, we override last_message to return Node, instead of dict.

        @property
        def _oai_messages(self):  # return a dict of list of dict
            x = defaultdict(list)
            for k, v in self.__oai_messages.items():
                x[k] = [n.data for n in v]
            return x

        @_oai_messages.setter
        def _oai_messages(self, value):  # convert the dict of list of dict to dict of list of Node
            with agent_scope(self.name):  # NOTE setters are not covered by trace_agent_scope
                assert isinstance(value, dict)
                for k, v in value.items():
                    assert isinstance(v, list)
                    self.__oai_messages[k] = [node(n) for n in v]

        def _append_oai_message(self, message: Node, role, conversation_id: Agent) -> bool:
            assert isinstance(message, Node), "message must be a Node type."

            ### Original code
            message = self._message_to_dict(message)
            # create oai message to be appended to the oai conversation that can be passed to oai directly.
            oai_message = {k: message[k] for k in ("content", "function_call", "name", "context") if k in message}
            if "content" not in oai_message:
                if "function_call" in oai_message:
                    oai_message["content"] = None  # if only function_call is provided, content will be set to None.
                else:
                    return False

            oai_message["role"] = "function" if message.get("role") == "function" else role
            if "function_call" in oai_message:
                oai_message["role"] = "assistant"  # only messages with role 'assistant' can have a function call.
                oai_message["function_call"] = dict(oai_message["function_call"])
            # self._oai_messages[conversation_id].append(oai_message)
            ### End of original code

            # XXX  Since self._oai_messages creates a copy, to trace the creation of
            # the oai_message, we need to append to self.__oai_messages
            # directly.
            node_oai_message = MessageNode(oai_message, description=f'[OAI_Message] This is the oai_message created based on a message.', inputs={'message': message})
            self.__oai_messages[conversation_id].append(node_oai_message)
            return True

        def clear_history(self, agent: Optional[Agent] = None):
            # XXX Since self._oai_messages creates a copy, to trace the creation of
            # the oai_message, we need to call clear with self.__oai_messages.
            # directly.
            if agent is None:
                self.__oai_messages.clear()
            else:
                self.__oai_messages[agent].clear()

        def last_message_node(self, agent: Optional[Agent] = None) -> Node:
            if agent is None:
                n_conversations = len(self._oai_messages)
                if n_conversations == 0:
                    return None
                if n_conversations == 1:
                    # for conversation in self._oai_messages.values():
                    for conversation in self.__oai_messages.values():  # XXX We return MessageNode
                        return conversation[-1]
                raise ValueError("More than one conversation is found. Please specify the sender to get the last message.")
            # return self._oai_messages[agent][-1]
            return self.__oai_messages[agent][-1]  # XXX We return MessageNode

        @property
        def chat_message_nodes(self) -> Dict[Agent, List[Node]]:
            return self.__oai_messages

        #### Modify self.receive to use the Node type
        def receive(
            self,
            message: Node,
            sender: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
        ):
            assert isinstance(message, Node), "message must be a Node type"
            # super().receive(message, sender, request_reply, silent)

            self._process_received_message(message, sender, silent)
            if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
                return
            # reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)

            # Since self.chat_messages returns a dict of list of dict. We need
            # to pass nodes in self.__oai_messages to the new generate_reply, so
            # the node usages can be traced.
            reply = self.generate_reply(messages=self.__oai_messages[sender], sender=sender)
            if reply is not None:
                self.send(reply, sender, silent=silent)

        async def a_receive(
            self,
            message: Union[Dict, str],
            sender: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
        ):
            raise NotImplementedError

        # _message_to_dict is called in self._process_received_message (called
        # in super().receive). It formats the incoming message into a dict
        # format. We don't touch its logic but wrap its output as a
        # MessaageNode. Since ConversibleAgent calls
        # self._message_to_dict(message), we cannot use it as a static method
        # anymore.
        @trace_operator('[To_dict] Convert message to the dict format of Autogen.')
        def _message_to_dict(self, message: Node) -> Node:
            assert isinstance(message, Node), "message must be a Node type"
            return super()._message_to_dict(message.data)

        #### Modify self.generate_reply to use the Node type
        def generate_reply(
            self,
            messages: Optional[List[Node]] = None,
            sender: Optional[Agent] = None,
            exclude: Optional[List[Callable]] = None,
        ) -> Union[Node, None]:
            if messages is not None:
                assert all(isinstance(m, Node) for m in messages), "messages must be a a list of Node types"

            # We trace the super().generate_reply.
            _generate_reply = super().generate_reply
            @trace_operator('[Agent] An agent generates the reply based on the messages it has received.')
            def generate_reply(messages, sender, exclude):
                return _generate_reply(messages=[m.data for m in messages] if messages is not None else messages, sender=sender, exclude=exclude)
            reply = generate_reply(messages, sender, exclude)
            return reply

        async def a_generate_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            exclude: Optional[List[Callable]] = None,
        ) -> Union[str, Dict, None]:
            raise NotImplementedError

    return TracedAgent
