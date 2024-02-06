from typing import Optional, List, Dict, Callable, Union, Type, Any, Tuple
from autogen.trace.nodes import MessageNode, Node, ParameterNode, GRAPH, USED_NODES, NAME_SCOPES, node
from autogen.trace.utils import for_all_methods
from autogen.agentchat.agent import Agent
from autogen.trace.trace_operators import trace_operator

from autogen.agentchat.conversable_agent import ConversableAgent
import inspect
from dill.source import getsource

# Here we implement wrapper of Autogen ConversableAgent class

@for_all_methods
def trace_agent_scope(fun):
    """ This is a decorator that can be applied on all methods of a
    ConversableAgent. For a decorated agent, the agent's name is added to the
    scope NAME_SCOPES when a method of an agent is called. This is to track
    which agents create nodes."""
    def wrapper(self, *args, **kwargs):
        assert isinstance(self, ConversableAgent)
        NAME_SCOPES.append(self.name)
        output = fun(self, *args, **kwargs)
        NAME_SCOPES.pop()
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
            super().__init__(*args, **kwargs)

        # We override the self._oai_system_message to use the ParameterNode type.
        # TODO Add other parameters
        @property
        def _oai_system_message(self):
            return [self.__oai_system_message.data]  # XXX Not sure why _oai_system_message in Autogen is always a list of length 1

        @_oai_system_message.setter
        def _oai_system_message(self, value):  # This is the parameter
            assert isinstance(value, list)
            assert len(value) == 1  # XXX Not sure why _oai_system_message in Autogen is always a list of length 1
            self.__oai_system_message = ParameterNode(value[0])

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

        # _append_oai_message is called in super().send, which stores the sent message in self._oai_messages. We cannot use trace_operator here because _append_oai_message performs in-place operation to the agent's memory self._oai_messages. So we instead override the stored message in self._oai_messages with a MessageNode.
        def _append_oai_message(self, message: Node, role, conversation_id: Agent) -> bool:
            assert isinstance(message, Node), "message must be a Node type"
            # We don't touch the logic within super()._append_oai_message, but we replace the final content with the MessageNode. A Node object can be used as a dict, so it is compatible with the original logic.
            output = super()._append_oai_message(message, role, conversation_id) # output is boolean
            data = self._oai_messages[conversation_id][-1]  # This is a oai_message dict created in super()._append_oai_message. We now replace it with a MessageNode.
            m_data = MessageNode(data, description=f'[OAI_Message] This is the oai_message created based on a message.', inputs={'message': message})
            self._oai_messages[conversation_id][-1] = m_data  # This is the message added by super._append_oai_message
            return output

        #### Modify self.receive to use the Node type
        def receive(
            self,
            message: Node,
            sender: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
        ):
            assert isinstance(message, Node), "message must be a Node type"
            super().receive(message, sender, request_reply, silent)

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
            @trace_operator('[AGENT] An agent generates the reply based on the messages it has received.')
            def generate_reply(messages, sender, exclude):
                return _generate_reply([m.data for m in messages] if messages is not None else messages, sender, exclude)
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
