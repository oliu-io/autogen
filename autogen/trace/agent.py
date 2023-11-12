from typing import Optional, List, Dict, Callable, Union, Type, Any
from autogen.trace.nodes import MessageNode, Node, trace
from autogen.agentchat.agent import Agent
from autogen.agentchat.conversable_agent import ConversableAgent


def traceAgent(AgentCls):
    """ return a decorated Agent class that will automatically learn
        from its experiences"""

    # make all the messages the MessageNode type
    assert issubclass(AgentCls, ConversableAgent)

    class _Agent(AgentCls):

        def initiate_chat(
            self,
            recipient: ConversableAgent,
            clear_history: Optional[bool] = True,
            silent: Optional[bool] = False,
            **context,
        ):
            self._prepare_chat(recipient, clear_history)
            self.send(Node(self.generate_init_message(**context)), recipient, silent=silent)

        # Modify the send and receive methods to use the MessageNode type
        def send(
            self,
            message: Node,
            recipient: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
        ) -> bool:
            assert isinstance(message, Node), "message must be a Node type"
            super().send(message, recipient, request_reply, silent)

        async def a_send(
            self,
            message: Union[Dict, str],
            recipient: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
        ) -> bool:
            raise NotImplementedError

        def _append_oai_message(self, message: Node, role, conversation_id: Agent) -> bool:
            assert isinstance(message, Node), "message must be a Node type"
            output = super()._append_oai_message(message.data, role, conversation_id)
            # replace with the MessageNode
            data = self._oai_messages[conversation_id][-1]
            m_data = MessageNode(data, f'identity(message)', kwargs={'message': message})
            self._oai_messages[conversation_id][-1] = m_data
            return output

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

        def _process_received_message(self, message: Node, sender, silent):
            assert isinstance(message, Node), "message must be a Node type"
            data = self._message_to_dict(message.data)
            message = MessageNode(data, f'_message_to_dict(message)', kwargs={'message': message})
            # When the agent receives a message, the role of the message is "user". (If 'role' exists and is 'function', it will remain unchanged.)
            valid = self._append_oai_message(message, "user", sender)  # message is a Node type
            if not valid:
                raise ValueError(
                    "Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
                )
            if not silent:
                self._print_received_message(message.data, sender)

        def generate_reply(
            self,
            messages: Optional[List[Node]] = None,
            sender: Optional[Agent] = None,
            exclude: Optional[List[Callable]] = None,
        ) -> Union[str, Dict, None]:
            if messages is not None:
                assert all(isinstance(m, Node) for m in messages), "messages must be a a list of Node types"
            reply = super().generate_reply([m.data for m in messages], sender, exclude)
            if reply is not None:
                reply = MessageNode(reply, f'generate_reply(messages)', args=messages)
            return reply

        async def a_generate_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            exclude: Optional[List[Callable]] = None,
        ) -> Union[str, Dict, None]:
            raise NotImplementedError

    return _Agent