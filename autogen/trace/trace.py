from typing import Optional, List, Dict, Callable, Union, Type, Any, Tuple
from autogen.trace.nodes import MessageNode, Node, ParameterNode, GRAPH, USED_NODES
from autogen.agentchat.agent import Agent
from autogen.agentchat.conversable_agent import ConversableAgent
import inspect

def node(message):
    # wrap the message as a Node if it's not
    if isinstance(message, Node):
        return message
    return Node(message)

def trace(fun):
    if inspect.isclass(fun):
        if issubclass(fun, ConversableAgent):
            return trace_ConversableAgent(fun)
    # it should be a function
    return trace_operator(fun)

class no_trace():
    def __enter__(self):
        GRAPH.TRACE = False
    def __exit__(self, type, value, traceback):
        GRAPH.TRACE = True

class trace_node_usage:
    """ A context manager to track which nodes are used in an operator. After
         leaving the context, the nodes used in the operator can be found in
         USED_NODES.nodes.
    """
    def __enter__(self):
        USED_NODES.reset()
        USED_NODES.open = True

    def __exit__(self, exc_type, exc_value, traceback):
        USED_NODES.open = False

def trace_operator(fun):
    # trace a function
    # The wrapped function returns a message node
    assert callable(fun), "fun must be a callable"
    def wrapper(*args, **kwargs):
        # call the function with the data
        result = fun(*args, **kwargs)
        # wrap the inputs and outputs as Nodes if they're not
        m_args = (node(v) for v in args)
        m_kwargs = {k: node(v) for k, v in kwargs.items()}
        mapping = inspect.getsource(fun)  # TODO how to describe the mapping and inputs?
        # get the source code
        # inspect.getdoc(fun)
        m_result = MessageNode(result, description=mapping, args=m_args, kwargs=m_kwargs) if not isinstance(result, MessageNode) else result # TODO
        return m_result
    return wrapper


def compatability(fun):

    if not inspect.isclass(fun):  # do nothing
        assert callable(fun), "fun must be a callable"
        return fun

    assert issubclass(fun, ConversableAgent)
    traced_Cls = trace_ConversableAgent(fun)

    class CompatibleAgent(traced_Cls):
        # Just add no_trace to all public methods that may create MessageNodes

        def send(
            self,
            message: Union[Dict, str, Node],
            recipient: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
        ):
            with no_trace():
                super().send(message, recipient, request_reply, silent)

        def receive(
            self,
            message: Node,
            sender: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
        ):
            with no_trace():
                super().receive(message, sender, request_reply, silent)

        def generate_reply(
            self,
            messages: Optional[List[Node]] = None,
            sender: Optional[Agent] = None,
            exclude: Optional[List[Callable]] = None,
        ) -> Union[Node, None]:
            with no_trace():
                return super().generate_reply(messages, sender, exclude)

    return CompatibleAgent




def trace_ConversableAgent(AgentCls):
    """ return a decorated Agent class that will automatically learn
        from its experiences"""

    # make all the messages the MessageNode type
    assert issubclass(AgentCls, ConversableAgent)

    class TracedAgent(AgentCls):

        def __init__(self, *args, **kwargs):
            self.__oai_system_message = None
            super().__init__(*args, **kwargs)

        @property
        def _oai_system_message(self):
            return [self.__oai_system_message.data]  # XXX Not sure why _oai_system_message is a list of length 1

        @_oai_system_message.setter
        def _oai_system_message(self, value):  # This is the parameter
            assert isinstance(value, list)
            assert len(value) == 1  # XXX Not sure why _oai_system_message is a list of length 1
            self.__oai_system_message = ParameterNode(value[0])

        @property
        def parameters(self):
            return [self.__oai_system_message]

        def generate_init_message(self, **context) -> Union[str, Dict]:
            return node(super().generate_init_message(**context))

        # Modify the send and receive methods to use the MessageNode type
        def send(
            self,
            message: Union[Dict, str, Node],
            recipient: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
        ):
            assert message is not None
            super().send(node(message), recipient, request_reply, silent)

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
            m_data = MessageNode(data, description=f'identity(message)', kwargs={'message': message})
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
            message = MessageNode(data, description=f'_message_to_dict(message)', kwargs={'message': message})
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
        ) -> Union[Node, None]:
            if messages is not None:
                assert all(isinstance(m, Node) for m in messages), "messages must be a a list of Node types"
            with trace_node_usage():
                reply = super().generate_reply([m.data for m in messages] if messages is not None else messages, sender, exclude)
            if reply is not None and not isinstance(reply, Node):
                reply = MessageNode(reply, description=f'generate_reply(messages)', args=USED_NODES.nodes)  # TODO
            return reply

        async def a_generate_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            exclude: Optional[List[Callable]] = None,
        ) -> Union[str, Dict, None]:
            raise NotImplementedError

        # def generate_oai_reply(
        #     self,
        #     messages: Optional[List[Dict]] = None,
        #     sender: Optional[Agent] = None,
        #     config: Optional[Any] = None,
        # ) -> Tuple[bool, Union[str, Dict, None]]:
        #     flag, reply = super().generate_oai_reply(messages, sender, config)
        #     if reply is not None:
        #         m_messages = [node(m) for m in messages]
        #         reply = MessageNode(reply, description=f'generate_oai_reply(*messages, system_message=system_message)', args=m_messages, kwargs={'system_message': self.__oai_system_message}) # XXX
        #     return flag, reply

    return TracedAgent



if __name__=='__main__':


    x = 'hello'
    @trace
    def test(x):
        return x+' world'

    y = test(x)
    print(y)
    print('Parent', y.parent)
    print('Children', y.children)
    print('Level', y._level)
