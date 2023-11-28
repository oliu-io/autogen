from typing import Optional, List, Dict, Callable, Union, Type, Any, Tuple
from autogen.trace.nodes import MessageNode, Node, ParameterNode, GRAPH, USED_NODES, NAME_SCOPES
from autogen.agentchat.agent import Agent
from autogen.agentchat.conversable_agent import ConversableAgent
import inspect
from dill.source import getsource

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


def trace_node_usage(fun, agent=None, description=None):
    """ A decorator to track which nodes are used in an operator. After
         leaving the context, the nodes used in the operator can be found in
         USED_NODES.nodes.
    """
    description = description or f'{fun.__name__}(*args, **kwargs)'  # TODO
    def wrapper(*args, **kwargs):
        assert not USED_NODES.open, "trace_node_usage can't be nested"  # TODO Is this the right behavior?
        USED_NODES.reset()
        USED_NODES.open = True
        output = fun(*args, **kwargs)
        USED_NODES.open = False
        # USED_NODES contains the nodes used in the operator fun
        if output is not None and not isinstance(output, Node):
            output = MessageNode(output, description=description, args=USED_NODES.nodes)  # TODO
        return output
    return wrapper

def for_all_methods(decorator):
    """ Applying a decorator to all methods of a class. """
    def decorate(cls):
        for attr in cls.__dict__: # there's propably a better way to do this
            if callable(getattr(cls, attr)) and not attr.startswith("__"):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate

@for_all_methods
def trace_agent_scope(fun):
    # When a method of an agent is called, the agent's name is added to the scope.
    def wrapper(self, *args, **kwargs):
        assert isinstance(self, ConversableAgent)
        NAME_SCOPES.append(self.name)
        output = fun(self, *args, **kwargs)
        NAME_SCOPES.pop()
        return output
    return wrapper


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
        mapping = getsource(fun)  # TODO how to describe the mapping and inputs?
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

    @trace_agent_scope
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
            generate_reply = trace_node_usage(super().generate_reply, description=f'generate_reply(messages)')
            reply = generate_reply([m.data for m in messages] if messages is not None else messages, sender, exclude)
            return reply

        async def a_generate_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            exclude: Optional[List[Callable]] = None,
        ) -> Union[str, Dict, None]:
            raise NotImplementedError

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
