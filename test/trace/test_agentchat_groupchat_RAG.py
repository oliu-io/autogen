# %% [markdown]
# <!--
# tags: ["group chat", "orchestration", "RAG"]
# description: |
#     Implement and manage a multi-agent chat system using AutoGen, where AI assistants retrieve information, generate code, and interact collaboratively to solve complex tasks, especially in areas not covered by their training data.
# -->

# %% [markdown]
# # Group Chat with Retrieval Augmented Generation
#
# AutoGen supports conversable agents powered by LLMs, tools, or humans, performing tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
# Please find documentation about this feature [here](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat).
#
# ````{=mdx}
# :::info Requirements
# Some extra dependencies are needed for this notebook, which can be installed via pip:
#
# ```bash
# pip install pyautogen[retrievechat]
# ```
#
# For more information, please refer to the [installation guide](/docs/installation/).
# :::
# ````

# %% [markdown]
# ## Set your API Endpoint
#
# The [`config_list_from_json`](https://microsoft.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.

# %%
import chromadb

import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.trace.trace import trace, trace_class, node
from autogen.trace.groupchat import GroupChat, GroupChatManager


config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

print("LLM models: ", [config_list[i]["model"] for i in range(len(config_list))])

# %% [markdown]
# ````{=mdx}
# :::tip
# Learn more about configuring LLMs for agents [here](/docs/llm_configuration).
# :::
# ````
#
# ## Construct Agents

# %%
llm_config = {
    "timeout": 60,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

# autogen.ChatCompletion.start_logging()


def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


boss = trace(autogen.UserProxyAgent)(
    name="Boss",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    system_message="The boss who ask questions and give tasks.",
    code_execution_config=False,  # we don't want to execute code in this case.
    default_auto_reply="Reply `TERMINATE` if the task is done.",
)

boss_aid = trace(RetrieveUserProxyAgent)(
    name="Boss_Assistant",
    is_termination_msg=termination_msg,
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
        "chunk_token_size": 1000,
        "model": config_list[0]["model"],
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,  # we don't want to execute code in this case.
)

coder = trace(AssistantAgent)(
    name="Senior_Python_Engineer",
    is_termination_msg=termination_msg,
    system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

pm = trace(autogen.AssistantAgent)(
    name="Product_Manager",
    is_termination_msg=termination_msg,
    system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

reviewer = trace(autogen.AssistantAgent)(
    name="Code_Reviewer",
    is_termination_msg=termination_msg,
    system_message="You are a code reviewer. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

PROBLEM = "How to use spark for parallel training in FLAML? Give me sample code."


def _reset_agents():
    boss.reset()
    boss_aid.reset()
    coder.reset()
    pm.reset()
    reviewer.reset()


def rag_chat():
    _reset_agents()
    groupchat = GroupChat(
        agents=[boss_aid, coder, pm, reviewer], messages=[], max_round=12, speaker_selection_method="round_robin"
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with boss_aid as this is the user proxy agent.
    boss_aid.initiate_chat(
        manager,
        problem=PROBLEM,
        n_results=3,
    )

    last_message = boss_aid.last_message_node()
    print("last_message", last_message.data)
    figure = last_message.backward("test", visualize=True, retain_graph=True)
    figure.view(filename="figures/rag_chat")


def norag_chat():
    _reset_agents()
    groupchat = GroupChat(
        agents=[boss, coder, pm, reviewer],
        messages=[],
        max_round=12,
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
    )

    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    # Start chatting with the boss as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        message=PROBLEM,
    )

    figure = boss.last_message_node().backward("test", visualize=True, simple_visualization=True)
    figure.view(filename="figures/norag_chat")


def call_rag_chat():
    _reset_agents()

    # In this case, we will have multiple user proxy agents and we don't initiate the chat
    # with RAG user proxy agent.
    # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
    # it from other agents.
    def retrieve_content(message, n_results=3):
        boss_aid.n_results = n_results  # Set the number of results to be retrieved.
        # Check if we need to update the context.
        update_context_case1, update_context_case2 = boss_aid._check_update_context(message)
        if (update_context_case1 or update_context_case2) and boss_aid.update_context:
            boss_aid.problem = message if not hasattr(boss_aid, "problem") else boss_aid.problem
            _, ret_msg = boss_aid._generate_retrieve_user_reply(message)
        else:
            ret_msg = boss_aid.generate_init_message(message, n_results=n_results)
        return ret_msg if ret_msg else message

    boss_aid.human_input_mode = "NEVER"  # Disable human input for boss_aid since it only retrieves content.

    llm_config = {
        "functions": [
            {
                "name": "retrieve_content",
                "description": "retrieve content for code generation and question answering.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
                        }
                    },
                    "required": ["message"],
                },
            },
        ],
        "config_list": config_list,
        "timeout": 60,
        "cache_seed": 42,
    }

    for agent in [coder, pm, reviewer]:
        # update llm_config for assistant agents.
        agent.llm_config.update(llm_config)

    for agent in [boss, coder, pm, reviewer]:
        # register functions for all agents.
        agent.register_function(
            function_map={
                "retrieve_content": retrieve_content,
            }
        )

    groupchat = GroupChat(
        agents=[boss, coder, pm, reviewer],
        messages=[],
        max_round=12,
        speaker_selection_method="random",
        allow_repeat_speaker=False,
    )

    manager_llm_config = llm_config.copy()
    manager_llm_config.pop("functions")
    manager = GroupChatManager(groupchat=groupchat, llm_config=manager_llm_config)

    # Start chatting with the boss as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        message=PROBLEM,
    )


# %% [markdown]
# ## Start Chat
#
# ### UserProxyAgent doesn't get the correct code
# [FLAML](https://github.com/microsoft/FLAML) was open sourced in 2020, so ChatGPT is familiar with it. However, Spark-related APIs were added in 2022, so they were not in ChatGPT's training data. As a result, we end up with invalid code.

# %%
norag_chat()

# %% [markdown]
### RetrieveUserProxyAgent get the correct code
# Since RetrieveUserProxyAgent can perform retrieval-augmented generation based on the given documentation file, ChatGPT can generate the correct code for us!

# %%
rag_chat()
# type exit to terminate the chat

# # # %% [markdown]
# # # ### Call RetrieveUserProxyAgent while init chat with another user proxy agent
# # # Sometimes, there might be a need to use RetrieveUserProxyAgent in group chat without initializing the chat with it. In such scenarios, it becomes essential to create a function that wraps the RAG agents and allows them to be called from other agents.

# # # %%
call_rag_chat()
