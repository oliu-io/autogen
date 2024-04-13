# class FunctionSumPropagator(FunctionPropagator):
#     """
#     This allows us to only reason about 2 processes
#     """

#     def _post_process_graph(self, graph: List[str]):
#         # now we do a post-process step
#         # grab the graph at the current level, if there are > 1 nodes
#         # we perform a merge (early sum)
#         # single chain representation
#         current_level_graph = list(filter(lambda x: x[0] == self.child.level + 1, graph))
#         if len(current_level_graph) > 1:
#             collect_all = []
#             for level, node in current_level_graph:
#                 func_form = node.split(" = ")[1]
#                 collect_all.append(func_form)
#             # instead of being literal total derivative
#             # we can use "Merge" as a symbol instead
#             new_graph = " + ".join(collect_all)
#             # graph = [(self.child.level, self.function_call), (self.child.level + 1, new_graph)]
#             # resolve and flatten the level
#             child_name, func_form = self.function_call.split(" = ")
#             new_graph = new_graph.replace(child_name, func_form)
#             graph = [(self.child.level, new_graph)]

#         return graph


# class FunctionDistributivePropagate(FunctionPropagator):
#     def _post_process_graph(self, graph: List[str]):
#         # now we do a post-process step
#         # we perform a distributive sum
#         current_level_graph = list(filter(lambda x: x[0] == self.child.level + 1, graph))
#         if len(current_level_graph) > 1:
#             collect_all = []
#             for level, node in current_level_graph:
#                 func_form = node.split(" = ")[1]
#                 child_name, child_func_form = self.function_call.split(" = ")
#                 # we keep the path derivative
#                 new_graph = func_form.replace(child_name, child_func_form)
#                 collect_all.append((self.child.level, new_graph))
#             graph = collect_all

#         return graph


# class LLMCallable(object):
#     def __init__(self, config_list):
#         autogen.OpenAIWrapper(config_list=config_list)

# if __name__ == "__main__":
# test_case_shallow_diamond(FunctionPropagator)
# test_case_shallow_diamond(FunctionSumPropagator)
# test_case_shallow_diamond(FunctionDistributivePropagate)
