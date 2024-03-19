"""
Design space:
- A set of python functions to sample:
  1. logic operators: >, <, ==, !=, >=, <=
  2. math operators: +, -, *, /, %, //, **

If we send in Torch tensor or jax, it should also be traced automatically.

Goal: min/max, OR, hit a target value?

TODO:
1. add unary operations
2. Broaden the list of math_ops
3. Define max/min target (or restrict input domain)
"""

from autogen.trace.nodes import node
from autogen.trace.propagators import FunctionPropagator
import string
import random
import numpy as np
from textwrap import dedent

from copy import copy

from typing import List
from autogen.trace.operators import *


def reformat(program_str: str):
    # remove empty lines and leading/trailing spaces
    return dedent(program_str).strip()


logic_ops = [">", "<", "==", "!=", ">=", "<="]
logic_ops_programs = {
    ">": reformat("""lambda a, b: a > b"""),
    "<": reformat("""lambda a, b: a < b"""),
    "==": reformat("""lambda a, b: a == b"""),
    "!=": reformat("""lambda a, b: a != b"""),
    ">=": reformat("""lambda a, b: a >= b"""),
    "<=": reformat("""lambda a, b: a <= b"""),
}

# Not suppoprting unary ops for now
math_ops = ["+", "-", "*", "/", "%", "//", "**"]
math_ops_programs = {
    "+": reformat("""lambda a, b: a + b"""),
    "-": reformat("""lambda a, b: a - b"""),
    "*": reformat("""lambda a, b: a * b"""),
    "/": reformat("""lambda a, b: a / b"""),
    "%": reformat("""lambda a, b: a % b"""),
    "//": reformat("""lambda a, b: a // b"""),
    "**": reformat("""lambda a, b: a ** b""")
}

variable_name_collide_list = set()
MAX_VALUE = 2
MIN_VALUE = -2

def create_input_var(input_min=-10, input_max=10):
    # sample and return a random 5 letter name
    retry = 10
    cnt = 0

    name = "node_" + "".join(random.choices(string.ascii_lowercase, k=5))

    while name in variable_name_collide_list and cnt < retry:
        cnt += 1
        name = "node_" + "".join(random.choices(string.ascii_lowercase, k=5))

    value = random.randint(input_min, input_max)
    return node(value, name)


def create_var():
    value = random.randint(MIN_VALUE, MAX_VALUE)
    return value


class NumericalProgramSampler:
    def __init__(self, chain_length, param_num=1, include_logic=False,
                 two_var_mixture=[0.4, 0.4, 0.2],
                 logic_prob=0.3,
                 max_gen_var=10,
                 seed=1234, verbose=False):
        """
        Args:
            chain_length:
            param_num: for this problem, more natural to have >1 param
            max_gen_var: how many latent variables (not input variables) to generate.
                         A good rule of thumb is -- make it 1.5 times the chain_length

            goal_output: target output to hit
            verbose: Print out the computation that's sampled
        """

        assert chain_length > 0, "Chain length should be positive"
        assert type(chain_length) == int
        assert type(max_gen_var) == int

        self.mixture_assertion_check(two_var_mixture, 3)
        assert logic_prob >= 0 and logic_prob <= 1, "Logic prob should be between 0 and 1"

        self.set_seed(seed)

        self.chain_length = chain_length
        self.include_logic = include_logic
        self.max_gen_var = max_gen_var

        self.two_var_mixture = two_var_mixture
        self.mixture_dec_space = [(1, 1), (2, 0), (0, 2)]
        # (num1, num2): sample {num1} vars in input_var_space, sample {num2} in gen_var_space

        self.logic_prob = logic_prob
        self.param_num = param_num

        self.input_var_space = []
        self.gen_var_space = []

        for _ in range(param_num):
            self.input_var_space.append(create_input_var())

        self._goal_output = self.__call__(self.get_current_input(), seed=seed, verbose=verbose)
        self._goal_input = copy(self.get_current_input())

    @property
    def goal_input(self):
        return [i.data for i in self._goal_input]

    @property
    def goal_output(self):
        return self._goal_output.data

    def display_computation_graph(self):
        return self._goal_output.backward(visualize='True', feedback='fine', propagate=FunctionPropagator())

    def mixture_assertion_check(self, mixture, num_elements=2):
        assert abs(sum(mixture) - 1) < 1e-6, "The mixture should sum to 1"
        assert len(mixture) == num_elements, f"The mixture should have {num_elements} elements"

    def reset(self):
        self.input_var_space = []
        self.gen_var_space = []

    def set_seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def sample_vars_from_space(self, var_space, num_sample, is_gen=False):
        if num_sample == 0:
            return []

        sampled_vars = []
        for _ in range(num_sample):
            if is_gen:
                max_curr_len = min(len(var_space), self.max_gen_var)
                prob_new_var = 1 - max_curr_len / self.max_gen_var
                if np.random.rand() < prob_new_var:
                    sampled_var = create_var()
                    var_space.append(sampled_var)
                else:
                    sampled_var = random.choice(var_space)
                sampled_vars.append(sampled_var)
            else:
                # for input_var
                # we do a probability curve that favors later items (to increase computational complexity)
                # once all input_vars have been used
                if len(self.input_var_space) > self.param_num:
                    weights = np.exp(np.arange(len(self.input_var_space)))
                    p = weights / np.sum(weights)
                else:
                    p = [1 / len(self.input_var_space)] * len(self.input_var_space)
                sampled_var_idx = np.random.choice(list(range(len(self.input_var_space))), p=p)
                sampled_var = var_space[sampled_var_idx]
                sampled_vars.append(sampled_var)

        return sampled_vars

    def sample_two_vars(self):
        idx = np.random.choice(range(len(self.mixture_dec_space)), p=self.two_var_mixture)
        sample_nums = self.mixture_dec_space[idx]
        sampled_vars = self.sample_vars_from_space(self.input_var_space, sample_nums[0], is_gen=False) + \
                       self.sample_vars_from_space(self.gen_var_space, sample_nums[1], is_gen=True)
        if sample_nums == (1, 1):
            is_gen_var = (False, True)
        elif sample_nums == (2, 0):
            is_gen_var = (False, False)
        else:
            is_gen_var = (True, True)

        return sampled_vars, is_gen_var

    def sample_op(self):
        """
        Automatic unit that does two actions:
        1. sample 2 values from the input_var_space and gen_var_space
          - [1, 1] means 1 value from each
          - [0, 2] means 2 values from gen_var_space
          - [2, 0] means 2 values from input_var_space
          if gen_var_space is sampled, we have a chance to create a new latent var
            The prob of creating a new one and use it is:
            1 - len(gen_var_space) / max_gen_var
          This is a probability curve -- the more latent vars we have, the less likely we create a new one

        2. sample an arithmetic operator

        Returns: op, (var1, var2), (is_gen_var, is_gen_var)
        """
        op = math_ops_programs[np.random.choice(math_ops)]
        sampled_vars, is_gen_var = self.sample_two_vars()

        return op, sampled_vars, is_gen_var

    def sample_step(self, verbose=False):
        """
        Sample whether we want a logic op or not
        If yes: sample 2 ops and 1 logic op
        If no: sample 1 op

        a variable is a transformed input_var as long as there's one input_var being sampled
        a variable is a gen_var is all vars come from gen_var

        Returns: computed value, input_var or gen_var
        """
        if self.include_logic and np.random.rand() < self.logic_prob:
            # sample 2 ops and 1 logic op
            op1, vars1, is_gen1 = self.sample_op()
            op2, vars2, is_gen2 = self.sample_op()

            logic_op = logic_ops_programs[np.random.choice(logic_ops)]
            sampled_vars, is_gen_var = self.sample_two_vars()

            if eval(logic_op)(sampled_vars[0], sampled_vars[1]):
                # first op
                out_var = eval(op1)(vars1[0], vars2[1])
                out_var_is_gen = is_gen1[0] * is_gen1[1]
            else:
                # second op
                out_var = eval(op2)(vars2[0], vars2[1])
                out_var_is_gen = is_gen2[0] * is_gen2[1]
        else:
            # sample 1 op
            op, vars, is_gen = self.sample_op()
            out_var = eval(op)(vars[0], vars[1])
            out_var_is_gen = is_gen[0] * is_gen[1]

        return out_var, out_var_is_gen

    def step(self, verbose=False):
        """
        If
        Returns:

        """
        out_var, out_var_is_gen = self.sample_step(verbose=verbose)
        if out_var_is_gen:
            self.gen_var_space.append(out_var)
        else:
            self.input_var_space.append(out_var)

        # we still return the value, just in case this is the final step
        return out_var

    def get_current_input(self):
        return self.input_var_space[:self.param_num]

    def __call__(self, input_params: List[int], seed=1234, verbose=False):
        """
        Args:
            input_params: a list of input parameters

        Returns: the final value of the program
        """
        if type(input_params) != list:
            input_params = [input_params]

        assert len(input_params) == self.param_num, "The number of input params should be the same as param_num"
        self.input_var_space = input_params

        # so we get the same computation graph actually
        # by choosing a seed
        self.set_seed(seed)

        for _ in range(self.chain_length):
            out_var = self.step(verbose=verbose)
        return out_var

