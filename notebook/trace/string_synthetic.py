# inspired by COLLIE: https://arxiv.org/pdf/2307.08689.pdf
# pip install collie-bench

"""
The goal here is similar to number synthetic
There is a series of transformation performed on an input string
LLMs need to reason about the transformation to choose the input string carefully

As a blackbox function, it's hard to "guess" the transformation, thus requiring many trials
But with trace, we can see into the transformation, therefore, much much much easier

(This is also why it's a toy task)

We have a target string, which is revealed to the LLM

Goal: target string must be in the output of the transformation
"""

from autogen.trace.nodes import node
import string
import random
import numpy as np
from textwrap import dedent

from typing import List

def reformat(program_str: str):
    # remove empty lines and leading/trailing spaces
    return dedent(program_str).strip()

unary_sg_ret_string_ops = ["capitalize", "lower", "upper", "swapcase", "title"]
unary_sg_ret_string_op_programs = {
    "capitalize": reformat("""lambda s: s.capitalize()"""),
    "lower": reformat("""lambda s: s.lower()"""),
    "upper": reformat("""lambda s: s.upper()"""),
    "swapcase": reformat("""lambda s: s.swapcase()"""),
    "title": reformat("""lambda s: s.title()"""),
}

# split, replace, concat