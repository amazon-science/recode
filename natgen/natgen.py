'''
This file includes transformations on code structure. Many of the transformations are built based on 
NatGEN (https://github.com/saikat107/NatGen).
Original Copyright 2021 Saikat Chakraborty. Licensed under the MIT License.
Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
'''

import copy
import re
from typing import Union, Tuple
from matplotlib.widgets import EllipseSelector

from tree_sitter import Language, Parser
import math, random
import numpy as np
import os
import string

from .python_processor_with_patch import PythonProcessor
from .transformation_base import TransformationBase
from .utils import *
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline


class BlockSwap(TransformationBase):
    """ Swapping if_else block
    """
    def __init__(self, parser_path, language):
        super(BlockSwap, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        self.name = 'BlockSwap'
#         self.transformations = processor_function[language]
        self.transformations = [PythonProcessor.block_swap]
        processor_map = {
#             "java": self.get_tokens_with_node_type,
#             "c": self.get_tokens_with_node_type,
#             "cpp": self.get_tokens_with_node_type,
#             "c_sharp": self.get_tokens_with_node_type,
#             "javascript": JavascriptProcessor.get_tokens,
            "python": PythonProcessor.get_tokens,
#             "php": PhpProcessor.get_tokens,
#             "ruby": self.get_tokens_with_node_type,
#             "go": self.get_tokens_with_node_type,
        }
        self.final_processor = processor_map[self.language]

    def transform_code(
            self,
            code: Union[str, bytes],
    ) -> Tuple[str, object]:
        success = False
        transform_functions = copy.deepcopy(self.transformations)
        while not success and len(transform_functions) > 0:
            function = np.random.choice(transform_functions)
            transform_functions.remove(function)
            modified_code, success = function(code, self)
            if success:
                code = modified_code
        root_node = self.parse_code(
            code=code
        )
        return_values = self.final_processor(
            code=code.encode(),
            root=root_node
        )
        if isinstance(return_values, tuple):
            tokens, types = return_values
        else:
            tokens, types = return_values, None
        return re.sub("[ \t\n]+", " ", " ".join(tokens)), \
               {
                   "types": types,
                   "success": success
               }


class DeadCodeInserter(TransformationBase):
    """ Insert deadcode with a meaningless code block
    """
    def __init__(
            self,
            parser_path: str,
            language: str
    ):
        super(DeadCodeInserter, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.name = "DeadCodeInserter"
#         self.processor = processor_function[self.language]
#         self.tokenizer_function = tokenizer_function[self.language]
#         self.insertion_function = insertion_function[self.language]
        self.processor = PythonProcessor #processor_function[self.language]
        self.tokenizer_function = PythonProcessor.get_tokens #tokenizer_function[self.language]
        self.insertion_function = PythonProcessor.get_tokens_insert_before #insertion_function[self.language]


    def insert_random_dead_code(self, code_string, max_node_in_statement=-1):
        root = self.parse_code(code_string)
        original_node_count = count_nodes(root)
        if max_node_in_statement == -1:
            max_node_in_statement = int(original_node_count / 2)
        if self.language == "ruby":
            statement_markers = ["assignment", "until", "call", "if", "for", "while"]
        else:
            statement_markers = None
        statements = extract_statement_within_size(
            root, max_node_in_statement, statement_markers,
            code_string=code_string, tokenizer=self.tokenizer_function,
        )
        original_code = " ".join(self.tokenizer_function(code_string, root))
        try:
            while len(statements) > 0:
                random_stmt, insert_before = np.random.choice(statements, 2)
                statements.remove(random_stmt)
                dead_coed_body = " ".join(self.tokenizer_function(code_string, random_stmt)).strip()
                dead_code_function = np.random.choice(
                    [
                        self.processor.create_dead_for_loop,
                        self.processor.create_dead_while_loop,
                        self.processor.create_dead_if
                    ]
                )
                dead_code = dead_code_function(dead_coed_body)
                modified_code = " ".join(
                    self.insertion_function(
                        code_str=code_string, root=root, insertion_code=dead_code,
                        insert_before_node=insert_before
                    )
                )
                if modified_code != original_code:
                    modified_root = self.parse_code(" ".join(modified_code))
                    return modified_root, modified_code, True
        except:
            pass
        return root, original_code, False

    def transform_code(
            self,
            code: Union[str, bytes]
    ) -> Tuple[str, object]:
        root, code, success = self.insert_random_dead_code(code, -1)
        code = re.sub("[ \n\t]+", " ", code)
        return code, {
            "success": success
        }


class ForWhileTransformer(TransformationBase):
    """ Change the `for` loops with `while` loops and vice versa.
    """
    def __init__(self, parser_path, language):
        super(ForWhileTransformer, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        self.name = "ForWhileTransformer"
        self.transformations = [PythonProcessor.for_to_while_random, PythonProcessor.while_to_for_random] #processor_function[language]
#         self.transformations = processor_function[language]
        processor_map = {
#             "java": self.get_tokens_with_node_type,
#             "c": self.get_tokens_with_node_type,
#             "cpp": self.get_tokens_with_node_type,
#             "c_sharp": self.get_tokens_with_node_type,
#             "javascript": JavascriptProcessor.get_tokens,
            "python": PythonProcessor.get_tokens,
#             "php": PhpProcessor.get_tokens,
#             "ruby": self.get_tokens_with_node_type,
#             "go": self.get_tokens_with_node_type,
        }
        self.final_processor = processor_map[self.language]

    def transform_code(
            self,
            code: Union[str, bytes],
    ) -> Tuple[str, object]:
        success = False
        transform_functions = copy.deepcopy(self.transformations)
        while not success and len(transform_functions) > 0:
            function = np.random.choice(transform_functions)
            transform_functions.remove(function)
            modified_root, modified_code, success = function(code, self)
            if success:
                code = modified_code
        root_node = self.parse_code(
            code=code
        )
        return_values = self.final_processor(
            code=code.encode(),
            root=root_node
        )
        if isinstance(return_values, tuple):
            tokens, types = return_values
        else:
            tokens, types = return_values, None
        return re.sub("[ \t\n]+", " ", " ".join(tokens)), \
               {
                   "types": types,
                   "success": success
               }


''' start of Amazon addition '''
class ForWhileTransformerFirst(ForWhileTransformer):
    """ Change the `for` loops with `while` loops and vice versa for the first appeared without randomness.
    """
    def __init__(self, parser_path, language):
        super(ForWhileTransformer, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        self.name = "ForWhileTransformerFirst"
        # self.transformations = [PythonProcessor.for_to_while_random, PythonProcessor.while_to_for_random] #processor_function[language]
        self.transformations = [PythonProcessor.for_to_while_first, PythonProcessor.while_to_for_first] #processor_function[language]
#         self.transformations = processor_function[language]
        processor_map = {
#             "java": self.get_tokens_with_node_type,
#             "c": self.get_tokens_with_node_type,
#             "cpp": self.get_tokens_with_node_type,
#             "c_sharp": self.get_tokens_with_node_type,
#             "javascript": JavascriptProcessor.get_tokens,
            "python": PythonProcessor.get_tokens,
#             "php": PhpProcessor.get_tokens,
#             "ruby": self.get_tokens_with_node_type,
#             "go": self.get_tokens_with_node_type,
        }
        self.final_processor = processor_map[self.language]

    def transform_code(
            self,
            code: Union[str, bytes],
    ) -> Tuple[str, object]:
        success = False
        transform_functions = copy.deepcopy(self.transformations)
        # import pdb; pdb.set_trace()
        init = True
        while not success and len(transform_functions) > 0:
            if init:
                for_idx = code.find("for")
                for_idx = for_idx if for_idx != -1 else float('inf')
                while_idx = code.find("while")
                while_idx = while_idx if while_idx != -1 else float('inf')
                if for_idx <= while_idx:
                    # for to while
                    function = transform_functions[0]
                else:
                    # while to for
                    function = transform_functions[1]
                init = False
            else:
                function = np.random.choice(transform_functions)
            transform_functions.remove(function)
            modified_root, modified_code, success = function(code, self)
            # print(function, init, success)
            if success:
                code = modified_code
        root_node = self.parse_code(
            code=code
        )
        return_values = self.final_processor(
            code=code.encode(),
            root=root_node
        )
        if isinstance(return_values, tuple):
            tokens, types = return_values
        else:
            tokens, types = return_values, None
        return re.sub("[ \t\n]+", " ", " ".join(tokens)), \
               {
                   "types": types,
                   "success": success
               }
''' end of Amazon addition '''


class OperandSwap(TransformationBase):
    """ Swapping Operand "a>b" becomes "b<a"
    """
    def __init__(self, parser_path, language):
        super(OperandSwap, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        self.name = "OperandSwap"
        self.transformations = [PythonProcessor.operand_swap] #processor_function[language]
#         self.transformations = processor_function[language]
        processor_map = {
#             "java": self.get_tokens_with_node_type,
#             "c": self.get_tokens_with_node_type,
#             "cpp": self.get_tokens_with_node_type,
#             "c_sharp": self.get_tokens_with_node_type,
#             "javascript": JavascriptProcessor.get_tokens,
            "python": PythonProcessor.get_tokens,
#             "php": PhpProcessor.get_tokens,
#             "ruby": self.get_tokens_with_node_type,
#             "go": self.get_tokens_with_node_type,
        }
        self.final_processor = processor_map[self.language]

    def transform_code(
            self,
            code: Union[str, bytes],
    ) -> Tuple[str, object]:
        success = False
        transform_functions = copy.deepcopy(self.transformations)
        while not success and len(transform_functions) > 0:
            function = np.random.choice(transform_functions)
            transform_functions.remove(function)
            modified_code, success = function(code, self)
            if success:
                code = modified_code
        root_node = self.parse_code(
            code=code
        )
        return_values = self.final_processor(
            code=code.encode(),
            root=root_node
        )
        if isinstance(return_values, tuple):
            tokens, types = return_values
        else:
            tokens, types = return_values, None
        return re.sub("[ \t\n]+", " ", " ".join(tokens)), \
               {
                   "types": types,
                   "success": success
               }

''' start of Amazon addition '''
class VarRenamerBase(TransformationBase):
    """ Base class for renaming variables
    """
    def __init__(
            self,
            parser_path: str,
            language: str, 
    ):
        super(VarRenamerBase, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.name = "VarRenamerBase"
        self.processor = PythonProcessor #processor_function[self.language]
        self.tokenizer_function = PythonProcessor.get_tokens #tokenizer_function[self.language]
#         self.processor = processor_function[self.language]
#         self.tokenizer_function = tokenizer_function[self.language]
        # C/CPP: function_declarator
        # Java: class_declaration, method_declaration
        # python: function_definition, call
        # js: function_declaration
        self.TYPE_VARS = ["float", "list", "List", "int", "bool", "tuple", "str", "dict", "True", "False"]
        self.not_var_ptype = ["function_declarator", "class_declaration", "method_declaration", "function_definition",
                              "function_declaration", "call", "local_function_statement"]

    def extract_var_names(self, root, code_string):
        var_names = []
        queue = [root]

        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if (current_node.type == "identifier" or current_node.type == "variable_name") and str(
                    current_node.parent.type) not in self.not_var_ptype:
                var_names.append(self.tokenizer_function(code_string, current_node)[0])
            for child in current_node.children:
                queue.append(child)
        return var_names

    def get_not_var_ptype_var_names(self, root, code_string):
        var_names = []
        queue = [root]

        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if (current_node.type == "identifier" or current_node.type == "variable_name") and str(current_node.parent.type) in self.not_var_ptype:
                var_names.append(self.tokenizer_function(code_string, current_node)[0])
            for child in current_node.children:
                queue.append(child)
        return var_names

    def get_import_var_names(self, root, code_string):
        # import_code_string = str(code_string)
        lines = code_string.split("\n")
        new_lines = []
        for line in lines:
            new_line = str(line)
            if "import" not in line:
                continue
            new_lines.append(new_line)
        import_code_string = "\n".join(new_lines)
        import_var_names = self.extract_var_names(self.parse_code(import_code_string), import_code_string)
        return list(set(import_var_names))

    def select_most_frequent_var(self, original_code, var_names):
        counts = {}
        for var in var_names:
            counts[var] = original_code.count(var)
        max_cnt = -1
        max_var = None
        for var in var_names:
            if counts[var] > max_cnt:
                max_cnt = counts[var]
                max_var = var
        return max_var

    def check_valid_var(self, var_name):
        return var_name[0].isalpha() and var_name.replace("_", "").isalnum()

    def var_renaming(self, code_string, method, debug=False):
        root = self.parse_code(code_string)
        original_code = self.tokenizer_function(code_string, root)
        var_names = self.extract_var_names(root, code_string)
        var_names = list(set(var_names))
        # for variables in import line, remove
        import_var_names = self.get_import_var_names(root, code_string)
        for ivn in import_var_names:
            var_names.remove(ivn)
        # for variables in type variables, remove
        for type_var in self.TYPE_VARS:
            if type_var in var_names:
                var_names.remove(type_var)
        # for variable name appears in function/class name, remove
        not_var_ptype_var_names = self.get_not_var_ptype_var_names(root, code_string)
        for nvpvn in not_var_ptype_var_names:
            if nvpvn in var_names:
                var_names.remove(nvpvn)
        # selected_var is None if no suitable variable to replace
        selected_var = self.select_most_frequent_var(original_code, var_names)
        if debug: import pdb; pdb.set_trace()

        replace_var = None
        if selected_var is not None:
            # we have suitable var to replace if selected_var is not None
            if method == "CodeBERT":
                new_code_string = code_string.replace(selected_var, " <mask> ")
                model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
                tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
                fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
                outputs = fill_mask(new_code_string[:model.config.max_position_embeddings])
                replace_var_opts = {}
                assert len(outputs) > 0, "CodeBERT no outputs!"
                for pos in range(len(outputs)):
                    if type(outputs[pos]) == list:
                        assert len(outputs[pos]) > 0, f"empty prediction for pos {pos}"
                        for options in range(len(outputs[pos])):
                            pred, score = outputs[pos][options]["token_str"], outputs[pos][options]["score"]
                            if pred not in replace_var_opts:
                                replace_var_opts[pred] = score
                            else:
                                replace_var_opts[pred] += score
                    else:
                        options = pos # if only one mask position, no pos dimension, set option to pos
                        pred, score = outputs[options]["token_str"], outputs[options]["score"]
                        if pred not in replace_var_opts:
                            replace_var_opts[pred] = score
                        else:
                            replace_var_opts[pred] += score

                # selected the predicted options with the highest sum score
                max_score = -1
                for pred in replace_var_opts:
                    if not self.check_valid_var(pred): continue
                    if replace_var_opts[pred] > max_score:
                        max_score = replace_var_opts[pred]
                        replace_var = pred
                if debug: print(replace_var_opts)
            elif method == "naive":
                replace_var = "VAR_0"
            elif method == "alpha-numeric":
                # random variable names, half alphabetics half numbers
                if selected_var is not None:
                    if len(selected_var) == 1:
                        replace_var = random.choice(string.ascii_uppercase + string.ascii_lowercase)
                    else:
                        replace_var = random.choice(string.ascii_uppercase + string.ascii_lowercase) + \
                                ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + \
                                string.digits * 6) for _ in range(len(selected_var) - 1))

            if replace_var is not None:
                replace_var = replace_var.replace(" ", "")
                if replace_var in self.TYPE_VARS or replace_var in var_names:
                    replace_var = replace_var + "2"
            
        if debug: print(selected_var, "=>", replace_var)

        modified_code = []
        for t in original_code:
            if selected_var is not None and replace_var is not None and t == selected_var:
                # selected_var is None means no suitable var to replace
                # replace_var is None means no suitable replacement found for selected_var
                modified_code.append(replace_var)
            else:
                modified_code.append(t)

        modified_code_string = " ".join(modified_code)
        modified_root = self.parse_code(modified_code_string)
        return modified_root, modified_code_string, modified_code != original_code


class VarRenamerCB(VarRenamerBase):
    """ Using CodeBERT to replace targeted variables
    """
    def __init__(
            self,
            parser_path: str,
            language: str, 
    ):
        super(VarRenamerCB, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.name = "VarRenamerCB"
        self.processor = PythonProcessor #processor_function[self.language]
        self.tokenizer_function = PythonProcessor.get_tokens #tokenizer_function[self.language]


    def transform_code(
            self,
            code: Union[str, bytes],
    ) -> Tuple[str, object]:
        root, code, success = self.var_renaming(code, method="CodeBERT")
        code = re.sub("[ \n\t]+", " ", code)
        return code, {
            "success": success
        }


class VarRenamerNaive(VarRenamerBase):
    """ Using naive name VAR_0 to replace targeted variables
    """
    def __init__(
            self,
            parser_path: str,
            language: str, 
    ):
        super(VarRenamerNaive, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.name = "VarRenamerNaive"
        self.processor = PythonProcessor #processor_function[self.language]
        self.tokenizer_function = PythonProcessor.get_tokens #tokenizer_function[self.language]

    def transform_code(
            self,
            code: Union[str, bytes],
    ) -> Tuple[str, object]:
        root, code, success = self.var_renaming(code, method="naive")
        code = re.sub("[ \n\t]+", " ", code)
        return code, {
            "success": success
        }


class VarRenamerRN(VarRenamerBase):
    """ Using random alpha-numeric to repalce targeted variables
    """
    def __init__(
            self,
            parser_path: str,
            language: str, 
    ):
        super(VarRenamerRN, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.name = "VarRenamerRN"
        self.processor = PythonProcessor #processor_function[self.language]
        self.tokenizer_function = PythonProcessor.get_tokens #tokenizer_function[self.language]

    def transform_code(
            self,
            code: Union[str, bytes],
    ) -> Tuple[str, object]:
        root, code, success = self.var_renaming(code, method="alpha-numeric")
        code = re.sub("[ \n\t]+", " ", code)
        return code, {
            "success": success
        }
''' end of Amazon addition '''

class VarRenamer(TransformationBase):
    """ Original variable renaming defined in NatGEN
    """
    def __init__(
            self,
            parser_path: str,
            language: str
    ):
        super(VarRenamer, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.name = "VarRenamer"
        self.processor = PythonProcessor #processor_function[self.language]
        self.tokenizer_function = PythonProcessor.get_tokens #tokenizer_function[self.language]
#         self.processor = processor_function[self.language]
#         self.tokenizer_function = tokenizer_function[self.language]
        # C/CPP: function_declarator
        # Java: class_declaration, method_declaration
        # python: function_definition, call
        # js: function_declaration
        self.not_var_ptype = ["function_declarator", "class_declaration", "method_declaration", "function_definition",
                              "function_declaration", "call", "local_function_statement"]

    def extract_var_names(self, root, code_string):
        var_names = []
        queue = [root]

        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if (current_node.type == "identifier" or current_node.type == "variable_name") and str(
                    current_node.parent.type) not in self.not_var_ptype:
                var_names.append(self.tokenizer_function(code_string, current_node)[0])
            for child in current_node.children:
                queue.append(child)
        return var_names

    def var_renaming(self, code_string):
        root = self.parse_code(code_string)
        original_code = self.tokenizer_function(code_string, root)
        # print(" ".join(original_code))
        var_names = self.extract_var_names(root, code_string)
        var_names = list(set(var_names))
        num_to_rename = math.ceil(0.2 * len(var_names))
        random.shuffle(var_names)
        var_names = var_names[:num_to_rename]
        var_map = {}
        for idx, v in enumerate(var_names):
            var_map[v] = f"VAR_{idx}"
        modified_code = []
        for t in original_code:
            if t in var_names:
                modified_code.append(var_map[t])
            else:
                modified_code.append(t)

        modified_code_string = " ".join(modified_code)
        if modified_code != original_code:
            modified_root = self.parse_code(modified_code_string)
            return modified_root, modified_code_string, True
        else:
            return root, code_string, False

    def transform_code(
            self,
            code: Union[str, bytes]
    ) -> Tuple[str, object]:
        root, code, success = self.var_renaming(code)
        code = re.sub("[ \n\t]+", " ", code)
        return code, {
            "success": success
        }


class NoTransformation(TransformationBase):
    """ Baseline natgen transformation that doing nothing
    Note that the format of the code will be changed from the original code
    """
    def __init__(self, parser_path: str, language: str) -> object:
        super().__init__(parser_path, language)
        self.name = "NoTransformation"
        if not os.path.exists(parser_path):
            raise ValueError(
                f"Language parser does not exist at {parser_path}. Please run `setup.sh` to properly set the "
                f"environment!")
        self.lang_object = Language(parser_path, language)
        self.parser = Parser()
        self.parser.set_language(self.lang_object)
        processor_map = {
#             "java": self.get_tokens_with_node_type,
#             "c": self.get_tokens_with_node_type,
#             "cpp": self.get_tokens_with_node_type,
#             "c_sharp": self.get_tokens_with_node_type,
#             "javascript": JavascriptProcessor.get_tokens,
            "python": PythonProcessor.get_tokens,
#             "php": PhpProcessor.get_tokens,
#             "ruby": self.get_tokens_with_node_type,
#             "go": self.get_tokens_with_node_type,
        }
        self.processor = processor_map[language]

    def transform_code(
            self,
            code: Union[str, bytes]
    ) -> Tuple[str, object]:
        root_node = self.parse_code(
            code=code
        )
        return_values = self.processor(
            code=code.encode(),
            root=root_node
        )
        if isinstance(return_values, tuple):
            tokens, types = return_values
        else:
            tokens, types = return_values, None
        return re.sub("[ \t\n]+", " ", " ".join(tokens)), \
               {
                   "types": types,
                   "success": False
               }
