'''
This file includes helper function for transformations on code structure.
Functions are from NatGEN (https://github.com/saikat107/NatGen).
Original Copyright 2021 Saikat Chakraborty. Licensed under the MIT License.
'''

import os
from typing import Union, Tuple, List

import tree_sitter
from tree_sitter import Language, Parser


def get_ancestor_type_chains(
        node: tree_sitter.Node
) -> List[str]:
    types = [str(node.type)]
    while node.parent is not None:
        node = node.parent
        types.append(str(node.type))
    return types


class TransformationBase:
    def __init__(
            self,
            parser_path: str,
            language: str
    ):
        if not os.path.exists(parser_path):
            raise ValueError(
                f"Language parser does not exist at {parser_path}. Please run `setup.sh` to properly set the "
                f"environment!")
        self.lang_object = Language(parser_path, language)
        self.parser = Parser()
        self.parser.set_language(self.lang_object)
        pass

    def parse_code(
            self,
            code: Union[str, bytes]
    ) -> tree_sitter.Node:
        """
        This function parses a given code and return the root node.
        :param code:
        :return: tree_sitter.Node, the root node of the parsed tree.
        """
        if isinstance(code, bytes):
            tree = self.parser.parse(code)
        elif isinstance(code, str):
            tree = self.parser.parse(code.encode())
        else:
            raise ValueError("Code must be character string or bytes string")
        return tree.root_node

    def get_tokens(
            self,
            code: bytes,
            root: tree_sitter.Node
    ) -> List[str]:
        """
        This function is for getting tokens recursively from a tree.
        :param code: the byte string corresponding to the code.
        :param root: the root node of the parsed tree
        :return: List of Tokens.
        """
        tokens = []
        if root.type == "comment":
            return tokens
        if "string" in str(root.type):
            parent = root.parent
            if "list" not in str(parent.type) and len(parent.children) == 1:
                return tokens
            else:
                return [code[root.start_byte:root.end_byte].decode()]
        if len(root.children) == 0:
            tokens.append(code[root.start_byte:root.end_byte].decode())
        else:
            for child in root.children:
                tokens += self.get_tokens(code, child)
        return tokens

    def get_token_string(
            self,
            code: str,
            root: tree_sitter.Node
    ) -> str:
        """
        This is a auxiliary function for just extracting the parsed token string.
        :param code: the byte string corresponding to the code.
        :param root: the root node of the parsed tree
        :return: str, the parsed code a string of tokens.
        """
        tokens = self.get_tokens(code.encode(), root)
        return " ".join(tokens)

    def get_tokens_with_node_type(
            self,
            code: bytes,
            root: tree_sitter.Node
    ) -> Tuple[List[str], List[List[str]]]:
        """
        This function extracts the tokens and types of the tokens.
        It returns a list of string as tokens, and a list of list of string as types.
        For every token, it extracts the sequence of ast node type starting from the token all the way to the root.
        :param code: the byte string corresponding to the code.
        :param root: the root node of the parsed tree
        :return:
            List[str]: The list of tokens.
            List[List[str]]: The AST node types corresponding to every token.
        """
        tokens, types = [], []
        if root.type == "comment":
            return tokens, types
        if "string" in str(root.type):
            return [code[root.start_byte:root.end_byte].decode()], [["string"]]
        if len(root.children) == 0:
            tokens.append(code[root.start_byte:root.end_byte].decode())
            types.append(get_ancestor_type_chains(root))
        else:
            for child in root.children:
                _tokens, _types = self.get_tokens_with_node_type(code, child)
                tokens += _tokens
                types += _types
        return tokens, types

    def transform_code(
            self,
            code: Union[str, bytes]
    ) -> Tuple[str, object]:
        """
        Transforms a piece of code and returns the transformed version
        :param code: The code to be transformed either as a character string of bytes string.
        :return:
            A tuple, where the first member is the transformed code.
            The second member might be other metadata (e.g. nde types) of the transformed code. It can be None as well.
        """
        pass
