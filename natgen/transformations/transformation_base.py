import os
from typing import Union, Tuple, List

import tree_sitter
from tree_sitter import Language, Parser, Node


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
        self.include_comments = True
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
        include_comments = self.include_comments
        tokens = []
        if root.type == "comment":
            if include_comments: 
                tokens.append(code[root.start_byte:root.end_byte].decode()) # append comments with #
                ed = root.end_byte
                while len(code) > ed and code[ed:ed+1].decode() == "\n":
                    tokens.append("NEWLINE")
                    ed += 1
            return tokens
        if "string" in str(root.type):
            parent = root.parent
            if "list" not in str(parent.type) and len(parent.children) == 1:
                if include_comments: 
                    tokens.append(code[root.start_byte:root.end_byte].decode()) # append comments with """
                    ed = root.end_byte
                    while len(code) > ed and code[ed:ed+1].decode() == "\n":
                        tokens.append("NEWLINE")
                        ed += 1
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
            ####### This is the key part to add NEWLINE #########
            ed = root.end_byte
            while len(code) > ed + 1 and code[ed:ed+1] == "\n".encode():
                tokens.append("NEWLINE")
                ed += 1
            ####### This is the key part to add NEWLINE #########
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


''' start of Amazon addition '''
class TransformationHelper(TransformationBase):
    """ Base class for renaming variables
    """
    def __init__(
            self,
            parser_path: str,
            language: str, 
    ):
        super(TransformationHelper, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.name = "TransformationHelper"
        self.TYPE_VARS = ["float", "list", "List", "int", "bool", "tuple", "str", "dict", "True", "False"]
        self.not_var_ptype = ["function_declarator", "class_declaration", "method_declaration", "function_definition",
                              "function_declaration", "call", "local_function_statement"]
    
    def get_func(self, code, root):
        # if root is not None:
        #     print(len(root.children) if root.children is not None else None, \
        #         root.parent.type if root.parent is not None else None, root.type, code[root.start_byte:root.end_byte])
        if isinstance(code, str):
            code = code.encode()
        assert isinstance(root, Node)
        if self.language == "java" or self.language == "c_sharp":
            if "identifier" in root.type and root.type != "type_identifier" and "method_declaration" in root.parent.type:
                # print(len(root.children) if root.children is not None else None, root.parent.type, root.type, code[root.start_byte:root.end_byte])
                return [code[root.start_byte:root.end_byte].decode()]
        elif self.language == "javascript" or self.language == "typescript":
            if "identifier" in root.type and "function_declaration" in root.parent.type:
                # print(len(root.children) if root.children is not None else None, root.parent.type, root.type, code[root.start_byte:root.end_byte])
                return [code[root.start_byte:root.end_byte].decode()]
        children = root.children
        if len(children) == 0:
            return []
        names = []
        for child in children:
            child_type = str(child.type)
            child_names = self.get_func(code, child)
            names += child_names
        return names

    def extract_func_names(self, code_string):
        root = self.parse_code(code_string)
        func_names = self.get_func(code_string, root)
        return func_names
