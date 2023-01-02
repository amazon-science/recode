'''
This file includes helper function for transformations on code structure. Many of the functions are built based on 
NatGEN (https://github.com/saikat107/NatGen).
Original Copyright 2021 Saikat Chakraborty. Licensed under the MIT License.
Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
'''

import numpy as np
import tokenize
from io import BytesIO
from tree_sitter import Node
from .utils import print_node, add_newline_token


class PythonProcessor:
    @classmethod
    def create_dead_for_loop(cls, body):
        control_variable = "_i_" + str(np.random.choice(list(range(10))))
        loop = f"NEWLINE for {control_variable} in range ( 0 ) : NEWLINE INDENT {body} NEWLINE DEDENT "
        return loop

    @classmethod
    def create_dead_while_loop(cls, body):
        p = np.random.uniform(0, 1)
        control_variable = "_i_" + str(np.random.choice(list(range(10))))
        if p < 0.33:
            return f"while False : NEWLINE INDENT {body} NEWLINE DEDENT"
        elif p < 0.66:
            return f"{control_variable} = 0 NEWLINE while {control_variable} < {control_variable} : NEWLINE INDENT {body} NEWLINE DEDENT"
        else:
            return f"{control_variable} = 0 NEWLINE while {control_variable} > {control_variable} : NEWLINE INDENT {body} NEWLINE DEDENT"

    @classmethod
    def create_dead_if(cls, body):
        p = np.random.uniform(0, 1)
        control_variable = "_i_" + str(np.random.choice(list(range(10))))
        if p < 0.33:
            return f"if False : NEWLINE INDENT {body} NEWLINE DEDENT"
        elif p < 0.66:
            return f"{control_variable} = 0 NEWLINE if {control_variable} < {control_variable} : NEWLINE INDENT {body} NEWLINE DEDENT"
        else:
            return f"{control_variable} = 0 NEWLINE if {control_variable} > {control_variable} : NEWLINE INDENT {body} NEWLINE DEDENT"

    @classmethod
    def get_tokens_insert_before(cls, code_str, root, insertion_code, insert_before_node, include_comments=True):
        if isinstance(code_str, str):
            code_str = code_str.encode()
        assert isinstance(root, Node)
        tokens = []
        # if root == insert_before_node:
        #     tokens += insertion_code.split()
        if root.type == "comment":
            if include_comments: 
                tokens.append(code_str[root.start_byte:root.end_byte].decode()) # append comments with #
                tokens += add_newline_token(code_str, root)
            return tokens
        if "string" in str(root.type):
            parent = root.parent
            if len(parent.children) == 1:
                if include_comments: 
                    tokens.append(code_str[root.start_byte:root.end_byte].decode()) # append comments with """
                    tokens += add_newline_token(code_str, root)
                return tokens
            else:
                return [code_str[root.start_byte:root.end_byte].decode()]
        if root.type == "decorator":
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            tokens.append("NEWLINE")
            return tokens
        if root == insert_before_node:
            tokens += insertion_code.split()
        children = root.children
        if len(children) == 0:
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            tokens += add_newline_token(code_str, root)
        for child in children:
            child_type = str(child.type)
            if child_type == "block":
                tokens += ["NEWLINE", "INDENT"]
            ts = cls.get_tokens_insert_before(code_str, child, insertion_code, insert_before_node)
            tokens += ts
            if child_type.endswith("statement"):
                tokens.append("NEWLINE")
            elif child_type == "block":
                tokens.append("DEDENT")
        # print(" ".join(tokens).replace("NEWLINE", "\n"))
        return tokens

    @classmethod
    def get_tokens(cls, code, root, include_comments=True):
        # print(len(root.children) if root.children is not None else None, root.type, print_node(code, root))
        if isinstance(code, str):
            code = code.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            if include_comments: 
                tokens.append(code[root.start_byte:root.end_byte].decode()) # append comments with #
                tokens += add_newline_token(code, root)
            return tokens
        if "string" in str(root.type):
            parent = root.parent
            if len(parent.children) == 1:
                if include_comments: 
                    tokens.append(code[root.start_byte:root.end_byte].decode()) # append comments with """
                    tokens += add_newline_token(code, root)
                return tokens
            else:
                return [code[root.start_byte:root.end_byte].decode()]
        if root.type == "decorator":
            tokens.append(code[root.start_byte:root.end_byte].decode())
            tokens.append("NEWLINE")
            return tokens
        children = root.children
        if len(children) == 0:
            tokens.append(code[root.start_byte:root.end_byte].decode())
            tokens += add_newline_token(code, root)
        for child in children:
            child_type = str(child.type)
            if child_type == "block":
                tokens += ["NEWLINE", "INDENT"]
            ts = cls.get_tokens(code, child)
            tokens += ts
            if child_type.endswith("statement"):
                tokens.append("NEWLINE")
            elif child_type == "block":
                tokens.append("DEDENT")
        # print(" ".join(tokens).replace("NEWLINE", "\n"))
        return tokens

    @classmethod
    def for_to_while_random(cls, code_string, parser):
        root = parser.parse_code(code_string)
        loops = cls.extract_for_loops(root, code_string)
        success = False
        try:
            while not success and len(loops) > 0:
                selected_loop = np.random.choice(loops)
                loops.remove(selected_loop)
                modified_root, modified_code_string, success = cls.for_to_while(
                    code_string, root, selected_loop, parser
                )
                if success:
                    root = modified_root
                    code_string = modified_code_string
        except:
            pass
        if not success:
            code_string = cls.beautify_python_code(cls.get_tokens(code_string, root))
        else:
            code_string = code_string.replace("\n", " NEWLINE ")
            code_string = cls.beautify_python_code(code_string.split())
        return root, code_string, success

    ''' Start of Amazon addition '''
    @classmethod
    def for_to_while_first(cls, code_string, parser):
        root = parser.parse_code(code_string)
        loops = cls.extract_for_loops(root, code_string)
        success = False
        try:
            while not success and len(loops) > 0:
                selected_loop = loops[0]
                loops.remove(selected_loop)
                modified_root, modified_code_string, success = cls.for_to_while(
                    code_string, root, selected_loop, parser
                )
                if success:
                    root = modified_root
                    code_string = modified_code_string
        except:
            pass
        if not success:
            code_string = cls.beautify_python_code(cls.get_tokens(code_string, root))
        else:
            code_string = cls.beautify_python_code(code_string.split())
        return root, code_string, success


    @classmethod
    def while_to_for_random(cls, code_string, parser):
        root = parser.parse_code(code_string)
        loops = cls.extract_while_loops(root)
        success = False
        try:
            while not success and len(loops) > 0:
                selected_loop = np.random.choice(loops)
                loops.remove(selected_loop)
                modified_root, modified_code_string, success = cls.while_to_for(
                    code_string, root, selected_loop, parser
                )
                if success:
                    root = modified_root
                    code_string = modified_code_string
            if not success:
                code_string = code_string.replace("\n", " NEWLINE ")
                code_string = cls.beautify_python_code(cls.get_tokens(code_string, root))
            else:
                code_string = cls.beautify_python_code(code_string.split())
        except:
            pass
        return root, code_string, False

    @classmethod
    def while_to_for_first(cls, code_string, parser):
        root = parser.parse_code(code_string)
        loops = cls.extract_while_loops(root)
        success = False
        try:
            while not success and len(loops) > 0:
                selected_loop = loops[0]
                loops.remove(selected_loop)
                modified_root, modified_code_string, success = cls.while_to_for(
                    code_string, root, selected_loop, parser
                )
                if success:
                    root = modified_root
                    code_string = modified_code_string
            if not success:
                code_string = cls.beautify_python_code(cls.get_tokens(code_string, root))
            else:
                code_string = cls.beautify_python_code(code_string.split())
        except:
            pass
        return root, code_string, False
    ''' End of Amazon addition '''

    @classmethod
    def extract_for_loops(cls, root, code_str):
        loops = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'for_statement':
                loops.append(current_node)
            for child in current_node.children:
                queue.append(child)
        return loops

    @classmethod
    def beautify_python_code(cls, tokens):
        indent_count = 0
        code = ""
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "NEWLINE":
                code += "\n"
                for _ in range(indent_count):
                    code += "\t"
            elif token == "INDENT":
                indent_count += 1
                code += "\t"
            elif token == "DEDENT":
                indent_count -= 1
                if code[-1] == "\t":
                    code = code[:-1]
            else:
                code += token + " "
            i += 1
        lines = code.split("\n")
        taken_lines = []
        for line in lines:
            if len(line.strip()) > 0:
                taken_lines.append(line.rstrip())
        code = "\n".join(taken_lines)
        return code

    @classmethod
    def get_tokens_replace_for(cls, code_str, for_node, root, while_node, include_comments=True):
        if isinstance(code_str, str):
            code_str = code_str.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            if include_comments:
                tokens.append(code_str[root.start_byte:root.end_byte].decode()) # append comments with #
                tokens += add_newline_token(code_str, root)
            return tokens
        if "string" in str(root.type):
            parent = root.parent
            if len(parent.children) == 1:
                if include_comments:
                    tokens.append(code_str[root.start_byte:root.end_byte].decode()) # append comments with """
                    tokens += add_newline_token(code_str, root)
                return tokens
            else:
                return [code_str[root.start_byte:root.end_byte].decode()]
        if root.type == "decorator":
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            tokens.append("NEWLINE")
            return tokens
        children = root.children
        if len(children) == 0:
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            tokens += add_newline_token(code_str, root)
        for child in children:
            if child == for_node:
                tokens += while_node
            else:
                child_type = str(child.type)
                if child_type == "block":
                    tokens += ["NEWLINE", "INDENT"]
                tokens += cls.get_tokens_replace_for(code_str, for_node, child, while_node)
                if child_type.endswith("statement"):
                    tokens.append("NEWLINE")
                elif child_type == "block":
                    tokens.append("DEDENT")
        return tokens

    @classmethod
    def for_to_while(cls, code_string, root, fl, parser):
        if "range" in print_node(code_string, fl):
            try:
                identifier = fl.children[1]
                in_node = fl.children[2]
                range_node = fl.children[3]
                body_node = fl.children[5]
                range_function = range_node.children[0]
                range_function_name = cls.get_tokens(code_string, range_function)[0]
                stop_only = False
                if range_function_name == "range" \
                        and (str(identifier.type) == "identifier" and len(identifier.children) == 0) \
                        and (str(in_node.type) == "in" and len(in_node.children) == 0):
                    argument_list = range_node.children[1].children
                    args = []
                    for a in argument_list:
                        k = str(a.type)
                        if k not in ["(", ",", ")"]:
                            args.append(a)
                    start, stop, step = ["0"], ["0"], ["1"]
                    if len(args) == 1:
                        stop = cls.get_tokens(code_string, args[0])
                        stop_only = True
                    elif len(args) == 2:
                        start = cls.get_tokens(code_string, args[0])
                        stop = cls.get_tokens(code_string, args[1])
                    else:
                        start = cls.get_tokens(code_string, args[0])
                        stop = cls.get_tokens(code_string, args[1])
                        step = cls.get_tokens(code_string, args[2])
                    identifier_name = cls.get_tokens(code_string, identifier)[0]
                    if step[0] != "-":
                        while_stmt = [identifier_name, "="] + start + ["NEWLINE"] + \
                                    ["while", identifier_name, "<"] + stop + \
                                    [":", "NEWLINE", "INDENT"] + \
                                    cls.get_tokens(code_string, body_node) + ["NEWLINE", identifier_name, "+="] + step + \
                                    ["DEDENT", "NEWLINE"]
                    else:
                        while_stmt = [identifier_name, "="] + start + ["NEWLINE"] + \
                                    ["while", identifier_name, ">"] + stop + \
                                    [":", "NEWLINE", "INDENT"] + \
                                    cls.get_tokens(code_string, body_node) + ["NEWLINE", identifier_name, "-="] + step[1:] + \
                                    ["DEDENT", "NEWLINE"]
                    tokens = cls.get_tokens_replace_for(
                        code_str=code_string,
                        for_node=fl,
                        while_node=while_stmt,
                        root=root
                    )
                    code = cls.beautify_python_code(tokens)
                    return parser.parse_code(code), " ".join(tokens), True
            except:
                pass

        elif "in" in print_node(code_string, fl.children[2]):
            # for x in x_list...
            try:
                identifier = fl.children[1]
                in_node = fl.children[2]
                range_node = fl.children[3]
                body_node = fl.children[5]
                if (str(identifier.type) == "identifier" and len(identifier.children) == 0) \
                        and (str(in_node.type) == "in" and len(in_node.children) == 0):
                    identifier_name = cls.get_tokens(code_string, identifier)[0]
                    range_node_name = print_node(code_string, range_node)
                    invariance_name = "_"+identifier_name+"_i"
                    # put the invariance += 1 after loop body to make the problem harder
                    while_stmt =  [invariance_name, "=", "0"] + \
                                    ["while", invariance_name, "<", "len", "(", range_node_name, ")", ":", "NEWLINE", "INDENT"] + \
                                    [identifier_name, "=", range_node_name, "[", invariance_name, "]", "NEWLINE"] + \
                                    cls.get_tokens(code_string, body_node) + ["NEWLINE", invariance_name, "+="] + ["1"] + \
                                    ["DEDENT", "NEWLINE"]
                    tokens = cls.get_tokens_replace_for(
                        code_str=code_string,
                        for_node=fl,
                        while_node=while_stmt,
                        root=root
                    )
                    code = cls.beautify_python_code(tokens)
                    return parser.parse_code(code), " ".join(tokens), True
            except:
                pass
        return root, code_string, False

    @classmethod
    def extract_while_loops(cls, root):
        loops = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'while_statement':
                loops.append(current_node)
            for child in current_node.children:
                queue.append(child)
        return loops

    @classmethod
    def while_to_for(cls, code_string, root, wl, parser):
        # children = wl.children
        # condition = children[1]
        # body = children[2]
        # if str(condition.type) == 'parenthesized_expression':
        #     expr_tokens = get_tokens(code_string, condition.children[1])
        #     body_tokens = get_tokens(code_string, body)
        #     if len(body_tokens) >= 2 and (body_tokens[0] == "{" and body_tokens[-1] == "}"):
        #         body_tokens = body_tokens[1:-1]
        #     tokens = cls.get_tokens_replace_while(
        #         code_str=code_string,
        #         while_node=wl,
        #         root=root,
        #         cond=expr_tokens,
        #         body=body_tokens
        #     )
        #     code = cls.beautify_python_code(tokens)
        #     return parser.parse_code(code), code, True
        return root, code_string, False

    @classmethod
    def get_tokens_replace_while(cls, code_str, while_node, root, cond, body):
        # if isinstance(code_str, str):
        #     code_str = code_str.encode()
        # assert isinstance(root, Node)
        # tokens = []
        # children = root.children
        # if len(children) == 0:
        #     tokens.append(code_str[root.start_byte:root.end_byte].decode())
        # for child in children:
        #     if child == while_node:
        #         tokens.extend(
        #             ["for", "(", ";"] + cond + [";", ")", "{"] + body + ["}"]
        #         )
        #     else:
        #         tokens += cls.get_tokens_replace_while(code_str, while_node, child, cond, body)
        # return tokens
        raise NotImplementedError

    @classmethod
    def extract_expression(self, root, code):
        expressions = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'comparison_operator':
                children_nodes = current_node.children
                keep = ["<", ">", "<=", ">=", "==", "!="]
                counter = 0
                for w in children_nodes:
                    if str(w.type) in keep:
                        counter = counter + 1
                if counter == 1:
                    expressions.append(current_node)
            for child in current_node.children:
                queue.append(child)
        return expressions

    @classmethod
    def get_tokens_for_opswap(cls, code, root, left_oprd, operator, right_oprd, include_comments=True):
        if isinstance(code, str):
            code = code.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            if include_comments: 
                tokens.append(code[root.start_byte:root.end_byte].decode()) # append comments with #
                tokens += add_newline_token(code, root)
            return tokens, None
        if "string" in str(root.type):
            parent = root.parent
            if len(parent.children) == 1:
                if include_comments: 
                    tokens.append(code[root.start_byte:root.end_byte].decode()) # append comments with """
                    tokens += add_newline_token(code, root)
                return tokens, None
            else:
                return [code[root.start_byte:root.end_byte].decode()], None
        if root.type == "decorator":
            # print(root.type, code[root.start_byte:root.end_byte].decode())
            tokens.append(code[root.start_byte:root.end_byte].decode())
            tokens.append("NEWLINE")
            return tokens, None
        children = root.children
        if len(children) == 0:
            if root.start_byte == operator.start_byte and root.end_byte == operator.end_byte:
                opt = (code[operator.start_byte:operator.end_byte].decode())
                if opt == '<':
                    tokens.append('>')
                elif opt == '>':
                    tokens.append('<')
                elif opt == '>=':
                    tokens.append('<=')
                elif opt == '<=':
                    tokens.append('>=')
                elif opt == '==':
                    tokens.append('==')
                elif opt == '!=':
                    tokens.append('!=')
            else:
                tokens.append(code[root.start_byte:root.end_byte].decode())
                tokens += add_newline_token(code, root)
        for child in children:
            child_type = str(child.type)
            if child_type == "block":
                tokens += ["NEWLINE", "INDENT"]
            if child.start_byte == left_oprd.start_byte and child.end_byte == left_oprd.end_byte:
                ts, _ = cls.get_tokens_for_opswap(code, right_oprd, left_oprd, operator, right_oprd)
            elif child.start_byte == right_oprd.start_byte and child.end_byte == right_oprd.end_byte:
                ts, _ = cls.get_tokens_for_opswap(code, left_oprd, left_oprd, operator, right_oprd)
            else:
                ts, _ = cls.get_tokens_for_opswap(code, child, left_oprd, operator, right_oprd)
            tokens += ts
            if child_type.endswith("statement"):
                tokens.append("NEWLINE")
            elif child_type == "block":
                tokens.append("DEDENT")
        return tokens, None

    @classmethod
    def operand_swap(cls, code_str, parser):
        code = code_str.encode()
        root = parser.parse_code(code)
        expressions = cls.extract_expression(root, code)
        success = False
        try:
            while not success and len(expressions) > 0:
                selected_exp = np.random.choice(expressions)
                expressions.remove(selected_exp)
                bin_exp = selected_exp
                condition = code[bin_exp.start_byte:bin_exp.end_byte].decode()
                bin_exp = bin_exp.children
                left_oprd = bin_exp[0]
                operator = bin_exp[1]
                right_oprd = bin_exp[2]
                try:
                    # code = code.replace("\n", " NEWLINE ")
                    code_list = cls.get_tokens_for_opswap(code, root, left_oprd, operator, right_oprd)[0]
                    code_string = ""
                    for w in code_list:
                        code_string = code_string + w + " "
                    code_string = code_string.strip()
                    success = True
                except:
                    success = False
                    continue
        except:
            pass
        if not success:
            code_string = cls.beautify_python_code(cls.get_tokens(code_str, root))
        else:
            code_string = code_string.replace("\n", " NEWLINE ")
            code_string = cls.beautify_python_code(code_string.split())
        return code_string, success

    @classmethod
    def operand_swap_first(cls, code_str, parser):
        code = code_str.encode()
        root = parser.parse_code(code)
        expressions = cls.extract_expression(root, code)
        success = False
        try:
            while not success and len(expressions) > 0:
                selected_exp = expressions[0]
                expressions.remove(selected_exp)
                bin_exp = selected_exp
                condition = code[bin_exp.start_byte:bin_exp.end_byte].decode()
                bin_exp = bin_exp.children
                left_oprd = bin_exp[0]
                operator = bin_exp[1]
                right_oprd = bin_exp[2]
                try:
                    # code = code.replace("\n", " NEWLINE ")
                    code_list = cls.get_tokens_for_opswap(code, root, left_oprd, operator, right_oprd)[0]
                    code_string = ""
                    for w in code_list:
                        code_string = code_string + w + " "
                    code_string = code_string.strip()
                    success = True
                except:
                    success = False
                    continue
        except:
            pass
        if not success:
            code_string = cls.beautify_python_code(cls.get_tokens(code_str, root))
        else:
            code_string = code_string.replace("\n", " NEWLINE ")
            code_string = cls.beautify_python_code(code_string.split())
        return code_string, success

    @classmethod
    def extract_if_else(cls, root, code_str, operator_list):
        ext_opt_list = ["&&", "&", "||", "|"]
        expressions = []
        queue = [root]
        not_consider = []
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'if_statement':
                clause = code_str[current_node.start_byte:current_node.end_byte].decode()
                des = (current_node.children)[1]
                cond = code_str[des.start_byte:des.end_byte].decode()
                stack = [des]
                nodes = []
                while len(stack) > 0:
                    root1 = stack.pop()
                    if len(root1.children) == 0:
                        nodes.append(root1)
                    for child in root1.children:
                        stack.append(child)
                nodes.reverse()
                counter = 0
                extra_counter = 0
                for w in nodes:
                    if str(w.type) in operator_list:
                        counter = counter + 1
                    if str(w.type) in ext_opt_list:
                        extra_counter = extra_counter + 1
                if not (counter == 1 and extra_counter == 0):
                    continue
                children_nodes = current_node.children
                flagx = 0
                flagy = 0
                for w in children_nodes:
                    if str(w.type) == "else_clause":
                        flagx = 1
                    if str(w.type) == "elif_clause":
                        flagy = 1
                if flagx == 1 and flagy == 0:
                    expressions.append([current_node, des])
            for child in current_node.children:
                if child not in not_consider:
                    queue.append(child)

        return expressions

    @classmethod
    def get_tokens_for_blockswap(cls, code, root, first_block, opt_node, second_block, flagx, flagy):
        if isinstance(code, str):
            code = code.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            return tokens, None
        if "string" in str(root.type):
            parent = root.parent
            if len(parent.children) == 1:
                return tokens, None
            else:
                return [code[root.start_byte:root.end_byte].decode()], None
        children = root.children
        if len(children) == 0:
            if root.start_byte == opt_node.start_byte and root.end_byte == opt_node.end_byte:
                op = code[root.start_byte:root.end_byte].decode()
                if op == "<":
                    tokens.append(">=")
                elif op == ">":
                    tokens.append('<=')
                elif op == ">=":
                    tokens.append('<')
                elif op == "<=":
                    tokens.append('>')
                elif op == "!=":
                    tokens.append('==')
                elif op == "==":
                    tokens.append('!=')
            else:
                tokens.append(code[root.start_byte:root.end_byte].decode())
        for child in children:
            child_type = str(child.type)
            if child_type == "block":
                tokens += ["NEWLINE", "INDENT"]
            if child.start_byte == first_block.start_byte and child.end_byte == first_block.end_byte and flagx == 0 \
                    and str(
                child.type) == str(first_block.type):
                flagx = 1
                ts, _ = cls.get_tokens_for_blockswap(code, second_block, first_block, opt_node, second_block, flagx,
                                                     flagy)
            elif child.start_byte == second_block.start_byte and child.end_byte == second_block.end_byte and flagy == \
                    0 and str(
                child.type) == str(second_block.type):
                flagy = 1
                ts, _ = cls.get_tokens_for_blockswap(code, first_block, first_block, opt_node, second_block, flagx,
                                                     flagy)
            else:
                ts, _ = cls.get_tokens_for_blockswap(code, child, first_block, opt_node, second_block, flagx, flagy)
            tokens += ts
            if child_type.endswith("statement"):
                tokens.append("NEWLINE")
            elif child_type == "block":
                tokens.append("DEDENT")
        return tokens, None

    @classmethod
    def block_swap(cls, code_str, parser):
        code = code_str.encode()
        root = parser.parse_code(code)
        operator_list = ['<', '>', '<=', '>=', '==', '!=']
        pair = cls.extract_if_else(root, code, operator_list)
        success = False
        lst = list(range(0, len(pair)))
        try:
            while not success and len(lst) > 0:
                selected = np.random.choice(lst)
                lst.remove(selected)
                clause = pair[selected][0]
                des = pair[selected][1]
                st = [des]
                nodes = []
                while len(st) > 0:
                    root1 = st.pop()
                    if len(root1.children) == 0:
                        nodes.append(root1)
                        if (code[root1.start_byte:root1.end_byte].decode()) in operator_list:
                            opt_node = root1
                            break
                    for child in root1.children:
                        st.append(child)
                nodes = clause.children
                flag = 0
                for current_node in nodes:
                    if str(current_node.type) == 'block':
                        first_block = current_node
                    elif str(current_node.type) == 'else_clause':
                        new_list = current_node.children
                        for w in new_list:
                            if str(w.type) == 'block':
                                second_block = w
                                break
                flagx = 0
                flagy = 0
                try:
                    code_list = \
                        cls.get_tokens_for_blockswap(code, root, first_block, opt_node, second_block, flagx, flagy)[0]
                    code_string = ""
                    for w in code_list:
                        code_string = code_string + w + " "
                    code_string = code_string.strip()
                    success = True
                except:
                    success = False
                    continue
        except:
            pass
        if not success:
            code_string = cls.beautify_python_code(cls.get_tokens(code_str, root))
        else:
            code_string = cls.beautify_python_code(code_string.split())
        return code_string, success


def get_python_tokens(code, root=None):
    if isinstance(code, bytes):
        code = code.decode()
    tokens = []
    for token in tokenize.tokenize(BytesIO(code.encode("utf-8")).readline):
        if token.type == 0 or token.type >= 58:
            continue
        elif token.type == 4:
            tokens.append("NEWLINE")
        elif token.type == 5:
            tokens.append("INDENT")
        elif token.type == 6:
            tokens.append("DEDENT")
        else:
            tokens.append(token.string)
    return tokens, None
