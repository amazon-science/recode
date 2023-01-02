import re

import numpy as np
from tree_sitter import Node

from .utils import get_tokens


class GoProcessor:
    @classmethod
    def create_dead_for_loop(cls, body):
        control_variable = "_i_" + str(np.random.choice(list(range(10))))
        return f"for {control_variable} := 0 ; {control_variable} < 0; {control_variable}++" + " { " + body + " } "

    @classmethod
    def create_dead_while_loop(cls, body):
        p = np.random.uniform(0, 1)
        control_variable = "_i_" + str(np.random.choice(list(range(10))))
        if p < 0.33:
            return f"for false " + " { " + body + " } "
        elif p < 0.66:
            return f"for {control_variable} > {control_variable} " + " { " + body + " } "
        else:
            return f"for {control_variable} < {control_variable} " + " { " + body + " } "

    @classmethod
    def create_dead_if(cls, body):
        p = np.random.uniform(0, 1)
        control_variable = "_i_" + str(np.random.choice(list(range(10))))
        if p < 0.33:
            return f"if false " + " { " + body + " } "
        elif p < 0.66:
            return f"if {control_variable} > {control_variable} " + " { " + body + " } "
        else:
            return f"if {control_variable} < {control_variable} " + " { " + body + " } "

    @classmethod
    def extract_expression(self, root, code):
        expressions = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'binary_expression':
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
    def get_tokens_for_opswap(cls, code, root, left_oprd, operator, right_oprd):
        if isinstance(code, str):
            code = code.encode()
        assert isinstance(root, Node)
        tokens = []
        
        if root.type == "comment":
            return tokens, None
        if "string" in str(root.type):
            return [code[root.start_byte:root.end_byte].decode()], None

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
        for child in children:
            if child.start_byte == left_oprd.start_byte and child.end_byte == left_oprd.end_byte:
                ts, _ = cls.get_tokens_for_opswap(code, right_oprd, left_oprd, operator, right_oprd)
            elif child.start_byte == right_oprd.start_byte and child.end_byte == right_oprd.end_byte:
                ts, _ = cls.get_tokens_for_opswap(code, left_oprd, left_oprd, operator, right_oprd)
            else:
                ts, _ = cls.get_tokens_for_opswap(code, child, left_oprd, operator, right_oprd)
            tokens += ts
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
            code_string = get_tokens(code_str, root)
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
                    if str(w.type) == "else":
                        flagx = 1
                    if str(w.type) == "if_statement":
                        not_consider.append(w)
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
                        if flag == 0:
                            first_block = current_node
                            flag = 1
                        else:
                            second_block = current_node

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
            code_string = cls.beautify_java_code(get_tokens(code_str, root))
        return code_string, success

    @classmethod
    def beautify_java_code(cls, tokens):
        code = " ".join(tokens)
        code = re.sub(" \\. ", "", code)
        code = re.sub(" \\+\\+", "++", code)
        return code
