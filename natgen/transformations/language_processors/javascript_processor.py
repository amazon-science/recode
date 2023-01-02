import re

import numpy as np
from tree_sitter import Node

from .utils import get_tokens


class JavascriptProcessor:
    @classmethod
    def create_dead_for_loop(cls, body):
        control_variable = "_i_" + str(np.random.choice(list(range(10))))
        p = np.random.uniform(0, 1)
        if p < 0.5:
            prefix = "for ( let " + control_variable + " = 0 ; " + control_variable + " > 0 ; " + control_variable + \
                     " ++ ) { "
            loop = prefix + body + " } "
            return loop
        else:
            return "for ( ; false ; ) { " + body + "}"

    @classmethod
    def create_dead_while_loop(cls, body):
        p = np.random.uniform(0, 1)
        control_variable = "_i_" + str(np.random.choice(list(range(10))))
        if p < 0.33:
            return "while ( false ) { " + body + " }"
        elif p < 0.66:
            return "while ( " + control_variable + " < " + control_variable + " ) { " + body + " } "
        else:
            return "while ( " + control_variable + " > " + control_variable + " ) { " + body + " } "

    @classmethod
    def create_dead_if(cls, body):
        p = np.random.uniform(0, 1)
        control_variable = "_i_" + str(np.random.choice(list(range(10))))
        if p < 0.33:
            return "if ( false ) { " + body + " }"
        elif p < 0.66:
            return "if ( " + control_variable + " < " + control_variable + " ) { " + body + " } "
        else:
            return "if ( " + control_variable + " > " + control_variable + " ) { " + body + " } "

    @classmethod
    def get_tokens_insert_before(cls, code_str, root, insertion_code, insert_before_node, include_comments=True):
        if not isinstance(insert_before_node, list):
            insert_before_node = [insert_before_node]
        if isinstance(code_str, str):
            code_str = code_str.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            if include_comments: 
                tokens.append(code_str[root.start_byte:root.end_byte].decode()) # append comments with #
                ed = root.end_byte
                while len(code_str) > ed + 1 and code_str[ed:ed+1] == "\n".encode():
                    tokens.append("NEWLINE")
                    ed += 1
            return tokens
        if "string" in str(root.type):
            return [code_str[root.start_byte:root.end_byte].decode()]
        if root in insert_before_node:
            tokens += insertion_code.split()
        children = root.children
        if len(children) == 0 or str(root.type) in ["string"]:
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            ####### This is the key part to add NEWLINE #########
            ed = root.end_byte
            while len(code_str) > ed + 1 and code_str[ed:ed+1] == "\n".encode():
                tokens.append("NEWLINE")
                ed += 1
            ####### This is the key part to add NEWLINE #########
        else:
            for child in children:
                ts = cls.get_tokens_insert_before(code_str, child, insertion_code, insert_before_node)
                tokens += ts
        return tokens

    @classmethod
    def get_tokens(cls, code, root, include_comments=True):
        # print(len(root.children) if root.children is not None else None, root.type, code[root.start_byte:root.end_byte])
        if isinstance(code, str):
            code = code.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            if include_comments: 
                tokens.append(code[root.start_byte:root.end_byte].decode()) # append comments with #
                ed = root.end_byte
                while len(code) > ed + 1 and code[ed:ed+1] == "\n".encode():
                    tokens.append("NEWLINE")
                    ed += 1
            return tokens
        if "string" in str(root.type):
            return [code[root.start_byte:root.end_byte].decode()]
        children = root.children
        if len(children) == 0 or str(root.type) in ["string"]:
            tokens.append(code[root.start_byte:root.end_byte].decode())
            ####### This is the key part to add NEWLINE #########
            ed = root.end_byte
            while len(code) > ed + 1 and code[ed:ed+1] == "\n".encode():
                tokens.append("NEWLINE")
                ed += 1
            ####### This is the key part to add NEWLINE #########
        else:
            for child in children:
                ts = cls.get_tokens(code, child)
                tokens += ts
        return tokens

    @classmethod
    def get_breaking_statements(cls, block):
        breakings = ['continue_statement', 'break_statement', 'return_statement']
        statements = []
        stack = [block]
        while len(stack) > 0:
            top = stack.pop()
            if str(top.type) in breakings:
                statements.append(top)
            else:
                for child in top.children:
                    stack.append(child)
        return statements

    @classmethod
    def for_to_while_random(cls, code_string, parser):
        root = parser.parse_code(code_string)
        loops = cls.extract_for_loops(root)
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
            ts = cls.get_tokens(code_string, root)
            code_string = cls.beautify_java_code(ts)
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
                ts = cls.get_tokens(code_string, root)
                code_string = cls.beautify_java_code(ts)
        except:
            pass
        return root, code_string, success

    @classmethod
    def extract_for_loops(cls, root):
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
    def beautify_java_code(cls, tokens):
        code = " ".join(tokens)
        # code = re.sub(" \\. ", "", code)
        # code = re.sub(" \\+\\+", "++", code)
        return code

    @classmethod
    def get_tokens_replace_for(cls, code_str, for_node, root, init, cond, update, body, include_comments=True):
        if isinstance(code_str, str):
            code_str = code_str.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            if include_comments: 
                tokens.append(code_str[root.start_byte:root.end_byte].decode()) # append comments with #
                ed = root.end_byte
                while len(code_str) > ed + 1 and code_str[ed:ed+1] == "\n".encode():
                    tokens.append("NEWLINE")
                    ed += 1
            return tokens
        if "string" in str(root.type):
            parent = root.parent
            return [code_str[root.start_byte:root.end_byte].decode()]
        children = root.children
        if len(children) == 0 or str(root.type) in ["string"]:
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            ####### This is the key part to add NEWLINE #########
            ed = root.end_byte
            while len(code_str) > ed + 1 and code_str[ed:ed+1] == "\n".encode():
                tokens.append("NEWLINE")
                ed += 1
            ####### This is the key part to add NEWLINE #########
        else:
            for child in children:
                if child == for_node:
                    tokens.extend(
                        init + [" NEWLINE ", "while", "("] + cond + [")", "{", " NEWLINE "] + body + update + ["}", " NEWLINE "]
                    )
                else:
                    tokens += cls.get_tokens_replace_for(code_str, for_node, child, init, cond, update, body)
        return tokens

    @classmethod
    def for_to_while(cls, code_string, root, fl, parser):
        children = fl.children
        init = children[2]
        init_tokens = cls.get_tokens(code_string, init)
        comparison = children[3]
        if str(comparison.type) != ";":
            comp_tokens = cls.get_tokens(code_string, comparison)
            if comp_tokens[-1] == ";":
                comp_tokens = comp_tokens[:-1]
            update = children[4]
            if str(update.type) == ")":
                update_tokens = []
                body = children[5]
            else:
                update_tokens = cls.get_tokens(code_string, update) + [";"]
                body = children[6]
            # body_tokens = cls.get_tokens(code_string, body)
            breaking_statements = cls.get_breaking_statements(body)
            body_tokens = cls.get_tokens_insert_before(
                code_string, body, " ".join(update_tokens), breaking_statements)
            if len(body_tokens) >= 2 and (body_tokens[0] == "{" and body_tokens[-1] == "}"):
                body_tokens = body_tokens[1:-1]
            tokens = cls.get_tokens_replace_for(
                code_str=code_string,
                for_node=fl,
                root=root,
                init=init_tokens,
                cond=comp_tokens,
                update=update_tokens,
                body=body_tokens
            )
            code = cls.beautify_java_code(tokens)
            return parser.parse_code(code), code, True
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
        children = wl.children
        condition = children[1]
        body = children[2]
        if str(condition.type) == 'parenthesized_expression':
            expr_tokens = cls.get_tokens(code_string, condition.children[1])
            body_tokens = cls.get_tokens(code_string, body)
            if len(body_tokens) >= 2 and (body_tokens[0] == "{" and body_tokens[-1] == "}"):
                body_tokens = body_tokens[1:-1]
            tokens = cls.get_tokens_replace_while(
                code_str=code_string,
                while_node=wl,
                root=root,
                cond=expr_tokens,
                body=body_tokens
            )
            code = cls.beautify_java_code(tokens)
            return parser.parse_code(code), code, True
        return root, code_string, False

    @classmethod
    def get_tokens_replace_while(cls, code_str, while_node, root, cond, body, include_comments=True):
        if isinstance(code_str, str):
            code_str = code_str.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            if include_comments: 
                tokens.append(code_str[root.start_byte:root.end_byte].decode()) # append comments with #
                ed = root.end_byte
                while len(code_str) > ed + 1 and code_str[ed:ed+1] == "\n".encode():
                    tokens.append("NEWLINE")
                    ed += 1
            return tokens
        if "string" in str(root.type):
            return [code_str[root.start_byte:root.end_byte].decode()]
        children = root.children
        if len(children) == 0 or str(root.type) in ["string"]:
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            ####### This is the key part to add NEWLINE #########
            ed = root.end_byte
            while len(code_str) > ed + 1 and code_str[ed:ed+1] == "\n".encode():
                tokens.append("NEWLINE")
                ed += 1
            ####### This is the key part to add NEWLINE #########
        else:
            for child in children:
                if child == while_node:
                    tokens.extend(
                        [" NEWLINE ", "for", "(", ";"] + cond + [";", ")", "{", " NEWLINE "] + body + ["}", " NEWLINE "]
                    )
                else:
                    tokens += cls.get_tokens_replace_while(code_str, while_node, child, cond, body)
        return tokens

    @classmethod
    def extract_expression(self, root, code):
        expressions = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'binary_expression':
                children_nodes = current_node.children
                keep = ["<", ">", "<=", ">=", "==", "!=", "===", "!=="]
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
                ed = root.end_byte
                while len(code) > ed + 1 and code[ed:ed+1] == "\n".encode():
                    tokens.append("NEWLINE")
                    ed += 1
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
                elif opt == '===':
                    tokens.append('===')
                elif opt == '!==':
                    tokens.append('!==')
            else:
                tokens.append(code[root.start_byte:root.end_byte].decode())
                ####### This is the key part to add NEWLINE #########
                ed = root.end_byte
                while len(code) > ed + 1 and code[ed:ed+1] == "\n".encode():
                    tokens.append("NEWLINE")
                    ed += 1
                ####### This is the key part to add NEWLINE #########
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
            code_string = cls.beautify_java_code(get_tokens(code_str, root))
        return code_string, success

    @classmethod
    def extract_if_else(cls, root, code_str, operator_list):
        ext_opt_list = ["&&", "&", "||", "|", "and", "or"]
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
                    if str(w.type) == "else" or str(w.type) == "else_clause":
                        flagx = 1
                        m = w.children
                        for x in m:
                            if str(x.type) == "if_statement":
                                not_consider.append(x)
                                flagy = 1
                                break
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

            elif child.start_byte == second_block.start_byte and child.end_byte == second_block.end_byte and flagy ==\
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
                    if str(current_node.type) == 'statement_block':
                        first_block = current_node
                    elif str(current_node.type) == 'else_clause':
                        new_list=current_node.children
                        for w in new_list:
                            if str(w.type) == 'statement_block':    
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
            code_string = cls.beautify_java_code(get_tokens(code_str, root))
        return code_string, success
