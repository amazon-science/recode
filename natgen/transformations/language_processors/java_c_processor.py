import re

import numpy as np
from tree_sitter import Node

from .utils import get_tokens


class JavaAndCPPProcessor:

    @classmethod
    def get_tokens(cls, code_str, root, include_comments=True):
        """ Get all the tokens
        """
        # print(len(root.children) if root.children is not None else None, root.type, code_str[root.start_byte:root.end_byte])
        if isinstance(code_str, str):
            code_str = code_str.encode()
        assert isinstance(root, Node)
        tokens = []
        if "string" in str(root.type):
            # return [code_str[root.start_byte:root.end_byte].decode()]
            parent = root.parent
            if len(parent.children) == 1:
                if include_comments: 
                    tokens.append(code_str[root.start_byte:root.end_byte].decode()) # append comments with """
                return tokens
            else:
                return [code_str[root.start_byte:root.end_byte].decode()]
        if root.type == "decorator":
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            tokens.append("NEWLINE")
            return tokens
        children = root.children
        # if root.type == ";": 
        #     tokens.append(";")
        #     import pdb; pdb.set_trace()
        #     return tokens
        if len(children) == 0:
            token = code_str[root.start_byte:root.end_byte].decode().strip()
            tokens.append(token)
            ####### This is the key part to add NEWLINE #########
            ed = root.end_byte
            while len(code_str) > ed + 1 and code_str[ed:ed+1] == "\n".encode():
                tokens.append("NEWLINE")
                ed += 1
            ####### This is the key part to add NEWLINE #########
        for child in children:
            tokens += cls.get_tokens(code_str, child)
        return tokens


    @classmethod
    def create_dead_for_loop(cls, body):
        control_variable = "_i_" + str(np.random.choice(list(range(10))))
        p = np.random.uniform(0, 1)
        if p < 0.5:
            prefix = "for ( int " + control_variable + " = 0 ; " + control_variable + " > 0 ; " + control_variable + \
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
    def for_to_while_random(cls, code_string, parser):
        root = parser.parse_code(code_string)
        loops = cls.extract_for_loops(root)
        success = False
        try:
            while not success and len(loops) > 0:
                selected_loop = np.random.choice(loops)
                loops.remove(selected_loop)
                modified_root, modified_code_string, success = JavaAndCPPProcessor.for_to_while(
                    code_string, root, selected_loop, parser
                )
                if success:
                    root = modified_root
                    code_string = modified_code_string
        except:
            pass
        if not success:
            code_string = cls.beautify_java_code(get_tokens(code_string, root))
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
                modified_root, modified_code_string, success = JavaAndCPPProcessor.while_to_for(
                    code_string, root, selected_loop, parser
                )
                if success:
                    root = modified_root
                    code_string = modified_code_string
            if not success:
                code_string = cls.beautify_java_code(get_tokens(code_string, root))
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
    def get_tokens_replace_for(cls, code_str, for_node, root, init, cond, update, body):
        # print(len(root.children) if root.children is not None else None, root.type, code_str[root.start_byte:root.end_byte])
        if isinstance(code_str, str):
            code_str = code_str.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            return tokens
        if "string" in str(root.type):
            return [code_str[root.start_byte:root.end_byte].decode()]
        children = root.children
        if len(children) == 0:
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            ####### This is the key part to add NEWLINE #########
            ed = root.end_byte
            while len(code_str) > ed + 1 and code_str[ed:ed+1] == "\n".encode():
                tokens.append("NEWLINE")
                ed += 1
            ####### This is the key part to add NEWLINE #########
        for child in children:
            if child == for_node:
                tokens.extend(
                    init + ["while", "("] + cond + [")", "{", " NEWLINE "] + body + update + ["}", " NEWLINE "]
                )
            else:
                tokens += JavaAndCPPProcessor.get_tokens_replace_for(code_str, for_node, child, init, cond, update,
                                                                     body)
        return tokens

    @classmethod
    def extract_for_contents(cls, for_loop, code_string):
        children = for_loop.children
        init_part = children[2]
        if str(init_part.type).endswith("expression"):
            next_part_start = 4
            init_tokens = get_tokens(code_string, init_part) + [";"]
        elif str(init_part.type).endswith("statement") or str(init_part.type).endswith("declaration"):
            next_part_start = 3
            init_tokens = get_tokens(code_string, init_part)
        else:
            next_part_start = 3
            init_tokens = []
        comp_part = children[next_part_start]
        if str(comp_part.type).endswith("expression"):
            next_part_start += 2
            comp_tokens = get_tokens(code_string, comp_part)
        else:
            comp_tokens = ["true"]
            next_part_start += 1
        update_part = children[next_part_start]
        if str(update_part.type).endswith("expression"):
            next_part_start += 2
            update_tokens = get_tokens(code_string, update_part) + [";"]
        else:
            update_tokens = []
            next_part_start += 1
        block_part = children[next_part_start]
        breaking_statements = cls.get_breaking_statements(block_part)
        block_tokens = cls.get_tokens_insert_before(
            code_string, block_part, " ".join(update_tokens), breaking_statements)
        return init_tokens, comp_tokens, update_tokens, block_tokens

    @classmethod
    def get_tokens_insert_before(cls, code_str, root, insertion_code, insert_before_node):
        if not isinstance(insert_before_node, list):
            insert_before_node = [insert_before_node]
        if isinstance(code_str, str):
            code_str = code_str.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            return tokens
        if "string" in str(root.type):
            parent = root.parent
            if len(parent.children) == 1:
                return tokens
            else:
                return [code_str[root.start_byte:root.end_byte].decode()]
        if root in insert_before_node:
            tokens += insertion_code.split()
        children = root.children
        if len(children) == 0:
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            ####### This is the key part to add NEWLINE #########
            ed = root.end_byte
            while len(code_str) > ed + 1 and code_str[ed:ed+1] == "\n".encode():
                tokens.append("NEWLINE")
                ed += 1
            ####### This is the key part to add NEWLINE #########
        for child in children:
            ts = cls.get_tokens_insert_before(code_str, child, insertion_code, insert_before_node)
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
    def for_to_while(cls, code_string, root, fl, parser):
        original_tokenized_code = " ".join(get_tokens(code_string, root))
        init_tokens, comp_tokens, update_tokens, body_tokens = cls.extract_for_contents(fl, code_string)
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
        if original_tokenized_code == " ".join(tokens):
            return root, original_tokenized_code, False
        code = cls.beautify_java_code(tokens)
        return parser.parse_code(code), code, True

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
            expr_tokens = get_tokens(code_string, condition.children[1])
            body_tokens = get_tokens(code_string, body)
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
    def get_tokens_replace_while(cls, code_str, while_node, root, cond, body):
        # print(len(root.children) if root.children is not None else None, root.type, code_str[root.start_byte:root.end_byte])
        if isinstance(code_str, str):
            code_str = code_str.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            return tokens
        if "string" in str(root.type):
            return [code_str[root.start_byte:root.end_byte].decode()]
        children = root.children
        if len(children) == 0:
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
            ####### This is the key part to add NEWLINE #########
            ed = root.end_byte
            while len(code_str) > ed + 1 and code_str[ed:ed+1] == "\n".encode():
                tokens.append("NEWLINE")
                ed += 1
            ####### This is the key part to add NEWLINE #########
        for child in children:
            if child == while_node:
                tokens.extend(
                    ["for", "(", ";"] + cond + [";", ")", "{", " NEWLINE "] + body + ["}", " NEWLINE "]
                )
            else:
                tokens += JavaAndCPPProcessor.get_tokens_replace_while(code_str, while_node, child, cond, body)
        return tokens

    # -----Confusion removal C------
    @classmethod
    def conditional_removal(cls, code_string, parser):
        # This function is for C, equavalent to extract_ternary_expression for Java
        root = parser.parse_code(code_string)
        assi_cond_expr, varde_cond_expr, ret_cond_expr = cls.extract_conditional_expression(root)
        success = False
        if len(assi_cond_expr) > 0:
            try:
                modified_tokens = cls.assignment_conditional_removal(code_string, assi_cond_expr, root, parser)
                code_string = cls.beautify_java_code(modified_tokens)
                root = parser.parse_code(code_string)
                success = True
                _, varde_cond_expr, ret_cond_expr = cls.extract_conditional_expression(root)
            except:
                # print("assignment ternary expression removal failed.")
                pass


        if len(varde_cond_expr) > 0:
            try:
                modified_tokens = cls.var_decl_ternary_removal(code_string, varde_cond_expr, root, parser)
                code_string = cls.beautify_java_code(modified_tokens)
                root = parser.parse_code(code_string)
                success = True
                _, _, ret_cond_expr = cls.extract_conditional_expression(root)
            except:
                # print("variable declaration ternary expression removal failed.")
                pass

        if len(ret_cond_expr) > 0:
            try:
                modified_tokens = cls.return_ternary_removal(code_string, ret_cond_expr, root, parser)
                code_string = cls.beautify_java_code(modified_tokens)
                root = parser.parse_code(code_string)
                success = True
            except:
                # print("return ternary expression removal failed.")
                pass

        return root, code_string, success

    @classmethod
    def assignment_conditional_removal(cls, code_string, assi_tern_expr, root, parser):
        if isinstance(code_string, str):
            code_string = code_string.encode()
        assert isinstance(root, Node)
        tokens = []
        children = root.children
        if len(children) == 0:
            tokens.append(cls.handle_terminal_node(root, code_string))
        for child in children:
            if child in assi_tern_expr:
                if str(child.children[0].type) == "conditional_expression":
                    cond_children = child.children[0].children
                    if str(cond_children[0].type) == "assignment_expression":
                        assignee_token = get_tokens(code_string, cond_children[0].children[0])[0]
                        condition_tokens = get_tokens(code_string, cond_children[0].children[2])
                        if str(cond_children[0].children[2].type) == 'parenthesized_expression':
                            condition_tokens = condition_tokens[1:-1]
                        br1_tokens = get_tokens(code_string, cond_children[2])
                        br2_tokens = get_tokens(code_string, cond_children[4])
                        tokens.extend(["if", "("] + condition_tokens + [")", "{", assignee_token, "="] + br1_tokens +
                                      [";", "}", "else", "{", assignee_token, "="] + br2_tokens + [";", "}"])
            else:
                tokens += JavaAndCPPProcessor.assignment_conditional_removal(code_string, assi_tern_expr, child, parser)
        return tokens

    @classmethod
    def extract_conditional_expression(cls, root):
        assi_con_expr = []
        varde_con_expr = []
        ret_con_expr = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'conditional_expression' and str(
                    current_node.children[0].type) == "assignment_expression":
                assi_con_expr.append(current_node.parent)  # include the ";" for now and later we skip it
            if str(current_node.type) == 'conditional_expression' and str(
                    current_node.parent.type) == "init_declarator":
                varde_con_expr.append(current_node.parent.parent)  # node type: declaration
            if str(current_node.type) == 'conditional_expression' and str(
                    current_node.parent.type) == "return_statement":
                ret_con_expr.append(current_node.parent)
            for child in current_node.children:
                queue.append(child)
        return assi_con_expr, varde_con_expr, ret_con_expr

    # -----Confusion removal Java------
    # TODO: Check whether java/C/CPP have the same "ternary_expression" node type
    @classmethod
    def ternary_removal(cls, code_string, parser):
        code_string = cls.remove_package_and_import(code_string)  # TODO: Check whether this will mess up the code
        root = parser.parse_code(code_string)
        assi_tern_expr, varde_tern_expr, ret_tern_expr = cls.extract_ternary_expression(root)
        success = False
        if len(assi_tern_expr) > 0:
            try:
                modified_tokens = cls.assignment_ternary_removal(code_string, assi_tern_expr, root, parser)
                code_string = cls.beautify_java_code(modified_tokens)
                root = parser.parse_code(code_string)
                success = True
                _, varde_tern_expr, ret_tern_expr = cls.extract_ternary_expression(root)
            except:
                # print("assignment ternary expression removal failed.")
                pass

        if len(varde_tern_expr) > 0:
            try:
                modified_tokens = cls.var_decl_ternary_removal(code_string, varde_tern_expr, root, parser)
                code_string = cls.beautify_java_code(modified_tokens)
                root = parser.parse_code(code_string)
                success = True
                _, _, ret_tern_expr = cls.extract_ternary_expression(root)
            except:
                pass
                # print("variable declaration ternary expression removal failed.")


        if len(ret_tern_expr) > 0:
            try:
                modified_tokens = cls.return_ternary_removal(code_string, ret_tern_expr, root, parser)
                code_string = cls.beautify_java_code(modified_tokens)
                root = parser.parse_code(code_string)
                success = True
            except:
                pass
                # print("return ternary expression removal failed.")

        return root, code_string, success

    @classmethod
    def ternary_body_write(cls, body, code_string, assignee, tokens, ret=False):
        body_children = body.children
        condition_tokens = get_tokens(code_string, body_children[0])
        if str(body_children[0].type) == 'parenthesized_expression':
            condition_tokens = condition_tokens[1:-1]
        br1_tokens = get_tokens(code_string, body_children[2])
        if str(body_children[2].type) == 'parenthesized_expression':
            br1_tokens = br1_tokens[1:-1]
        br2_tokens = get_tokens(code_string, body_children[4])
        if str(body_children[4].type) == 'parenthesized_expression':
            br2_tokens = br2_tokens[1:-1]
        assignee_token = get_tokens(code_string, assignee)[0]
        if ret:  # in return statement, assignee is the keyword "return"
            tokens.extend(["if", "("] + condition_tokens + [")", "{", assignee_token] + br1_tokens +
                          [";", "}", "else", "{", assignee_token] + br2_tokens + [";", "}"])
        else:
            tokens.extend(["if", "("] + condition_tokens + [")", "{", assignee_token, "="] + br1_tokens +
                          [";", "}", "else", "{", assignee_token, "="] + br2_tokens + [";", "}"])
        return tokens

    @classmethod
    def assignment_ternary_removal(cls, code_string, assi_tern_expr, root, parser):
        if isinstance(code_string, str):
            code_string = code_string.encode()
        assert isinstance(root, Node)
        tokens = []
        children = root.children
        if len(children) == 0:
            tokens.append(cls.handle_terminal_node(root, code_string))
        for child in children:
            if child in assi_tern_expr:
                te_children = child.children
                assignee = te_children[0]
                # children[1] should be "="
                body = te_children[2]
                tokens = cls.ternary_body_write(body, code_string, assignee, tokens)
                break
            else:
                tokens += JavaAndCPPProcessor.assignment_ternary_removal(code_string, assi_tern_expr, child, parser)
        return tokens

    @classmethod
    def var_decl_ternary_removal(cls, code_string, var_decl_tern_expr, root, parser):
        if isinstance(code_string, str):
            code_string = code_string.encode()
        assert isinstance(root, Node)
        tokens = []
        children = root.children
        if len(children) == 0:
            tokens.append(cls.handle_terminal_node(root, code_string))
        for child in children:
            if child in var_decl_tern_expr:
                # tokens.extend(["return"])
                for c in child.children:
                    if str(c.type) == ";":
                        continue
                    elif str(c.type) == "variable_declarator" or str(c.type) == "init_declarator":  # the former is
                        # for java and the latter is for c:
                        te_children = c.children
                        assignee = te_children[0]
                        assignee_token = get_tokens(code_string, assignee)[0]
                        tokens.extend([assignee_token, ";"])
                        # children[1] should be "="
                        body = te_children[2]
                        tokens = cls.ternary_body_write(body, code_string, assignee, tokens)
                    else:
                        tokens += get_tokens(code_string, c)
            else:
                tokens += JavaAndCPPProcessor.var_decl_ternary_removal(code_string, var_decl_tern_expr, child, parser)
        return tokens

    @classmethod
    def return_ternary_removal(cls, code_string, ret_tern_expr, root, parser):
        if isinstance(code_string, str):
            code_string = code_string.encode()
        assert isinstance(root, Node)
        tokens = []
        children = root.children
        if len(children) == 0:
            tokens.append(cls.handle_terminal_node(root, code_string))
        for child in children:
            if child in ret_tern_expr:
                te_children = child.children
                assignee = te_children[0]
                # children[1] should be "="
                body = te_children[1]
                tokens = cls.ternary_body_write(body, code_string, assignee, tokens, ret=True)
                break
            else:
                tokens += JavaAndCPPProcessor.return_ternary_removal(code_string, ret_tern_expr, child, parser)
        return tokens

    @classmethod
    def extract_ternary_expression(cls, root):
        assi_ten_expr = []
        varde_ten_expr = []
        ret_ten_expr = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'ternary_expression' and str(
                    current_node.parent.type) == "assignment_expression":
                assi_ten_expr.append(current_node.parent)
            if str(current_node.type) == 'ternary_expression' and str(
                    current_node.parent.type) == "variable_declarator":
                varde_ten_expr.append(current_node.parent.parent)
            if str(current_node.type) == 'ternary_expression' and str(current_node.parent.type) == "return_statement":
                ret_ten_expr.append(current_node.parent)
            for child in current_node.children:
                queue.append(child)
        return assi_ten_expr, varde_ten_expr, ret_ten_expr

    # -----Post increment/decrement removal------
    @classmethod
    def incre_decre_removal(cls, code_string, parser):
        # code_string = cls.remove_package_and_import(code_string)  # TODO: Check whether this will mess up the code
        root = parser.parse_code(code_string)
        pre_expr, post_expr = cls.extract_incre_decre_expression(root, code_string)
        success = False
        if len(pre_expr) > 0:
            try:
                modified_tokens = cls.pre_incre_decre_removal(code_string, pre_expr, root, parser)
                code_string = cls.beautify_java_code(modified_tokens)
                root = parser.parse_code(code_string)
                success = True
                _, post_expr = cls.extract_incre_decre_expression(root, code_string)
            except:
                pass
                # print("pre incre/decre expression removal failed.")


        if len(post_expr) > 0:
            try:
                modified_tokens = cls.post_incre_decre_removal(code_string, post_expr, root, parser)
                code_string = cls.beautify_java_code(modified_tokens)
                root = parser.parse_code(code_string)
                success = True
            except:
                pass
                # print("post incre/decre expression removal failed.")

        return root, code_string, success

    @classmethod
    def pre_incre_decre_removal(cls, code_string, pre_expr, root, parser):
        if isinstance(code_string, str):
            code_string = code_string.encode()
        assert isinstance(root, Node)
        tokens = []
        children = root.children
        if len(children) == 0:
            tokens.append(cls.handle_terminal_node(root, code_string))
        for child in children:
            if child in pre_expr:
                expr = child.children[0]
                assignee = expr.children[0]
                assignee_token = get_tokens(code_string, assignee)[0]
                # check it is increment or decrement
                op = ""
                if str(expr.children[2].children[0].type) == "--":
                    op = "-="
                elif str(expr.children[2].children[0].type) == "++":
                    op = "+="
                assigner = expr.children[2].children[-1]
                assigner_token = get_tokens(code_string, assigner)[0]
                tokens.extend([assigner_token, op, "1", ";", assignee_token, "=", assigner_token, ";"])
                # break
            else:
                tokens += JavaAndCPPProcessor.pre_incre_decre_removal(code_string, pre_expr, child, parser)
        return tokens

    @classmethod
    def post_incre_decre_removal(cls, code_string, post_expr, root, parser):
        if isinstance(code_string, str):
            code_string = code_string.encode()
        assert isinstance(root, Node)
        tokens = []
        children = root.children
        if len(children) == 0:
            tokens.append(cls.handle_terminal_node(root, code_string))
        for child in children:
            if child in post_expr:
                expr = child.children[0]
                assignee = expr.children[0]
                assignee_token = get_tokens(code_string, assignee)[0]
                op = ""
                if str(expr.children[2].children[-1].type) == "--":
                    op = "-="
                elif str(expr.children[2].children[-1].type) == "++":
                    op = "+="
                assigner = expr.children[2].children[0]
                assigner_token = get_tokens(code_string, assigner)[0]
                tokens.extend([assignee_token, "=", assigner_token, ";", assigner_token, op, "1", ";"])
                # break
            else:
                tokens += JavaAndCPPProcessor.post_incre_decre_removal(code_string, post_expr, child, parser)
        return tokens

    @classmethod
    def extract_incre_decre_expression(cls, root, code_string):
        pre_expr = []
        post_expr = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if (str(current_node.type) == '++' or str(current_node.type) == "--") and \
                    (str(current_node.parent.type) == "update_expression" or str(
                        current_node.parent.type) == "postfix_unary_expression" or str(
                        current_node.parent.type) == "prefix_unary_expression") and \
                    str(current_node.parent.parent.type) == "assignment_expression":
                nodes = current_node.parent.parent.children
                if len(nodes) == 3 and str(nodes[
                                               0].type) == "identifier":  # this line is to double check whether the
                    # unary operation happens inside an assinemnt expression
                    if str(nodes[2].children[0].type) == "++" or str(nodes[2].children[0].type) == "--":
                        pre_expr.append(current_node.parent.parent.parent)
                    else:
                        post_expr.append(current_node.parent.parent.parent)
            for child in current_node.children:
                queue.append(child)
        return pre_expr, post_expr

    @classmethod
    def handle_terminal_node(cls, root_node, code_string):
        if root_node.type == "comment":
            str_const = ""
        else:
            str_const = code_string[root_node.start_byte:root_node.end_byte].decode("utf-8")
        return str_const

    @classmethod
    def remove_package_and_import(cls, code):
        if isinstance(code, str):
            code = code.encode()
        code = code.decode().split("\n")
        lines = [line.rstrip("\n") for line in code]
        current_code_lines = []
        for line in lines:
            if line.strip().startswith("import") or line.strip().startswith("package") or line.strip().startswith(
                    "#include"):
                # TODO: How to deal with the #if_def kind of code?
                continue
            current_code_lines.append(line)
        code = "\n".join(current_code_lines) if len(current_code_lines) else ""
        return code.encode()

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
    def block_swap_java(cls, code_str, parser):
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
    def block_swap_c(cls, code_str, parser):
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
                    if str(current_node.type) == 'compound_statement':
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
