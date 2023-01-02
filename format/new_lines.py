import random

def new_lines(code, entry_point, language="python", ratio=0.5):
    """ Adding new lines in a random fashion.
    Default ratio 0.5 means add a \n on 50% of total lines of partial code
    This is a specific function to MBXP and humaneval that adding random newlines only after docstring
    """
    single_doc = code.find("\'\'\'")
    double_doc = code.find("\"\"\"")
    if single_doc == -1: doc_type = "\"\"\""
    elif double_doc == -1: doc_type = "\'\'\'"
    elif single_doc != -1 and double_doc != -1:
        doc_type = "\"\"\""
    else:
        print("doc_type not supported!")
        exit()
    header_end = code.find("\n", code.find(entry_point))
    doc_start = code.find(doc_type, code.find(entry_point))
    doc_end = code.find(doc_type, doc_start + 3) + 3

    partial_code = code[doc_end:]
    header_and_doc = code[:doc_end]

    num_lines = 0
    added_lines = 0
    new_partial_code = ""
    for ch in partial_code:
        new_partial_code += ch
        if ch == "\n": 
            num_lines += 1
            if random.random() < ratio:
                new_partial_code += "\n"
                added_lines += 1

    # print(f"original partial code has {num_lines}, now we add {added_lines} new lines with ratio {ratio}")
    # import pdb; pdb.set_trace()
    return header_and_doc + new_partial_code

def new_lines_general(code, entry_point, language="python", ratio=0.5):
    """ Adding new lines in a random fashion.
    Default ratio 0.5 means add a \n on 50% of total lines of partial code
    This is a general function that adding random newlines at any place
    """
    partial_code = code
    num_lines = 0
    added_lines = 0
    new_partial_code = ""
    for ch in partial_code:
        new_partial_code += ch
        if ch == "\n": 
            num_lines += 1
            if random.random() < ratio:
                new_partial_code += "\n"
                added_lines += 1

    # print(f"original partial code has {num_lines}, now we add {added_lines} new lines with ratio {ratio}")
    # import pdb; pdb.set_trace()
    return new_partial_code


def new_line_aftercode(code, entry_point, language="python"):
    """ Adding new lines directly after partial code
    Easy to trigger robusntess problems
    """
    return code + "\n"


def indent_new_line_aftercode(code, entry_point, language="python"):
    # adding a newline following the previous line indent
    lines =  code.split("\n")
    for line in lines[::-1]:
        if line.replace(" ", "").replace("\t", "").replace("\n", "") != "":
            indent = ""
            for ch in line:
                if ch not in [" ", "\t"]: 
                    break
                indent += ch
            break
    return code + indent + "\n"


def new_line_afterdoc(code, entry_point, language="python"):
    """ Adding new lines after docstring.
    """
    single_doc = code.find("\'\'\'")
    double_doc = code.find("\"\"\"")
    if single_doc == -1: doc_type = "\"\"\""
    elif double_doc == -1: doc_type = "\'\'\'"
    elif single_doc != -1 and double_doc != -1:
        doc_type = "\"\"\""
    else:
        print("doc_type not supported!")
        exit()
    header_end = code.find("\n", code.find(entry_point))
    doc_start = code.find(doc_type, code.find(entry_point))
    doc_end = code.find(doc_type, doc_start + 3) + 3
    partial_code = code[doc_end:]
    header_and_doc = code[:doc_end]
    return header_and_doc + "\n" + partial_code