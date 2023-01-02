
def tab_to_indent(code, indent_type):
    """ Transform \t to spaces for indents
    """
    # assert indent_type == "\t"
    new_lines = []
    lines = code.split("\n")
    for line in lines:
        new_line = str(line)
        if line.replace(" ", "").replace("\t", "").replace("\n", "") == "": # empty line
            new_lines.append(new_line)
            continue
        # extract the indent in each line and we only replace for indent
        indent = ""
        for ch_idx, ch in enumerate(line):
            if ch in [" ", "\t"]:
                indent += ch
            else:
                break
        # assert indent_type in indent
        if indent_type not in indent:
            # import pdb; pdb.set_trace()
            new_lines.append(new_line)
            continue
        new_line = line[ch_idx:]
        indent = indent.replace(indent_type, "    ")
        new_lines.append(indent + new_line)
    new_code = "\n".join(new_lines)

    # code = code.replace(indent_type, "    ")
    return new_code

def indent_to_tab(code, indent_type):
    """ Transform spaces to \t for indents
    """
    # assert indent_type in ["        ", "    ", "  ", " "], f"type {indent_type} wrong"
    new_lines = []
    lines = code.split("\n")
    for line in lines:
        new_line = str(line)
        if line.replace(" ", "").replace("\t", "").replace("\n", "") == "": # empty line
            new_lines.append(new_line)
            continue
        # extract the indent in each line and we only replace for indent
        indent = ""
        for ch_idx, ch in enumerate(line):
            if ch in [" ", "\t"]:
                indent += ch
            else:
                break
        # assert indent_type in indent
        if indent_type not in indent:
            # import pdb; pdb.set_trace()
            new_lines.append(new_line)
            continue
        new_line = line[ch_idx:]
        indent = indent.replace(indent_type, "\t")
        new_lines.append(indent + new_line)
    new_code = "\n".join(new_lines)

    # we cannot simply do replace as following:
    # if indent_type is just " ", will cause problems!
    # code = code.replace(indent_type, "\t")
    return new_code


def detect_indent_type(code, entry_point):
    """ A help function to detect the type of indent used in the code snipet
    """
    header_end = code.find("\n", code.find(entry_point))
    indent_type = ""
    for ch in code[header_end + 1:]:
        # if ch == "\n": import pdb; pdb.set_trace()
        # assert ch != "\n"
        if ch == "\n": continue # empty line direct after func signature
        if ch in [" ", "\t"]:
            indent_type += ch
        else:
            break
    return indent_type


def tab_indent(code, entry_point, language="python"):
    """ The main function to decide which type of tab_indent transformations used
    """
    assert language == "python"
    indent_type = detect_indent_type(code, entry_point)
    # if indent_type not in ["        ", "    ", "  "]:
    #     import pdb; pdb.set_trace()
    if "\t" in indent_type:
        new_code = tab_to_indent(code, indent_type)
    elif indent_type == "":
        # there is no indent...
        new_code = code
    else:
        new_code = indent_to_tab(code, indent_type)
    # import pdb; pdb.set_trace()
    return new_code

