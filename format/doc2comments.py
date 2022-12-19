
from matplotlib import docstring

def doc2comments(code, entry_point):
    """ change \"\"\" to # comments
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
    
    doc_start = code.find(doc_type, code.find(entry_point))
    # get all the indent before """
    doc_line_start = doc_start - 1
    indent = ""
    while True:
        ch = code[doc_line_start]
        if ch in [" ", "\t"]:
            doc_line_start -= 1
            indent = ch + indent
        else:
            break
    if code[doc_line_start] != "\n": import pdb; pdb.set_trace()
    assert code[doc_line_start] == "\n"
    doc_line_start += 1

    doc_end = code.find(doc_type, doc_start + 3)
    # in case there are spaces after doc_end of """
    doc_line_end = doc_end + 3
    for ch in code[doc_end:]:
        if ch in [" ", "\t"]:
            doc_line_end += 1
        else:
            break
    if code[doc_line_end] != "\n": import pdb; pdb.set_trace()
    assert code[doc_line_end] == "\n"

    doc_lines = code[doc_line_start: doc_line_end + 1]

    lines = doc_lines.split("\n")

    new_lines = []
    for line in lines:
        new_line = str(line)
        if line.replace(" ", "").replace("\t", "").replace("\n", "") == doc_type:
            # remove these lines
            continue
        if "\"\"\"" in line:
            new_line = new_line.replace(f"{doc_type} ", "").replace(doc_type, "")
        space_ahead = ""
        # import pdb; pdb.set_trace()
        for ch_idx, ch in enumerate(line):
            if ch in [" ", "\t"]:
                space_ahead += ch
            else:
                break
        
        # new_line = space_ahead.replace(indent, indent + "# ") + new_line[ch_idx:]
        if indent in space_ahead:
            new_line = space_ahead[: len(indent)] + "# " + space_ahead[len(indent):]  + new_line[ch_idx:]
        new_lines.append(new_line)

    new_code = code[:doc_line_start] + "\n".join(new_lines) + code[doc_line_end + 1:]

    # import pdb; pdb.set_trace()
    return new_code

