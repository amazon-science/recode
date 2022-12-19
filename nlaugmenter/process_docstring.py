'''
This file includes code to apply transformations on code generation tasks.
Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
'''

from natgen import VarRenamerBase, sep


def sep_doc(data, prompt, entry_point):
    """ A help function to seperate docstring
    Note that there is another sep function in natgen which is used to seperate 
    the whole documents in prompt including docstring & examples.
    """
    # sep into header, docstring only (without any special charaters like """), examples
    if data in ["humaneval", "mbpp"]:
        start = prompt.find('"""', prompt.find(entry_point)) + 3
        # some transformation might remove \n, so we need to keep \n \t in head part
        special = start + 1
        while prompt[special] in [" ", "\n", "\t"]:
            special += 1
        start = special
        end = prompt.find(">>>", prompt.find(entry_point))
        # some transformation might remove \n, so we need to keep \n \t in cases part
        special = end - 1
        while prompt[special] in [" ", "\n", "\t"]:
            special -= 1
        end = special + 1
        return prompt[:start], prompt[start:end], prompt[end:]
    elif data in ["mbjp", "mbjsp"]:
        start = prompt.find('/**', prompt.find(entry_point)) + 3
        special = start + 1
        while prompt[special] in [" ", "\n", "\t", "*"]:
            special += 1
        start = special
        end = prompt.find("* >", prompt.find(entry_point))
        special = end - 1
        while prompt[special] in [" ", "\n", "\t"]:
            special -= 1
        end = special + 1
        return prompt[:start], prompt[start:end], prompt[end:]
    elif data in ["mbphp", "mbkp", "mbrbp"]:
        if data == "mbphp":
            start_flag = "You are an expert PHP programmer, and here is your task.\n *"
            end_flag = "* php >"
        elif data == "mbkp":
            start_flag = "You are an expert Kotlin programmer, and here is your task.\n *"
            end_flag = "* >>>"
        elif data == "mbrbp":
            start_flag = "You are an expert Ruby programmer, and here is your task.\n#"
            end_flag = "# irb>"
        start = prompt.find(start_flag) + len(start_flag)
        special = start + 1
        while prompt[special] in [" ", "\n", "\t", "*"]:
            special += 1
        start = special
        end = prompt.find(end_flag)
        special = end - 1
        while prompt[special] in [" ", "\n", "\t"]:
            special -= 1
        end = special + 1
        return prompt[:start], prompt[start:end], prompt[end:]
    else:
        print(f"Dataset {data} not supported for transformation!")
        exit()

def word_blacklist(entry):
    """ A help function to get a blacklist of words in the docstring that cannot be perturbed by nlaugmenter
    """
    header, doc, body = sep(entry["prompt"] + entry["canonical_solution"], entry['entry_point'])
    code_string = header + body
    tsf = VarRenamerBase("natgen/languages.so", "python")
    
    root = tsf.parse_code(code_string)
    var_names = tsf.extract_var_names(root, code_string)
    var_names = set(var_names)
    # for variables in type variables, add
    var_names.update(tsf.TYPE_VARS)
    # for variable name appears in function/class name, add
    not_var_ptype_var_names = tsf.get_not_var_ptype_var_names(root, code_string)
    var_names.update(not_var_ptype_var_names)
    return var_names

    # docstring_words = docstring.replace("\n", "").split(" ")
    # black_list = []
    # for word in docstring_words:
    #     if word in var_names:
    #         black_list.append(word)

    # return black_list


def clean_word(word):
    """ A help function to clean up words
    """
    # if word is "\ncost[]"", => "cost"
    new_word = ""
    start = False
    end = False
    for ch in word:
        if ch.isalnum():
            if not start: start = True
            new_word += ch
        else:
            if start: break
    return new_word


def preprocess_docstring(black_list, docstring, tsf):
    """ Preprocess docstring to replace variables in the blacklist to <|||> such that we will not perturb them
    """
    if hasattr(tsf, "perturb_level") and tsf.perturb_level in ["word-level", "char-level"]:
        # import pdb; pdb.set_trace()
        docstring_words = docstring.split(" ")
        new_words = []
        replaces = []
        for word in docstring_words:
            if clean_word(word) in black_list:
                new_words.append("<|||>")
                replaces.append(word)
            else:
                new_words.append(word)
        new_doc = ' '.join(new_words)
        # import pdb; pdb.set_trace()
        return new_doc, replaces
    else:
        return docstring, []


def postprocess_docstring(docstring, replaces):
    """ Postprocess docstring to replace <|||> to original words
    """
    docstring = docstring.replace("< ||| >", "<|||>") # sometimes will be separated due to tokenization
    for replace in replaces:
        docstring = docstring.replace("<|||>", replace, 1)
    return docstring