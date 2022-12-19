'''
This file includes to code apply transformations on code generation tasks.
Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
'''

from transformations.butter_fingers_perturbation.transformation import ButterFingersPerturbation
from transformations.swap_characters.transformation import SwapCharactersPerturbation
from transformations.change_char_case.transformation import ChangeCharCase
from transformations.english_inflectional_variation.transformation import EnglishInflectionalVariation
from transformations.synonym_substitution.transformation import SynonymSubstitution

def post_process_name(s):
    # clean up the perturbed function name
    # return None if no simple way to make it a valid name
    if len(s) == 0:
        return ""
    res = ""
    start = False
    for ch in s:
        if ch.isalpha:
            # make sure the first character isalpha()
            start = True
        if start:
            if ch.isalnum() or ch == '_':
                res += ch
    return res


def replace_func_name(code, entry_point, new_func_name):
    new_func_name = post_process_name(new_func_name)
    if new_func_name == "":
        return code, entry_point

    return code.replace(entry_point, new_func_name), new_func_name


def FuncRenameSynonymSub(code, entry_point):
    t = SynonymSubstitution()
    if "_" in entry_point:
        # has_close_elements => receive_close_element
        new_func_name = " ".join(entry_point.split("_"))
        new_func_name = t.generate(new_func_name)[0]
        new_func_name = "_".join(new_func_name.split(" "))
    else:
        # hasCloseElements => receiveCloseElements
        _, new_func_name = FuncRenameCamelCase(code, entry_point)
        new_func_name = " ".join(new_func_name.split("_"))
        new_func_name = t.generate(new_func_name)[0]
        new_func_name = "_".join(new_func_name.split(" "))
        _, new_func_name = FuncRenameCamelCase(code, new_func_name)
    
    new_code, new_func_name = replace_func_name(code, entry_point, new_func_name)
    # print(f"function name from \"{entry_point}\" to \"{new_func_name}\"")
    # import pdb; pdb.set_trace()
    return new_code, new_func_name


def FuncRenameInflectionalVariation(code, entry_point):
    t = EnglishInflectionalVariation()
    new_func_name = t.generate(entry_point)[0]
    new_code, new_func_name = replace_func_name(code, entry_point, new_func_name)
    # print(f"function name from \"{entry_point}\" to \"{new_func_name}\"")
    # import pdb; pdb.set_trace()
    return new_code, new_func_name


def FuncRenameChangeChar(code, entry_point):
    t = ChangeCharCase(prob=0.35)
    # default 0.1 would be too small since func name is short
    new_func_name = t.generate(entry_point)[0]
    new_code, new_func_name = replace_func_name(code, entry_point, new_func_name)
    # print(f"function name from \"{entry_point}\" to \"{new_func_name}\"")
    # import pdb; pdb.set_trace()
    return new_code, new_func_name


def FuncRenameSwapChar(code, entry_point):
    t = SwapCharactersPerturbation()
    new_func_name = t.generate(entry_point)[0]
    new_code, new_func_name = replace_func_name(code, entry_point, new_func_name)
    # print(f"function name from \"{entry_point}\" to \"{new_func_name}\"")
    # import pdb; pdb.set_trace()
    return new_code, new_func_name


def FuncRenameButterFinger(code, entry_point):
    t = ButterFingersPerturbation()
    new_func_name = t.generate(entry_point)[0]
    new_code, new_func_name = replace_func_name(code, entry_point, new_func_name)
    # print(f"function name from \"{entry_point}\" to \"{new_func_name}\"")
    # import pdb; pdb.set_trace()
    return new_code, new_func_name


def FuncRenameCamelCase(code, entry_point):
    if "_" in entry_point:
        # has_close_elements => hasCloseElements
        new_func_name = str(entry_point)
        words = new_func_name.split("_")
        new_words = [words[0]]
        for word in words[1:]:
            chars = list(word)
            chars[0] = chars[0].upper()
            new_words.append("".join(chars))
        new_func_name = "".join(new_words)
    else:
        # hasCloseElements => has_close_elements
        new_func_name = entry_point[0]
        for ch in entry_point[1:]:
            if ch.isupper():
                new_func_name += "_" + ch.lower()
            else:
                new_func_name += ch

    new_code, new_func_name = replace_func_name(code, entry_point, new_func_name)
    # print(f"function name from \"{entry_point}\" to \"{new_func_name}\"")
    # import pdb; pdb.set_trace()
    return new_code, new_func_name