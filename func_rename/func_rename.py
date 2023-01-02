from nlaugmenter.transformations.butter_fingers_perturbation.transformation import ButterFingersPerturbation
from nlaugmenter.transformations.swap_characters.transformation import SwapCharactersPerturbation
from nlaugmenter.transformations.change_char_case.transformation import ChangeCharCase
from nlaugmenter.transformations.english_inflectional_variation.transformation import EnglishInflectionalVariation
from nlaugmenter.transformations.synonym_substitution.transformation import SynonymSubstitution

def post_process_name(s):
    """ clean up the perturbed function name
    return None if no simple way to make it a valid name
    """
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
    """ A general function to replace function name with the new name
    """
    new_func_name = post_process_name(new_func_name)
    if new_func_name == "":
        return code, entry_point

    return code.replace(entry_point, new_func_name), new_func_name


def FuncRenameSynonymSub(code, entry_point):
    """ Using synonum substitution to perturb function name
    >>> example1: has_close_elements => receive_close_element
    >>> example2: hasCloseElements => receiveCloseElements
    """
    t = SynonymSubstitution()
    new_code = str(code)
    if not isinstance(entry_point, list):
        entry_points = [entry_point]
    else:
        entry_points = entry_point
    for entry_point in entry_points:
        if "_" in entry_point:
            # has_close_elements => receive_close_element
            new_func_name = " ".join(entry_point.split("_"))
            new_func_name = t.generate(new_func_name)[0]
            new_func_name = "_".join(new_func_name.split(" "))
        else:
            # hasCloseElements => receiveCloseElements
            _, new_func_name = FuncRenameCamelCase(new_code, entry_point)
            new_func_name = " ".join(new_func_name.split("_"))
            new_func_name = t.generate(new_func_name)[0]
            new_func_name = "_".join(new_func_name.split(" "))
            _, new_func_name = FuncRenameCamelCase(new_code, new_func_name)
        
        new_code, new_func_name = replace_func_name(new_code, entry_point, new_func_name)
        # print(f"function name from \"{entry_point}\" to \"{new_func_name}\"")
        # import pdb; pdb.set_trace()
    return new_code, new_func_name


def FuncRenameInflectionalVariation(code, entry_point):
    """ Using english inflectional variation to perturb function name
    """
    t = EnglishInflectionalVariation()
    new_code = str(code)
    if not isinstance(entry_point, list):
        entry_points = [entry_point]
    else:
        entry_points = entry_point
    for entry_point in entry_points:
        # new_func_name = t.generate(entry_point)[0]
        if "_" in entry_point:
            # has_close_elements => receive_close_element
            new_func_name = " ".join(entry_point.split("_"))
            new_func_name = t.generate(new_func_name)[0]
            new_func_name = "_".join(new_func_name.split(" "))
        else:
            # hasCloseElements => receiveCloseElements
            _, new_func_name = FuncRenameCamelCase(new_code, entry_point)
            new_func_name = " ".join(new_func_name.split("_"))
            new_func_name = t.generate(new_func_name)[0]
            new_func_name = "_".join(new_func_name.split(" "))
            _, new_func_name = FuncRenameCamelCase(new_code, new_func_name)
        new_code, new_func_name = replace_func_name(new_code, entry_point, new_func_name)
        # print(f"function name from \"{entry_point}\" to \"{new_func_name}\"")
        # import pdb; pdb.set_trace()
    return new_code, new_func_name


def FuncRenameChangeChar(code, entry_point):
    """ Using random character case changes to perturb function name
    """
    t = ChangeCharCase(prob=0.35)
    # default 0.1 would be too small since func name is short
    new_code = str(code)
    if not isinstance(entry_point, list):
        entry_points = [entry_point]
    else:
        entry_points = entry_point
    for entry_point in entry_points:
        new_func_name = t.generate(entry_point)[0]
        new_code, new_func_name = replace_func_name(new_code, entry_point, new_func_name)
        # print(f"function name from \"{entry_point}\" to \"{new_func_name}\"")
        # import pdb; pdb.set_trace()
    return new_code, new_func_name


def FuncRenameSwapChar(code, entry_point):
    """ Using character swap to perturb function name
    """
    t = SwapCharactersPerturbation()
    new_code = str(code)
    if not isinstance(entry_point, list):
        entry_points = [entry_point]
    else:
        entry_points = entry_point
    for entry_point in entry_points:
        new_func_name = t.generate(entry_point)[0]
        # new_func_name = t.generate(entry_point, prob=0.1)[0]
        new_code, new_func_name = replace_func_name(new_code, entry_point, new_func_name)
        # print(f"function name from \"{entry_point}\" to \"{new_func_name}\"")
        # import pdb; pdb.set_trace()
    return new_code, new_func_name


def FuncRenameButterFinger(code, entry_point):
    """ Using butterfinger to perturb function name
    """
    t = ButterFingersPerturbation()
    new_code = str(code)
    if not isinstance(entry_point, list):
        entry_points = [entry_point]
    else:
        entry_points = entry_point
    for entry_point in entry_points:
        new_func_name = t.generate(entry_point)[0]
        new_code, new_func_name = replace_func_name(new_code, entry_point, new_func_name)
        # print(f"function name from \"{entry_point}\" to \"{new_func_name}\"")
        # import pdb; pdb.set_trace()
    return new_code, new_func_name


def FuncRenameCamelCase(code, entry_point):
    """ Perturb between two versions of function name
    We might have input function names as list which means we need to perturb all these names
    >>> example1: has_close_elements => hasCloseElements
    >>> example2: hasCloseElements => has_close_elements
    """
    new_code = str(code)
    if not isinstance(entry_point, list):
        entry_points = [entry_point]
    else:
        entry_points = entry_point
    for entry_point in entry_points:
        if entry_point == "": continue
        if "_" in entry_point:
            # has_close_elements => hasCloseElements
            new_func_name = str(entry_point)
            words = new_func_name.split("_")
            new_words = [words[0]]
            for word in words[1:]:
                chars = list(word)
                if len(chars) == 0: continue
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

        new_code, new_func_name = replace_func_name(new_code, entry_point, new_func_name)
        # print(f"function name from \"{entry_point}\" to \"{new_func_name}\"")
        # import pdb; pdb.set_trace()
    return new_code, new_func_name