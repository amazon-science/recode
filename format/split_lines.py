import random

def split_lines(code, entry_point, language, method="longest"):
    """ The main function to config which line split methods needed to be used
    Default to be longest line split
    """
    if method == "longest":
        new_code = longest_splits(code, entry_point, language)
    elif method == "random":
        new_code = random_splits(code, entry_point, language)
    return new_code


def get_line_length(line):
    """ A help function to get the line length without considering space or indent
    """
    return len(line.replace(" ", "").replace("\t", "").replace("\n", ""))


def longest_splits(code, entry_point, language):
    """ Find the longest line in code and split that line in half
    Note that we do not split for comments and entry point line
    (we might need to consider filter out the following string cases)
    code = "s = \"abscssdfsdfsdf. sdfsdfsfewersdf sdfsd sdfsd\"\n import pdb; pdb.set_trace()"
    """
    partial_code = code
    added_splits = 0
    new_partial_code = ""
    lines = partial_code.split("\n")

    new_lines = []
    # find the longest line in the code to split
    longest_idx = 0
    longest_length = -1
    comment = False
    for line_idx, line in enumerate(lines):
        if line.replace(" ", "").replace("\t", "").replace("\n", "") == "": # empty line
            continue
        # skip comment lines so as not to perturb these lines
        if language == "python":
            if "\'\'\'" in line or "\"\"\"" in line:
                comment = not comment
                continue
            if "#" in line or comment:
                continue
        else:
            raise NotImplementedError

        if entry_point in line:
            # not perturbing function definition line for now
            continue

        indent = "" # should not split at the place of indent
        for ch_idx, ch in enumerate(line):
            if ch in [" ", "\t"]:
                indent += ch
            else:
                break
        line = line[ch_idx:]

        post = "" # should not split at the end of \n
        for ch_idx in range(1, len(line)):
            ch = line[len(line) - ch_idx]
            if ch in [" ", "\n"]:
                post = ch + post
            else:
                break
        line = line[:len(line) - ch_idx + 1]

        line_length = get_line_length(line)
        words = line.split(" ")
        # print(line_length, line)
        if len(words) >= 2 and line_length > longest_length:
            longest_idx = line_idx
            longest_length = line_length
    
    comment = False
    for line_idx, line in enumerate(lines):
        new_line = str(line)
        if line.replace(" ", "").replace("\t", "").replace("\n", "") == "": # empty line
            new_lines.append(new_line)
            continue
        if language == "python":
            if "\'\'\'" in line or "\"\"\"" in line:
                comment = not comment
                new_lines.append(new_line)
                continue
            if "#" in line or comment:
                new_lines.append(new_line)
                continue
        else:
            not NotImplementedError
        if entry_point in line:
            new_lines.append(new_line)
            continue

        if line_idx == longest_idx:
            indent = "" # should not split at the place of indent
            for ch_idx, ch in enumerate(line):
                if ch in [" ", "\t"]:
                    indent += ch
                else:
                    break
            line = line[ch_idx:]

            post = "" # should not split at the end of \n
            for ch_idx in range(1, len(line)):
                ch = line[len(line) - ch_idx]
                if ch in [" ", "\n"]:
                    post = ch + post
                else:
                    break
            line = line[:len(line) - ch_idx + 1]
            new_line = str(line)

            if "\"" in line:
                # cannot split in string!
                string_list = []
                string_start = new_line.find("\"")
                while string_start != -1:
                    string_end = new_line.find("\"", string_start + 1)
                    assert string_end != -1
                    string_list.append(line[string_start: string_end+1])
                    new_line = new_line[:string_start] + "@!string!@" + new_line[string_end + 1:]
                    string_start = new_line.find("\"")

            if language == "python":
                words = new_line.split(" ")
                if len(words) >= 2:
                    # split_pos = random.randint(1, len(words) - 1) # random position split
                    split_pos = len(words) // 2 # middle position split
                    words.insert(split_pos, " \\\n ")
                    new_line = indent + " ".join(words) + post
                    added_splits += 1
            else:
                raise NotImplementedError

            string_start = new_line.find("@!string!@")
            while string_start != -1:
                assert string_list
                orig_string = string_list.pop(0)
                string_end = string_start + len("@!string!@")
                new_line = new_line[:string_start] + orig_string + new_line[string_end + 1:]
                string_start = new_line.find("@!string!@")

        new_lines.append(new_line)
    new_partial_code = "\n".join(new_lines)

    # print(f"original partial code has {len(lines)}, now we add {added_splits} new splits")
    # import pdb; pdb.set_trace()
    return new_partial_code


def random_splits(code, entry_point, language="python", ratio=0.5):
    """ Splitting lines in a random fashion.
    Default ratio 0.5 means split lines for 50% of total lines of partial code
    The position to split in each line is also randomly selected (excluding spaces or indents)
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
    # ratio 0.1 means add 10% of total lines of partial code
    header_end = code.find("\n", code.find(entry_point))
    doc_start = code.find(doc_type, code.find(entry_point))
    doc_end = code.find(doc_type, doc_start + 3) + 3

    partial_code = code[doc_end:]
    header_and_doc = code[:doc_end]

    added_splits = 0
    new_partial_code = ""
    lines = partial_code.split("\n")

    new_lines = []
    for line in lines:
        new_line = str(line)
        if line.replace(" ", "").replace("\t", "").replace("\n", "") == "": # empty line
            new_lines.append(new_line)
            continue
        if random.random() < ratio: # enable split at this line
            indent = "" # should not split at the place of indent
            for ch_idx, ch in enumerate(line):
                if ch in [" ", "\t"]:
                    indent += ch
                else:
                    break
            line = line[ch_idx:]

            post = "" # should not split at the end of \n
            for ch_idx in range(1, len(line)):
                ch = line[len(line) - ch_idx]
                if ch in [" ", "\n"]:
                    post = ch + post
                else:
                    break
            line = line[:len(line) - ch_idx + 1]

            words = line.split(" ")
            if len(words) >= 2:
                words.insert(random.randint(1, len(words) - 1), " \ \n ")
                new_line = indent + " ".join(words) + post
                added_splits += 1
        new_lines.append(new_line)
    new_partial_code = "\n".join(new_lines)

    # print(f"original partial code has {len(lines)}, now we add {added_splits} new splits with ratio {ratio}")
    # import pdb; pdb.set_trace()
    return header_and_doc + new_partial_code

