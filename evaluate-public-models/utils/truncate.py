import ast


def is_valid_python(code):
    try:
        try:
            pared_code = ast.parse(code)
        except SyntaxError:
            return False
    except Exception as e:
        print("Exception: ", e)
        return False
    return pared_code


def get_function_from_ast(parsed_code, code, option="func_ast_last"):
    # This grabs the entire generation up til the last function that is valid
    # this is still a greedy approach at function level
    # another approach is to grab up to the first function
    # note: there can be import statements, variables, before first function

    # print(f"Starting from the following code\n{code}\n\n\n")
    # 1. get the last function
    assert option in [
        "func_ast_first",
        "func_ast_last",
    ], f"Invalid post process option {option}"
    for i in range(len(parsed_code.body)):
        idx = -i - 1 if option == "func_ast_last" else i
        if type(parsed_code.body[idx]) == ast.FunctionDef:
            break
        idx = None
    assert idx is not None, "No function found"
    function_segment = ast.get_source_segment(code, parsed_code.body[idx])
    # print(f"Found function segment at idx = {-i-1}\n{function_segment}\n\n\n")
    position = code.find(function_segment)
    # function_segment = code[position: position+len(function_segment)]
    function_segment_plus_previous = code[: position + len(function_segment)]
    return function_segment_plus_previous



def get_token_position_by_string(target_str, outputs, tokenizer, skip_special_tokens):
    for position in range(1, len(outputs) + 1):
        gen_str = tokenizer.decode(
            outputs[:position],
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
        if gen_str.rstrip() == target_str.rstrip():
            return position  # not including outputs[position]
        if gen_str.startswith(target_str) and target_str != "":
            print("Cannot find an exact match, use approx!")
            print(f"output length: {len(outputs)}")
            print(target_str)
            print("-----------------------")
            print(gen_str)
            return position
    if target_str.rstrip() == "":
        if target_str == "":
            print("generated empty string!")
        else:
            print("generated only white space!")
        return 0
    print(f"output length: {len(outputs)}")
    print(target_str)
    print("-----------------------")
    print(gen_str)
    raise RuntimeError("Cannot match prefix returned by AST.")



def filter_valid_code(
    true_str_input,
    execution_prompt,
    inputs,
    sequences,
    initial_context_length,
    tokenizer,
    task_id=None,
    has_special_tokens=False,
    post_process="greedy",
    replace_unk=False,
    skip_special_tokens=True,
    mean_logp=None,
    use_language_tag=0,
):
    """
    Due to tokenizer non lossless-ness, the decoded original prompt and
    the real original prompt are not the same.

    Due to constrained generation, input tokens not not necessarily match
    with the new input tokens (but match by characters instead)
    """
    samples = []
    # need both to handle CG / non losslessness of tokenizer
    decoded_context_string = tokenizer.batch_decode(
        inputs[:, use_language_tag:initial_context_length],
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )[0]
    decoded_original_prompt = tokenizer.batch_decode(
        inputs[:, use_language_tag:],
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )[0]
    processed_prompt = decoded_context_string

    assert execution_prompt is None, "only support execution_prompt is None here"
    processed_execution_prompt = processed_prompt

    output_lists = sequences[:, initial_context_length:]
    for sample_id, outputs in enumerate(output_lists):
        is_valid = False
        for position in range(len(outputs), 0, -1):
            gen_up_to_pos_toks = outputs[:position]
            gen_up_to_pos_str = tokenizer.decode(
                gen_up_to_pos_toks,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
            origin_pred = gen_up_to_pos_str
            code = (
                processed_execution_prompt + gen_up_to_pos_str
            )  # something is off for python
            parsed_code = is_valid_python(code)
            if parsed_code:
                is_valid = True
                # print(f"valid at position {position} / {len(outputs) - 1}. ")
                if post_process != "greedy":
                    try:
                        function_segment_plus_previous = get_function_from_ast(
                            parsed_code,
                            code,
                            option=post_process,
                        )
                        generated_part = function_segment_plus_previous[
                            len(processed_execution_prompt) :
                        ]
                    except Exception as e:
                        print("Something went wrong...", e)
                        generated_part = gen_up_to_pos_str
                elif post_process == "greedy":
                    generated_part = gen_up_to_pos_str
                else:
                    assert False, f"post processing method {post_process} not supported"

                if task_id is None:
                    return generated_part
                if mean_logp is None:
                    score = None
                else:
                    if post_process != "greedy":
                        position = get_token_position_by_string(
                            generated_part,
                            outputs,
                            tokenizer,
                            skip_special_tokens,
                        )
                    if position == 0:
                        score = -1e8
                    else:
                        score = mean_logp[sample_id][position - 1]

                samples.append(
                    dict(
                        task_id=task_id,
                        completion=(processed_prompt + generated_part)[
                            len(decoded_original_prompt) :
                        ],
                        ori_pred=(processed_prompt + origin_pred)[
                            len(decoded_original_prompt) :
                        ],
                        input=true_str_input,
                        mean_logp=score,
                    )
                )
                break
        if not is_valid:
            predictions = tokenizer.decode(
                outputs,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
            origin_pred = predictions
            print("Warning - no valid substring")
            if task_id is None:
                return predictions
            samples.append(
                dict(
                    task_id=task_id,
                    completion=(processed_prompt + predictions)[
                        len(decoded_original_prompt) :
                    ],
                    ori_pred=(processed_prompt + origin_pred)[
                        len(decoded_original_prompt) :
                    ],
                    input=true_str_input,
                    mean_logp=-1e8,
                )
            )

    return samples


def inference_cut_off(
    true_str_input,
    inputs,
    sequences,
    token_len_prompt_input,
    tokenizer,
    skip_special_tokens,
    task_id,
    language,
    input_indent=0,
    mean_logp=None,
    java_class_completion=True,
):
    # truncation for other langauges
    str_seqs = tokenizer.batch_decode(
        [seq for seq in sequences],
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )
    str_input = tokenizer.batch_decode(
        inputs,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )[0]
    results = []
    for sample_id, str_seq in enumerate(str_seqs):
        assert (
            str_seq.find(str_input) == 0
        ), f"raw output = \n{str_seq}\n\n raw input = \n{str_input}"
        str_output = str_seq[len(str_input) :]
        # Complete function braces for brace-based languages
        # For Java, close with another brace for the class
        balance = 0
        if language in [
            "java",
            "javascript",
            "typescript",
            "kotlin",
            "php",
            "rust",
            "cpp",
            "csharp",
            "go",
        ]:
            i = 0
            for i in range(len(str_output)):
                if str_output[i] == "{":
                    balance += 1
                elif str_output[i] == "}":
                    balance -= 1
                if balance == -1:
                    break
            generated_part = str_output[: i + 1]
            end_token_position = get_token_position_by_string(
                str_input + generated_part,
                sequences[sample_id],
                tokenizer,
                skip_special_tokens,
            )
            if mean_logp is not None:
                score = mean_logp[sample_id][
                    end_token_position - 1 - token_len_prompt_input
                ]
            else:
                score = -1e8
            if language == "java" and java_class_completion:
                generated_part += "\n}"
            results.append(
                dict(
                    task_id=task_id,
                    completion=generated_part,
                    ori_pred=str_output,
                    input=true_str_input,
                    mean_logp=score,
                )
            )
            # print("balance (number of open braces) =", balance)

        elif language == "python":
            # Not used in offline evaluation
            output_lines = str_output.split("\n")
            cutoff_output = output_lines[0]
            for line in output_lines[1:]:
                if not line.strip():
                    cutoff_output += "\n" + line
                elif len(line) - len(line.lstrip()) <= input_indent:
                    cutoff_output += "\n" + line[: len(line) - len(line.lstrip())]
                    break
                else:
                    cutoff_output += "\n" + line
            results.append(
                dict(
                    task_id=task_id,
                    completion=cutoff_output,
                    ori_pred=str_output,
                    input=true_str_input,
                    mean_logp=-1e8,
                )
            )
        elif language == "ruby":
            output_lines = str_output.split("\n")
            cutoff_output = output_lines[0]
            for line in output_lines[1:]:
                if not line.strip():
                    cutoff_output += "\n" + line
                elif len(line) - len(line.lstrip()) <= input_indent:
                    cutoff_output += "\n" + line[: len(line) - len(line.lstrip())]
                    break
                else:
                    cutoff_output += "\n" + line
            cutoff_output += "\nend\n"
            results.append(
                dict(
                    task_id=task_id,
                    completion=cutoff_output,
                    ori_pred=str_output,
                    input=true_str_input,
                    mean_logp=-1e8,
                )
            )
        else:
            assert False, f"Language {language} unsupported"
    return results
