import math
import random
import re
from typing import Union, Tuple
import os

from .language_processors import (
    JavaAndCPPProcessor,
    CSharpProcessor,
    PythonProcessor,
    JavascriptProcessor,
    PhpProcessor,
    GoProcessor,
    RubyProcessor
)
# from .language_processors.go_processor import GoProcessor
# from language_processors.ruby_processor import RubyProcessor
from .language_processors.utils import get_tokens
from .transformation_base import TransformationBase
import os
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline
import string

java_code = """
    import a.b
    // just for test
    void foo(){
        /*
        * test test
        */
        int time = 20;
        if (time < 18) {
            time=10;
        }
            else {
            System.out.println("Good evening.");
        }
    }
    """
python_code = """
    from typing import List

    # test
    @ decorator
    def factorize(n: int) -> List[int]:
        \"\"\" test
        test test
        \"\"\"
        import math
        fact = []
        i = 2
        while i <= int(math.sqrt(n) + 1):
            if n % i == 0:
                fact.append(i)
                n //= i
            else:
                i += 1
        if n > 1:
            fact.append(n)
        return fact
    """
c_code = """
    void foo(){
        int time = 20;
        if (time < 18) {
            time=10;
        }
            else {
            System.out.println("Good evening.");
        }
    }
    """
cs_code = """
    void foo(){
        int time = 20;
        if (time < 18) {
            time=10;
        }
            else {
            System.out.println("Good evening.");
        }
    }
    """
js_code = """function foo(n) {
        if (time < 10) {
            greeting = "Good morning";
        } 
        else {
            greeting = "Good evening";
        }
    }
    """
ruby_code = """
    x = 1
    if x > 2
        puts "x is greater than 2"   
    else
        puts "I can't guess the number"
    end
    """
go_code = """
    func main() {
        /* local variable definition */
        var a int = 100;

        /* check the boolean condition */
        if( a < 20 ) {
            /* if condition is true then print the following */
            fmt.Printf("a is less than 20\n" );
        } else {
            /* if condition is false then print the following */
            fmt.Printf("a is not less than 20\n" );
        }
        fmt.Printf("value of a is : %d\n", a);
    }
    """
php_code = """
    <?php 
    $t = date("H");
    if ($t < "10") {
        echo "Have a good morning!";
    }  else {
        echo "Have a good night!";
    }
    ?> 
    """
input_map = {
    "java": ("java", java_code),
    "c": ("c", c_code),
    "cpp": ("cpp", c_code),
    "cs": ("c_sharp", cs_code),
    "js": ("javascript", js_code),
    "python": ("python", python_code),
    "php": ("php", php_code),
    "ruby": ("ruby", ruby_code),
    "go": ("go", go_code),
}


test_code = {"python": python_code, "java": java_code, "javascript": js_code}

processor_function = {
    "java": JavaAndCPPProcessor,
    "c": JavaAndCPPProcessor,
    "cpp": JavaAndCPPProcessor,
    "c_sharp": CSharpProcessor,
    "python": PythonProcessor,
    "javascript": JavascriptProcessor,
    "go": GoProcessor,
    "php": PhpProcessor,
    "ruby": RubyProcessor,
}

tokenizer_function = {
    "java": get_tokens,
    "c": get_tokens,
    "cpp": get_tokens,
    "c_sharp": get_tokens,
    "python": PythonProcessor.get_tokens,
    "javascript": JavascriptProcessor.get_tokens,
    "go": get_tokens,
    "php": PhpProcessor.get_tokens,
    "ruby": get_tokens,
}


class VarRenamerBase(TransformationBase):
    """ Base class for renaming variables
    """
    def __init__(
            self,
            parser_path: str,
            language: str, 
    ):
        super(VarRenamerBase, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.name = "VarRenamerBase"
        self.processor = processor_function[self.language]
        self.tokenizer_function = tokenizer_function[self.language]
        # C/CPP: function_declarator
        # Java: class_declaration, method_declaration
        # python: function_definition, call
        # js: function_declaration
        self.TYPE_VARS = ["float", "list", "List", "int", "bool", "tuple", "str", "dict", "True", "False", "self", "return", "$", "in"]
        self.not_var_ptype = ["function_declarator", "class_declaration", "method_declaration", "function_definition",
                              "function_declaration", "call", "local_function_statement"]

    def extract_var_names(self, root, code_string):
        var_names = []
        queue = [root]

        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if (current_node.type == "identifier" or current_node.type == "variable_name") and str(
                    current_node.parent.type) not in self.not_var_ptype:
                var_names.append(self.tokenizer_function(code_string, current_node)[0])
            for child in current_node.children:
                queue.append(child)
        return var_names

    def get_not_var_ptype_var_names(self, root, code_string):
        var_names = []
        queue = [root]

        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if (current_node.type == "identifier" or current_node.type == "variable_name") and str(current_node.parent.type) in self.not_var_ptype:
                var_names.append(self.tokenizer_function(code_string, current_node)[0])
            for child in current_node.children:
                queue.append(child)
        return var_names

    def get_import_var_names(self, root, code_string):
        # import_code_string = str(code_string)
        lines = code_string.split("\n")
        new_lines = []
        for line in lines:
            new_line = str(line)
            if "import" not in line:
                continue
            new_lines.append(new_line)
        import_code_string = "\n".join(new_lines)
        import_var_names = self.extract_var_names(self.parse_code(import_code_string), import_code_string)
        return list(set(import_var_names))

    def select_most_frequent_var(self, original_code, var_names):
        counts = {}
        for var in var_names:
            counts[var] = original_code.count(var)
        max_cnt = -1
        max_var = None
        for var in var_names:
            if counts[var] > max_cnt:
                max_cnt = counts[var]
                max_var = var
        return max_var

    def check_valid_var(self, var_name):
        if var_name == "":
            return False
        if var_name in self.TYPE_VARS:
            return False
        return var_name[0].isalpha() and var_name.replace("_", "").isalnum()

    def var_renaming(self, code_string, method, debug=False):
        # code_string = '"""User permissions."""\n\n# Django REST Framework\nfrom rest_framework.permissions import BasePermission\n\n\nclass IsAccountOwner(BasePermission):\n    """Allow access only to objects owned by the requesting user."""\n\n    def has_permission(self, request, view):\n        """Let object permission grant access."""\n        obj = view.get_object()\n        return self.has_object_permission(request, view, obj)\n        \n    # gga\n    # gga\n    def has_object_permission(self, request, view, obj):\n        """Check obj and user are the same. """\n        # happy\n        for i in range(10):\n            print("happy")\n        return request.user == obj'
        max_position_embeddings = 256
        root = self.parse_code(code_string)
        original_code = self.tokenizer_function(code_string, root)
        var_names = self.extract_var_names(root, code_string)
        var_names = list(set(var_names))
        # for variables in import line, remove
        import_var_names = self.get_import_var_names(root, code_string)
        for ivn in import_var_names:
            if ivn in var_names: # possibly import in comments
                var_names.remove(ivn)
        # for variables in type variables, remove
        for type_var in self.TYPE_VARS:
            if type_var in var_names:
                var_names.remove(type_var)
        # for variable name appears in function/class name, remove
        not_var_ptype_var_names = self.get_not_var_ptype_var_names(root, code_string)
        for nvpvn in not_var_ptype_var_names:
            if nvpvn in var_names:
                var_names.remove(nvpvn)
        # selected_var is None if no suitable variable to replace
        selected_var = self.select_most_frequent_var(original_code, var_names)
        if debug: import pdb; pdb.set_trace()

        replace_var = None
        if selected_var is not None:
            # we have suitable var to replace if selected_var is not None
            if method == "CodeBERT":
                new_code_string = code_string.replace(selected_var, " <mask> ")
                model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
                tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
                fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
                input_start = 0
                input_end = min(input_start + max_position_embeddings, len(new_code_string))
                replace_var_opts = {}
                while input_end <= len(new_code_string):
                    # split the code into a couple blocks
                    outputs = []
                    assert len(new_code_string[input_start: input_end]) <= max_position_embeddings, f"length {len(new_code_string) > {max_position_embeddings}}"
                    # print(len(new_code_string[input_start: input_end]))
                    if "<mask>" in new_code_string[input_start: input_end]:
                        outputs = fill_mask(new_code_string[input_start: input_end])
                    # assert len(outputs) > 0, "CodeBERT no outputs!"
                    for pos in range(len(outputs)):
                        if type(outputs[pos]) == list:
                            assert len(outputs[pos]) > 0, f"empty prediction for pos {pos}"
                            for options in range(len(outputs[pos])):
                                pred, score = outputs[pos][options]["token_str"], outputs[pos][options]["score"]
                                pred = pred.replace(" ", "")
                                if pred not in replace_var_opts:
                                    replace_var_opts[pred] = score
                                else:
                                    replace_var_opts[pred] += score
                        else:
                            options = pos # if only one mask position, no pos dimension, set option to pos
                            pred, score = outputs[options]["token_str"], outputs[options]["score"]
                            pred = pred.replace(" ", "")
                            if pred not in replace_var_opts:
                                replace_var_opts[pred] = score
                            else:
                                replace_var_opts[pred] += score
                    input_start = input_end
                    if input_end >= len(new_code_string): break
                    input_end = min(input_start + max_position_embeddings, len(new_code_string))

                # selected the predicted options with the highest sum score
                max_score = -1
                for pred in replace_var_opts:
                    if not self.check_valid_var(pred): continue
                    if replace_var_opts[pred] > max_score:
                        max_score = replace_var_opts[pred]
                        replace_var = pred
                if debug: print(replace_var_opts)
            elif method == "naive":
                replace_var = "VAR_0"
            elif method == "alpha-numeric":
                # random variable names, half alphabetics half numbers
                if selected_var is not None:
                    if len(selected_var) == 1:
                        replace_var = random.choice(string.ascii_uppercase + string.ascii_lowercase)
                    else:
                        replace_var = random.choice(string.ascii_uppercase + string.ascii_lowercase) + \
                                ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + \
                                string.digits * 6) for _ in range(len(selected_var) - 1))

            if replace_var is not None:
                replace_var = replace_var.replace(" ", "")
                if replace_var in self.TYPE_VARS or replace_var in var_names:
                    replace_var = replace_var + "2"
            
        if debug: print(selected_var, "=>", replace_var)

        modified_code = []
        for t in original_code:
            if selected_var is not None and replace_var is not None and t == selected_var:
                # selected_var is None means no suitable var to replace
                # replace_var is None means no suitable replacement found for selected_var
                modified_code.append(replace_var)
            else:
                modified_code.append(t)

        modified_code_string = " ".join(modified_code)
        modified_root = self.parse_code(modified_code_string)
        return modified_root, modified_code_string, modified_code != original_code


class VarRenamerCB(VarRenamerBase):
    """ Using CodeBERT to replace targeted variables
    """
    def __init__(
            self,
            parser_path: str,
            language: str, 
    ):
        super(VarRenamerCB, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.name = "VarRenamerCB"
        self.processor = processor_function[self.language]
        self.tokenizer_function = tokenizer_function[self.language]


    def transform_code(
            self,
            code: Union[str, bytes],
            first_half=False
    ) -> Tuple[str, object]:
        ##### just for test #####
        # code = test_code[self.language]
        ##### just for test #####
        
        root, code, success = self.var_renaming(code, method="CodeBERT")
        # code = re.sub("[ \n\t]+", " ", code)
        code = code.replace("\n", " NEWLINE ")
        return code, {
            "success": success
        }


class VarRenamerNaive(VarRenamerBase):
    """ Using naive name VAR_0 to replace targeted variables
    """
    def __init__(
            self,
            parser_path: str,
            language: str, 
    ):
        super(VarRenamerNaive, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.name = "VarRenamerNaive"
        self.processor = processor_function[self.language]
        self.tokenizer_function = tokenizer_function[self.language]

    def transform_code(
            self,
            code: Union[str, bytes],
            first_half=False
    ) -> Tuple[str, object]:
        ##### just for test #####
        # code = test_code[self.language]
        ##### just for test #####

        root, code, success = self.var_renaming(code, method="naive")
        code = code.replace("\n", " NEWLINE ")
        # code = re.sub("[ \n\t]+", " ", code)
        return code, {
            "success": success
        }


class VarRenamerRN(VarRenamerBase):
    """ Using random alpha-numeric to repalce targeted variables
    """
    def __init__(
            self,
            parser_path: str,
            language: str, 
    ):
        super(VarRenamerRN, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.name = "VarRenamerRN"
        self.processor = processor_function[self.language]
        self.tokenizer_function = tokenizer_function[self.language]

    def transform_code(
            self,
            code: Union[str, bytes],
            first_half=False
    ) -> Tuple[str, object]:
        ##### just for test #####
        # code = test_code[self.language]
        ##### just for test #####

        root, code, success = self.var_renaming(code, method="alpha-numeric")
        # code = re.sub("[ \n\t]+", " ", code)
        code = code.replace("\n", " NEWLINE ")
        return code, {
            "success": success
        }
''' end of Amazon addition '''



class VarRenamer(TransformationBase):
    def __init__(
            self,
            parser_path: str,
            language: str
    ):
        super(VarRenamer, self).__init__(
            parser_path=parser_path,
            language=language,
        )
        self.language = language
        self.processor = processor_function[self.language]
        self.tokenizer_function = tokenizer_function[self.language]
        # C/CPP: function_declarator
        # Java: class_declaration, method_declaration
        # python: function_definition, call
        # js: function_declaration
        self.not_var_ptype = ["function_declarator", "class_declaration", "method_declaration", "function_definition",
                              "function_declaration", "call", "local_function_statement"]

    def extract_var_names(self, root, code_string):
        var_names = []
        queue = [root]

        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if (current_node.type == "identifier" or current_node.type == "variable_name") and str(
                    current_node.parent.type) not in self.not_var_ptype:
                var_names.append(self.tokenizer_function(code_string, current_node)[0])
            for child in current_node.children:
                queue.append(child)
        return var_names

    def var_renaming(self, code_string):
        root = self.parse_code(code_string)
        original_code = self.tokenizer_function(code_string, root)
        # print(" ".join(original_code))
        var_names = self.extract_var_names(root, code_string)
        var_names = list(set(var_names))
        num_to_rename = math.ceil(0.2 * len(var_names))
        random.shuffle(var_names)
        var_names = var_names[:num_to_rename]
        var_map = {}
        for idx, v in enumerate(var_names):
            var_map[v] = f"VAR_{idx}"
        modified_code = []
        for t in original_code:
            if t in var_names:
                modified_code.append(var_map[t])
            else:
                modified_code.append(t)

        modified_code_string = " ".join(modified_code)
        if modified_code != original_code:
            modified_root = self.parse_code(modified_code_string)
            return modified_root, modified_code_string, True
        else:
            return root, code_string, False

    def transform_code(
            self,
            code: Union[str, bytes],
            first_half=False
    ) -> Tuple[str, object]:
        root, code, success = self.var_renaming(code)
        code = re.sub("[ \n\t]+", " ", code)
        return code, {
            "success": success
        }


if __name__ == '__main__':
    input_map = {
        "java": ("java", java_code),
        "c": ("c", c_code),
        "cpp": ("cpp", c_code),
        "cs": ("c_sharp", cs_code),
        "js": ("javascript", js_code),
        "python": ("python", python_code),
        "php": ("php", php_code),
        "ruby": ("ruby", ruby_code),
        "go": ("go", go_code),
    }
    code_directory = os.path.realpath(os.path.join(os.path.realpath(__file__), '../../../..'))
    parser_path = os.path.join(code_directory, "parser/languages.so")
    for lang in ["c", "cpp", "java", "python", "php", "ruby", "js", "go", "cs"]:
        lang, code = input_map[lang]
        var_renamer = VarRenamer(
            parser_path, lang
        )
        print(lang)
        code, meta = var_renamer.transform_code(code)
        print(re.sub("[ \t\n]+", " ", code))
        print(meta)
        print("=" * 150)
