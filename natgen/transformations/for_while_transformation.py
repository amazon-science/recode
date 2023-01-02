import copy
import os
import re
from typing import Union, Tuple

import numpy as np

from .language_processors import (
    JavaAndCPPProcessor,
    CSharpProcessor,
    PythonProcessor,
    JavascriptProcessor,
    PhpProcessor
)
from .transformation_base import TransformationBase

java_code = """
import a.b;
// just for test
class A{
    /*
     * test test
     */
    int foo(int n){
        int res = 0;
        for(int i = 0; i < n; i++) {
            int j = 0;
            while (j < i){
                res += j; 
            }
        }
        return res;
    }
}
"""
python_code = """
from a import b as c
# just for test
@ decorator
def foo(n):
    \"\"\" test
    test test
    \"\"\"
    res = 0
    for i in range(0, 19, 2):
        res += i
    i = 0
    while i in range(n):
        res += i
        i += 1
    return res
"""
c_code = """
    int foo(int n){
        int res = 0;
        for(int i = 0; i < n; i++) {
            int j = 0;
            while (j < i){
                res += j; 
            }
        }
        return res;
    }
"""
cs_code = """
int foo(int n){
        int res = 0, i = 0;
        while(i < n) {
            int j = 0;
            while (j < i){
                res += j; 
            }
        }
        return res;
    }
"""
js_code = """
import a.b;
// just for test
function foo(n) {
    /*
     * test test
     */
    let res = '';
    for(let i = 0; i < 10; i++){
        res += i.toString();
        res += '<br>';
    } 
    while ( i < 10 ) { 
        res += 'bk'; 
    }
    return res;
}
"""
ruby_code = """
    for i in 0..5 do
        puts "Value of local variable is #{i}"
        if false then
            puts "False printed"
            while i == 10 do
                print i;
            end
            i = u + 8
        end
    end
    """
go_code = """
    func main() {
        sum := 0;
        i := 0;
        for ; i < 10;  {
            sum += i;
        }
        i++;
        fmt.Println(sum);
    }
    """
php_code = """
<?php 
for ($x = 0; $x <= 10; $x++) {
    echo "The number is: $x <br>";
}
$x = 0 ; 
while ( $x <= 10 ) { 
    echo "The number is:  $x  <br> "; 
    $x++; 
} 
?> 
"""

test_code = {"python": python_code, "java": java_code, "javascript": js_code}


processor_function = {
    "java": [JavaAndCPPProcessor.for_to_while_random, JavaAndCPPProcessor.while_to_for_random],
    "c": [JavaAndCPPProcessor.for_to_while_random, JavaAndCPPProcessor.while_to_for_random],
    "cpp": [JavaAndCPPProcessor.for_to_while_random, JavaAndCPPProcessor.while_to_for_random],
    "c_sharp": [CSharpProcessor.for_to_while_random, CSharpProcessor.while_to_for_random],
    "python": [PythonProcessor.for_to_while_random, PythonProcessor.while_to_for_random],
    "javascript": [JavascriptProcessor.for_to_while_random, JavascriptProcessor.while_to_for_random],
    "go": [CSharpProcessor.for_to_while_random, CSharpProcessor.while_to_for_random],
    "php": [PhpProcessor.for_to_while_random, PhpProcessor.while_to_for_random],
    "ruby": [CSharpProcessor.for_to_while_random, CSharpProcessor.while_to_for_random],
}

# need to reimplement first version for other languages!
processor_function_first = {
    "java": [JavaAndCPPProcessor.for_to_while_random, JavaAndCPPProcessor.while_to_for_random],
    "c": [JavaAndCPPProcessor.for_to_while_random, JavaAndCPPProcessor.while_to_for_random],
    "cpp": [JavaAndCPPProcessor.for_to_while_random, JavaAndCPPProcessor.while_to_for_random],
    "c_sharp": [CSharpProcessor.for_to_while_random, CSharpProcessor.while_to_for_random],
    "python": [PythonProcessor.for_to_while_first, PythonProcessor.while_to_for_first],
    "javascript": [JavascriptProcessor.for_to_while_random, JavascriptProcessor.while_to_for_random],
    "go": [CSharpProcessor.for_to_while_random, CSharpProcessor.while_to_for_random],
    "php": [PhpProcessor.for_to_while_random, PhpProcessor.while_to_for_random],
    "ruby": [CSharpProcessor.for_to_while_random, CSharpProcessor.while_to_for_random],
}


class ForWhileTransformer(TransformationBase):
    """
    Change the `for` loops with `while` loops and vice versa.
    """

    def __init__(self, parser_path, language):
        super(ForWhileTransformer, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        self.transformations = processor_function[language]
        self.transformations_first_half = processor_function_first[language]
        processor_map = {
            "java": self.get_tokens_with_node_type,
            "c": self.get_tokens_with_node_type,
            "cpp": self.get_tokens_with_node_type,
            "c_sharp": self.get_tokens_with_node_type,
            "javascript": JavascriptProcessor.get_tokens,
            "python": PythonProcessor.get_tokens,
            "php": PhpProcessor.get_tokens,
            "ruby": self.get_tokens_with_node_type,
            "go": self.get_tokens_with_node_type,
        }
        self.final_processor = processor_map[self.language]

    def transform_code(
            self,
            code: Union[str, bytes],
            first_half=False
    ) -> Tuple[str, object]:
        ##### just for test #####
        # code = test_code[self.language]
        ##### just for test #####
        # mainly need to fix get_tokens_replace_while; get_tokens_replace_for

        success = False
        if not first_half:
            transform_functions = copy.deepcopy(self.transformations)
        else:
            transform_functions = copy.deepcopy(self.transformations_first_half)
        while not success and len(transform_functions) > 0:
            function = np.random.choice(transform_functions)
            transform_functions.remove(function)
            modified_root, modified_code, success = function(code, self)
            if success:
                code = modified_code
        root_node = self.parse_code(
            code=code
        )
        return_values = self.final_processor(
            code=code.encode(),
            root=root_node
        )
        if isinstance(return_values, tuple):
            tokens, types = return_values
        else:
            tokens, types = return_values, None
        code = " ".join(tokens)
        code = code.replace("\n", " NEWLINE ")
        # code = re.sub("[ \t\n]+", " ", code)
        return code, \
               {
                   "types": types,
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
    code_directory = os.path.realpath(os.path.join(os.path.realpath(__file__), '../../../../'))
    parser_path = "my-languages.so"
    parser_path = os.path.join(code_directory, parser_path)
    # for lang in ["java", "python", "js", "c", "cpp", "php", "go", "ruby", "cs"]:
    for lang in ["java", "python", "js"]:
        lang, code = input_map[lang]
        for_while_transformer = ForWhileTransformer(parser_path, lang)
        print(lang, end="\t")
        code, meta = for_while_transformer.transform_code(code)
        if lang == "python":
            code = PythonProcessor.beautify_python_code(code.split())
        print(code)
        print(meta["success"])
        print("=" * 150)