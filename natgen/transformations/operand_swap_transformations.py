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
    PhpProcessor,
    GoProcessor,
    RubyProcessor
)
# from transformations import TransformationBase
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
from typing import List

# just for test
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

processor_function = {
    "java": [JavaAndCPPProcessor.operand_swap],
    "c": [JavaAndCPPProcessor.operand_swap],
    "cpp": [JavaAndCPPProcessor.operand_swap],
    "c_sharp": [CSharpProcessor.operand_swap],
    "python": [PythonProcessor.operand_swap],
    "javascript": [JavascriptProcessor.operand_swap],
    "go": [GoProcessor.operand_swap],
    "php": [PhpProcessor.operand_swap],
    "ruby": [RubyProcessor.operand_swap],
}

# need to reimplement first version for other languages!
processor_function_first = {
    "java": [JavaAndCPPProcessor.operand_swap],
    "c": [JavaAndCPPProcessor.operand_swap],
    "cpp": [JavaAndCPPProcessor.operand_swap],
    "c_sharp": [CSharpProcessor.operand_swap],
    "python": [PythonProcessor.operand_swap_first],
    "javascript": [JavascriptProcessor.operand_swap],
    "go": [GoProcessor.operand_swap],
    "php": [PhpProcessor.operand_swap],
    "ruby": [RubyProcessor.operand_swap],
}

test_code = {"python": python_code, "java": java_code, "javascript": js_code}

class OperandSwap(TransformationBase):
    """
    Swapping Operand "a>b" becomes "b<a"
    """

    def __init__(self, parser_path, language):
        super(OperandSwap, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        self.transformations = processor_function[language]
        self.transformations_first = processor_function_first[language]
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

        success = False
        if first_half:
            transform_functions = copy.deepcopy(self.transformations_first)
        else:
            transform_functions = copy.deepcopy(self.transformations)
        while not success and len(transform_functions) > 0:
            function = np.random.choice(transform_functions)
            transform_functions.remove(function)
            modified_code, success = function(code, self)
            if success:
                code = modified_code
        root_node = self.parse_code(
            code=code
        )
        # import pdb; pdb.set_trace()
        return_values = self.final_processor(
            code=code.encode(),
            root=root_node
        )
        if isinstance(return_values, tuple):
            tokens, types = return_values
        else:
            tokens, types = return_values, None
        
        # code = re.sub("[ \t\n]+", " ", " ".join(tokens))
        code = " ".join(tokens)
        code = code.replace("\n", " NEWLINE ")
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
    parser_path = os.path.join(code_directory, "parser/languages.so")
    for lang in ["java", "python", "js", "c", "cpp", "php", "go", "ruby",
                 "cs"]:  # ["c", "cpp", "java", "cs", "python",
        # "php", "go", "ruby"]:
        # lang = "php"
        lang, code = input_map[lang]
        operandswap = OperandSwap(
            parser_path, lang
        )
        print(lang)
        # print("-" * 150)
        # print(code)
        # print("-" * 150)
        code, meta = operandswap.transform_code(code)
        print(meta["success"])
        print("=" * 150)
