import copy
from typing import Union, Tuple

import numpy as np
import os

from .language_processors import (
    JavaAndCPPProcessor,
    PythonProcessor,
    JavascriptProcessor,
    PhpProcessor
)
from . import TransformationBase

processor_function = {
    "java": [JavaAndCPPProcessor.incre_decre_removal, JavaAndCPPProcessor.ternary_removal],
    "c": [JavaAndCPPProcessor.incre_decre_removal, JavaAndCPPProcessor.conditional_removal],
    "cpp": [JavaAndCPPProcessor.incre_decre_removal, JavaAndCPPProcessor.conditional_removal],
    "c_sharp": [JavaAndCPPProcessor.incre_decre_removal, JavaAndCPPProcessor.conditional_removal]
}


class ConfusionRemover(TransformationBase):
    """
    Change the `for` loops with `while` loops and vice versa.
    """

    def __init__(self, parser_path, language):
        super(ConfusionRemover, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        if language in processor_function:
            self.transformations = processor_function[language]
        else:
            self.transformations = []
        processor_map = {
            "java": self.get_tokens_with_node_type,  # yes
            "c": self.get_tokens_with_node_type,  # yes
            "cpp": self.get_tokens_with_node_type,  # yes
            "c_sharp": self.get_tokens_with_node_type,  # yes
            "javascript": JavascriptProcessor.get_tokens,  # yes
            "python": PythonProcessor.get_tokens,  # no
            "php": PhpProcessor.get_tokens,  # yes
            "ruby": self.get_tokens_with_node_type,  # yes
            "go": self.get_tokens_with_node_type,  # no
        }
        self.final_processor = processor_map[self.language]

    def transform_code(
            self,
            code: Union[str, bytes],
    ) -> Tuple[str, object]:
        success = False
        transform_functions = copy.deepcopy(self.transformations)
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
        return " ".join(tokens), \
               {
                   "types": types,
                   "success": success
               }


if __name__ == '__main__':
    java_code = """
    class A{
        int foo(int[] nums, int lower, upper){
            for(int i = 0; i < n; i++) {
                static long start = i == 0 ? lower : (long)nums[i - 1] + 1;
                static long end = i == nums.length ? upper : (long)nums[i] - 1;
                start = (lower + nums[j] > upper) ? lower + nums[j] : upper;
                lower += 1;
                lower = upper++;
                lower = ++upper;
            }
            return i == end ? -1 : start;
        }
    }
    """
    python_code = """def foo(n):
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
            int res;
            for(int i = 0; i < n; i++) {
                int j = 0;
                if (j == 0) { i = j; }
                j = (j == 0) ? (i + j) : i - j;
                int i = (i == 0) ? (i + j) : i - j;
                }
            i = j ++;
            j = i--;
            j = -- i;
            j = ++i;
            return i == 0 ? -1 : j;
            }
            
    """
    cs_code = """int foo(int n){
            x = n++;
            n = x--;
            x = ++n;
            n = ++x;
            return x != 0.0 ? Math.Sin(x) / x : 1.0;
        }
    """
    js_code = """function foo(n) {
        let res = '';
        for(let i = 0; i < 10; i++){
            res += i.toString();
            res += '<br>';
        } 
        while ( i < 10 ; ) { 
            res += 'bk'; 
        }
        return res;
    }
    """
    ruby_code = """
        for i in 0..5
           puts "Value of local variable is #{i}"
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
    for lang in ["c", "cpp", "java", "cs", "python", "php", "go", "ruby"]:
        # lang = "php"
        lang, code = input_map[lang]
        confusion_remover = ConfusionRemover(
            parser_path, lang
        )
        print(lang)
        code, types = confusion_remover.transform_code(code)
        print(types["success"])
        print("=" * 100)
