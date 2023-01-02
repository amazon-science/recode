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
    RubyProcessor,
    GoProcessor
)
from .transformation_base import TransformationBase

processor_function = {
    "java": [JavaAndCPPProcessor.block_swap_java],
    "c": [JavaAndCPPProcessor.block_swap_c],
    "cpp": [JavaAndCPPProcessor.block_swap_c],
    "c_sharp": [CSharpProcessor.block_swap],
    "python": [PythonProcessor.block_swap],
    "javascript": [JavascriptProcessor.block_swap],
    "go": [GoProcessor.block_swap],
    "php": [PhpProcessor.block_swap],
    "ruby": [RubyProcessor.block_swap],
}


class BlockSwap(TransformationBase):
    """
    Swapping if_else block
    """

    def __init__(self, parser_path, language):
        super(BlockSwap, self).__init__(parser_path=parser_path, language=language)
        self.language = language
        self.transformations = processor_function[language]
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
        success = False
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
        return_values = self.final_processor(
            code=code.encode(),
            root=root_node
        )
        if isinstance(return_values, tuple):
            tokens, types = return_values
        else:
            tokens, types = return_values, None
        return re.sub("[ \t\n]+", " ", " ".join(tokens)), \
               {
                   "types": types,
                   "success": success
               }


if __name__ == '__main__':
    java_code = """
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
    python_code = """
    from typing import List
    
    def factorize(n: int) -> List[int]:
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
    code_directory = os.path.realpath(os.path.join(os.path.realpath(__file__), '../../../../'))
    parser_path = os.path.join(code_directory, "parser/languages.so")
    for lang in ["java", "python", "js", "c", "cpp", "php", "go", "ruby", "cs"]:
        lang, code = input_map[lang]
        no_transform = BlockSwap(
            "/home/saikatc/HDD_4TB/NatGen/parser/languages.so", lang
        )
        print(lang)
        code, meta = no_transform.transform_code(code)
        code = re.sub("[ \t\n]+", " ", code)
        if lang == "python":
            code = PythonProcessor.beautify_python_code(code.split())
        print(code)
        # print(re.sub("[ \t\n]+", " ", code))
        print(meta)
        print("=" * 150)
