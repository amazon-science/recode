from typing import Union, Tuple

from . import TransformationBase


class DemoTransformation(TransformationBase):
    def __init__(self, parser, language):
        super(DemoTransformation, self).__init__(
            parser_path=parser,
            language=language,
        )

    def transform_code(
            self,
            code: Union[str, bytes]
    ) -> Tuple[str, object]:
        root_node = self.parse_code(
            code=code
        )
        tokens, types = self.get_tokens_with_node_type(
            code=code.encode(),
            root=root_node
        )
        return " ".join(tokens), types
