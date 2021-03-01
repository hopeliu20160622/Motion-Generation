from typing import List


from pymo.parsers import BVHParser

# Need to be processed by length X 1 X embeddings
# Which corresponds to total_frames X 1 X embeddings 

class Preprocessor:

    def __init__(self, files: List, max_length: int) -> None:
        self.files = files
        self.parser = BVHParser()
        self.max_length = max_length

    def process(self):

        for file in self.files:
            bvh_parsed = self.parser.parse(file)
            bvh_length = bvh_parsed.values.shape[0]
