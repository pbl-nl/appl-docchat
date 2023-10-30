import os

# TODO content iterator should only yield documents that are not processed yet
class ContentIterator():
    def __init__(self, content_path):
        self.content_path = content_path

    def __iter__(self):
        if os.path.isfile(self.content_path):
            yield self.content_path
        else:
            for file in os.listdir(self.content_path):
                # if file.endswith(".pdf"):
                yield os.path.join(self.content_path, file)
