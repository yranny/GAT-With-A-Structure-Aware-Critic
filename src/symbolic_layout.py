class SymbolicLayout:
    """
    A simple symbolic program parser for VQA reasoning over forest graphs.
    """

    def __init__(self):
        pass

    def parse(self, question):
        if "closest" in question:
            return ["Find", "Filter(species=birch)", "Nearest"]
        elif "tallest" in question:
            return ["Find", "Argmax(height)"]
        elif "species of center tree" in question:
            return ["Find", "CenterNode", "Query(species)"]
        else:
            return ["Find", "Query(species)"]

