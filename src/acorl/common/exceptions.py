class InputAdaptionComputationError(Exception):
    """
    Exception raised when an error occurs during the computation of the relevant input.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
