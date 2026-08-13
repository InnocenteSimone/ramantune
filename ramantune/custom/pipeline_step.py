from ramanspy.preprocessing import PreprocessingStep

class RamanPipelineStep(PreprocessingStep):
    """
    Wrap ``ramanspy.preprocessing.PreprocessingStep`` for project usage.

    Notes
    -----
    Custom preprocessing steps in this project should inherit from this class
    instead of directly inheriting from ``PreprocessingStep``.
    """
    def __init__(self, func, **kwargs):
        """Initialize a Raman pipeline preprocessing step.

        Parameters
        ----------
        func : callable
            Callable implementing preprocessing logic.
        **kwargs : dict
            Additional keyword arguments forwarded to ``PreprocessingStep``.
        """
        super().__init__(func , **kwargs)