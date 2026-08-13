from sklearn.pipeline import Pipeline


class PipelineBuilder:
    """
    Utility builder for assembling scikit-learn pipelines.
    Collect and assemble ordered pipeline steps.

    Attributes
    ----------
    steps : list of tuple[str, object]
        Ordered ``(name, transformer_or_estimator)`` pairs used to create
        a scikit-learn ``Pipeline``.
    """
    def __init__(self):
        """Initialize an empty pipeline step collection."""
        self.steps = []

    def insert_step(self, index, name, step):
        """
        Insert a step at a specific position.

        Parameters
        ----------
        index : int
            Insertion index in the internal step list.
        name : str
            Unique pipeline step name.
        step : object
            Scikit-learn compatible transformer or estimator.
        """
        self.steps.insert(index, (name, step))

    def add_step(self, name, step):
        """
        Add a step to the end of the pipeline.

        Parameters
        ----------
        name : str
            Unique pipeline step name.
        step : object
            Scikit-learn compatible transformer or estimator.
        """
        self.steps.append((name, step))


    def build(self):
        """
        Build a scikit-learn ``Pipeline`` from collected steps.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Pipeline instance using the current ordered steps.
        """
        return Pipeline(self.steps)

    def reset(self):
        """Clear all currently collected pipeline steps."""
        self.steps = []
