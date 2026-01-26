import pickle
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression

from buzzer import BuzzerParameters, Buzzer


class ThresholdParameters(BuzzerParameters):
    """
    Parameters for a threshold-based buzzer.

    Parameters include:
    - threshold: minimum confidence required to buzz
    - cutoff: minimum question length required to allow buzzing
    """

    def __init__(self, customized_params: Optional[List[Tuple[str, type, Any, str]]] = None) -> None:
        super().__init__()
        self.name: str = "threshold_buzzer"

        if customized_params:
            self.params += customized_params
        else:
            threshold_params = [
                ("threshold", float, 0.5, "If guesser has confidence over this value, we buzz"),
                ("cutoff", int, 25, "Don't buzz if the question is shorter than this"),
            ]
            self.params += threshold_params


class ThresholdBuzzer(Buzzer):
    """
    A buzzer that decides to buzz based on a fixed confidence threshold
    and a minimum question length cutoff.
    """

    def __init__(self, filename: str, run_length: int) -> None:
        """
        Initialize the threshold buzzer.

        Args:
            filename: Base filename for saving/loading parameters.
            run_length: Number of words to reveal per step.
        """
        super().__init__(filename=filename, run_length=run_length, num_guesses=1)

        self.threshold: float = 0.85
        self.cutoff: int = 125

        # Dummy classifier to satisfy inherited interfaces
        self._classifier: LogisticRegression = LogisticRegression()
        self._classifier.coef_ = [[]]

        self.threshold_feature: Optional[str] = None
        
    def initialize_threshold(self, threshold: float, cutoff: int) -> None:
        self.threshold = threshold
        self.cutoff = cutoff

    @staticmethod
    def threshold_predict(
        question: str,
        confidence: float,
        confidence_threshold: float,
        length_threshold: int,
    ) -> bool:
        """
        Decide whether to buzz based on confidence and question length.

        Args:
            question: Current question text.
            confidence: Confidence score of the guess.
            confidence_threshold: Minimum confidence required to buzz.
            length_threshold: Minimum question length required.

        Returns:
            True if the system should buzz, False otherwise.
        """
        # HW TODO: check the two conditions then return True/False
        return False

    def set_confidence_feature(self, feat_vec: Iterable[str]) -> None:
        """
        Identify and store the feature name corresponding to confidence.

        Args:
            feat_vec: Iterable of feature names.

        Raises:
            AssertionError: If there is not exactly one confidence feature.
        """
        assert sum(1 for x in feat_vec if x.endswith("confidence")) == 1, \
            "Too many confidences"

        for feat_name in feat_vec:
            if feat_name.endswith("confidence"):
                self.threshold_feature = feat_name

    def load(self) -> None:
        """
        Load threshold and cutoff parameters from disk.
        """
        try:
            with open(f"{self.filename}.pkl", "rb") as infile:
                self.threshold, self.cutoff = pickle.load(infile)
        except (IOError, EOFError):
            print("Could not load threshold buzzer parameters through .pkl, using params from arg parse")

    def save(self) -> None:
        """
        Save threshold and cutoff parameters to disk.
        """
        with open(f"{self.filename}.pkl", "wb") as outfile:
            pickle.dump((self.threshold, self.cutoff), outfile)

    def train(self, questions: Iterable[str]) -> None:
        """
        Train the threshold buzzer parameters. Hint: You may want to
        train with data/qanta.buzztrain.json.gz to find best threshold and length cutoff.
        """

        assert len(self._features) == len(self._questions), "Features not built.  Did you run build_features?"
        self.set_confidence_feature(self)

        lengths = set(len(x) for x in self._questions)
        confidences = set(x[self.threhshold_feature] for x in self._features)
        results = defaultdict(Counter)



        best_length = -1
        best_threshold = float("-inf")
        self.threshold = best_threshold  
        self.cutoff = best_length

    def single_predict(
        self,
        question: str,
        run: str,
        guess_history: Any,
    ) -> Iterator[Tuple[bool, Dict[str, Any], Dict[str, Any]]]:
        """
        Predict whether to buzz for a single question run.

        Args:
            question: Full question text.
            run: Current revealed portion of the question.
            guess_history: Prior guesses (unused).

        Yields:
            Tuples of (buzz_decision, guess, metadata).
        """
        assert len(self._guessers) == 1, "Can only handle a single guesser"

        guesser = self._guessers[max(self._guessers)]
        guesses = guesser(run)

        assert len(guesses) == 1, "Can only handle one guess"
        guess = guesses[0]

        buzz = self.threshold_predict(
            run,
            guess["confidence"],
            self.threshold,
            self.cutoff,
        )

        yield buzz, guess, {}

    def predict(
        self,
        questions: Iterable[str],
    ) -> Tuple[List[int], None, Any, Any, Any]:
        """
        Predict buzz decisions for a batch of questions.

        Args:
            questions: Iterable of questions (unused directly).

        Returns:
            Tuple containing predictions and associated metadata.
        """
        assert len(self._features) == len(self._questions), \
            "Features not built. Did you run build_features?"

        predictions: List[int] = []

        # JBG's TODO: threshold_feature is referenced inconsistently here
        self.set_confidence_feature(self._features[0].keys())
        assert self.threshold_feature is not None

        for question, feat_vec in zip(self._runs, self._features):
            # HW TODO: now we only use confidence threshold, change this to use threshold_predict to cover both conditions
            if feat_vec[self.threshold_feature] > self.threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions, None, self._features, self._correct, self._metadata
