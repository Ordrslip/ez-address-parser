import pickle
import warnings
from pathlib import Path

import sklearn_crfsuite

from .features import transform, load_data
from .utils import tokenize


class AddressParser:
    def __init__(self, use_pretrained=True):
        if use_pretrained:
            here = Path(__file__).parent.absolute()
            with open(here / "model", "rb") as f:
                self.crf = pickle.load(f)
        else:
            self.crf = None
            warnings.warn("Run train() to learn from some existing labelled data")

    def parse(self, address):
        if not self.crf:
            raise RuntimeError("Model is not loaded")

        tokens = tokenize(address)
        labels = self.crf.predict([transform(address)])[0]
        parsed = list(zip(tokens, labels))
        lineone = ["StreetNumber", "StreetName", "StreetType", "StreetDirection"]
        linetwo = ["Unit", "UnitNumber", "Building", "BuildingNumber"]
        parsed_address = {
            "address_line_1": ' '.join([y[0] for x, y in enumerate(parsed) if y[1] in lineone]),
            "address_line_2": ' '.join([y[0] for x, y in enumerate(parsed) if y[1] in linetwo]),
            "city": [y[0] for y in parsed if y[1] == "Municipality"][0],
            "state": [y[0] for y in parsed if y[1] == "Province"][0],
            "postal_code": ' '.join([y[0] for x, y in enumerate(parsed) if y[1] == 'PostalCode']),
        }
        return parsed_address

    def train(self, data, c1=0.1, c2=0.1):
        X, y = load_data(data)

        crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            max_iterations=100,
            all_possible_transitions=True,
            c1=c1,
            c2=c2,
        )
        crf.fit(X, y)

        self.crf = crf
