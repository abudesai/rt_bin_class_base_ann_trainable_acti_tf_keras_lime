import numpy as np, pandas as pd
import json
import os, sys
import time
from interpret.blackbox import LimeTabular

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.classifier as classifier


# get model configuration parameters
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema):
        self.model_path = model_path
        self.preprocessor = None
        self.model = None
        self.data_schema = data_schema
        self.id_field_name = self.data_schema["inputDatasets"][
            "binaryClassificationBaseMainInput"
        ]["idField"]
        self.has_local_explanations = True
        self.MAX_LOCAL_EXPLANATIONS = 1

    def _get_preprocessor(self):
        if self.preprocessor is None:
            self.preprocessor = pipeline.load_preprocessor(self.model_path)
        return self.preprocessor

    def _get_model(self):
        if self.model is None:
            self.model = classifier.load_model(self.model_path)
        return self.model

    def _get_predictions(self, data):
        preprocessor = self._get_preprocessor()
        model = self._get_model()

        if preprocessor is None:
            raise Exception("No preprocessor found. Did you train first?")
        if model is None:
            raise Exception("No model found. Did you train first?")

        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data)
        # Grab input features for prediction
        pred_X = proc_data["X"].astype(np.float)
        # make predictions
        preds = model.predict(pred_X)
        return preds

    def predict_proba(self, data):

        preds = self._get_predictions(data)
        # get class names (labels)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)
        # get the name for the id field

        # return te prediction df with the id and class probability fields
        preds_df = data[[self.id_field_name]].copy()
        preds_df[class_names[0]] = 1 - preds
        preds_df[class_names[-1]] = preds

        return preds_df

    def predict(self, data):
        preds = self._get_predictions(data)

        # inverse transform the prediction probabilities to class labels
        pred_classes = pipeline.get_inverse_transform_on_preds(
            self.preprocessor, model_cfg, preds
        )
        # return te prediction df with the id and prediction fields
        preds_df = data[[self.id_field_name]].copy()
        preds_df["prediction"] = pred_classes

        return preds_df

    def _get_preds_array(self, X):
        model = self._get_model()
        preds = model.predict(X)
        preds_arr = np.concatenate([1 - preds, preds], axis=1)
        return preds_arr

    def predict_to_json(self, data):
        preds_df = self.predict_proba(data)
        class_names = preds_df.columns[1:]
        preds_df["__label"] = pd.DataFrame(
            preds_df[class_names], columns=class_names
        ).idxmax(axis=1)

        predictions_response = []
        for rec in preds_df.to_dict(orient="records"):
            pred_obj = {}
            pred_obj[self.id_field_name] = rec[self.id_field_name]
            pred_obj["label"] = str(rec["__label"])
            pred_obj["probabilities"] = {
                str(k): np.round(v, 5)
                for k, v in rec.items()
                if k not in [self.id_field_name, "__label"]
            }
            predictions_response.append(pred_obj)
        return predictions_response

    def explain_local(self, data):

        if data.shape[0] > self.MAX_LOCAL_EXPLANATIONS:
            msg = f"""Warning!
            Maximum {self.MAX_LOCAL_EXPLANATIONS} explanation(s) allowed at a time. 
            Given {data.shape[0]} samples. 
            Selecting top {self.MAX_LOCAL_EXPLANATIONS} sample(s) for explanations."""
            print(msg)

        preprocessor = self._get_preprocessor()
        model = self._get_model()
        data2 = data.head(self.MAX_LOCAL_EXPLANATIONS)
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        proc_data = preprocessor.transform(data2)
        pred_X = proc_data["X"].astype(np.float)

        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)
        feature_names = list(pred_X.columns)

        print(f"Generating local explanations for {pred_X.shape[0]} sample(s).")
        lime = LimeTabular(
            predict_fn=self._get_preds_array,
            data=pd.DataFrame(model.train_X, columns=feature_names),
            class_names=class_names,
            random_state=1,
        )

        # Get local explanations
        # start = time.time()
        lime_local = lime.explain_local(
            X=pred_X, y=None, name=f"{classifier.MODEL_NAME} local explanations"
        )

        # create the dataframe of local explanations to return
        ids = list(data2[self.id_field_name])
        explanations = []

        for i, sample_exp in enumerate(lime_local._internal_obj["specific"]):
            sample_expl_dict = {}
            # intercept
            sample_expl_dict["baseline"] = np.round(sample_exp["extra"]["scores"][0], 5)

            sample_expl_dict["feature_scores"] = {
                f: np.round(s, 5)
                for f, s in zip(sample_exp["names"], sample_exp["scores"])
            }
            sample_expl_dict[
                "comment_"
            ] = f"Explanations are w.r.t. class '{class_names[1]}'"

            class_prob = np.round(sample_exp["perf"]["predicted"], 5)
            probabilities = {
                class_names[0]: np.round(1 - class_prob, 5),
                class_names[1]: np.round(class_prob, 5),
            }
            explanations.append(
                {
                    self.id_field_name: ids[i],
                    "label": class_names[0] if class_prob < 0.5 else class_names[1],
                    "probabilities": probabilities,
                    "explanations": sample_expl_dict,
                }
            )
        explanations = {"predictions": explanations}
        explanations = json.dumps(explanations, cls=utils.NpEncoder, indent=2)
        return explanations
