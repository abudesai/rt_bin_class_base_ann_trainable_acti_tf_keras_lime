import numpy as np, pandas as pd
import json
import os, sys
import time
from lime import lime_tabular

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
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)

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

    def _predict_proba(self, X):
        model = self._get_model()
        preds = model.predict(X)
        preds_arr = np.concatenate([1 - preds, preds], axis=1)
        return preds_arr

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
        pred_X, ids = proc_data["X"].astype(np.float), proc_data["ids"]

        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)
        feature_names = list(pred_X.columns)

        print(f"Generating local explanations for {pred_X.shape[0]} sample(s).")
        explainer = lime_tabular.LimeTabularExplainer(
            model.train_X,
            mode="classification",
            class_names=class_names,
            feature_names=feature_names,
        )

        model = self._get_model()
        explanations = []
        for i, row in pred_X.iterrows():

            explanation = explainer.explain_instance(
                row, model.predict_proba, top_labels=len(class_names)
            )
            pred_class = class_names[int(explanation.predict_proba.argmax())]
            class_prob = explanation.predict_proba.max()
            other_class = (
                class_names[0] if class_names[1] == pred_class else class_names[1]
            )
            probabilities = {
                other_class: np.round(1 - class_prob, 5),
                pred_class: np.round(class_prob, 5),
            }

            sample_expl_dict = {}
            sample_expl_dict["explanations_per_class"] = {}
            for j, c in enumerate(class_names):
                class_exp_dict = {}
                class_exp_dict["class_prob"] = round(
                    float(explanation.predict_proba[j]), 5
                )
                class_exp_dict["intercept"] = np.round(explanation.intercept[j], 5)
                feature_impacts = {}
                for feature_idx, feature_impact in explanation.local_exp[j]:
                    feature_impacts[feature_names[feature_idx]] = np.round(
                        feature_impact, 5
                    )

                class_exp_dict["feature_impacts"] = feature_impacts
                sample_expl_dict["explanations_per_class"][str(c)] = class_exp_dict

            explanations.append(
                {
                    self.id_field_name: ids[i],
                    "label": pred_class,
                    "probabilities": probabilities,
                    "explanations": sample_expl_dict["explanations_per_class"],
                }
            )
        explanations = {"predictions": explanations}
        explanations = json.dumps(explanations, cls=utils.NpEncoder, indent=2)
        return explanations
