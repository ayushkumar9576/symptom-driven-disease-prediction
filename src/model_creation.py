from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
import logging

logger = logging.getLogger(__name__)

class RandomForestModel:
    def __init__(self, path: str | Path = "model/predict_model.pkl")->None:
        self._model_path = Path(path)
        self._model: RandomForestClassifier | None = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series)-> None:
        logger.info("Training and optimizing the RandomForest Model")

        cv = StratifiedKFold(n_splits= 3, shuffle= True, random_state= 42)
        search = RandomizedSearchCV(
            estimator=RandomForestClassifier(bootstrap= True, class_weight= "balanced", random_state= 42, n_jobs= 1),
            param_distributions={"n_estimators": [300, 500, 700], "max_depth": [10, 15, 20, 25, 30], "min_samples_split": [2, 5, 10, 20], "min_samples_leaf": [1, 2, 4, 8, 16], "max_features": ["sqrt", "log2", 0.3, 0.5], "max_samples": [0.6, 0.8, 1.0]},
            n_iter= 75, scoring="f1_macro", cv= cv, random_state= 42, n_jobs= -1, refit= True, verbose= 1, return_train_score= True)
        
        search.fit(X_train, y_train)
        
        self._model = search.best_estimator_
        self._best_parameters = search.best_params_
        self._best_cv_score = search.best_score_
        self._cv_results = search.cv_results_

        best_index = search.best_index_
        train_score = search.cv_results_["mean_train_score"][best_index]
        cv_score = search.cv_results_["mean_test_score"][best_index]
        self._train_f1 = train_score
        self._cv_f1 = cv_score

        if train_score - cv_score > 0.10:
            logger.warning(f"Warning: Possible overfitting detected (Train={train_score:.4f}, CV={cv_score:.4f})")

        val_predict = self._model.predict(X_val)
        val_f1 = f1_score(y_val, val_predict, average="macro")
        self._validation_f1 = val_f1

        self._feature_importances = (pd.Series(self._model.feature_importances_, index=X_train.columns).sort_values(ascending=False))

        logger.info(f"Training F1 Score: {train_score:.4f}")
        logger.info(f"CV F1 Score: {cv_score:.4f}")
        logger.info(f"Validation F1 Score: {val_f1:.4f}")

        logger.info("\nTop 15 Feature Importances")
        logger.info(self._feature_importances.head(15))
        
        logger.info("Model training and hyperparameter optimization completed")
        logger.info("Saving the Prediction and full model to 'model/'")

        joblib.dump(self._model, self._model_path)
        joblib.dump(self, "model/full_model.pkl")
        logger.info("Model Saving completed")

    
    def predict(self, X: pd.DataFrame)-> int:
        logger.info("Predicting The Disease for the Patient")
        return self._model.predict(X)[0]
    
    @classmethod
    def load(cls, path: str | Path = "model/predict_model.pkl")-> "RandomForestModel":
        logger.info("Loading the Prediction Model From Saved Models")
        instance = cls(path)
        instance._model = joblib.load(instance._model_path)
        return instance
    
    @classmethod
    def load_full(cls ,path: str | Path = "model/full_model.pkl")-> "RandomForestModel":
        logger.info("Loading the full model from the Saved Models")
        return joblib.load(path)
    
    def predict_proba(self, X: pd.DataFrame):
        return self._model.predict_proba(X)