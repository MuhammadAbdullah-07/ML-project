import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj,evaluate_model



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initaite_model_trainer(self,train_array,test_array):
        try:
            logging.info("split Train and test input data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "LinearRegression":LinearRegression(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "CatBoostRegressor":CatBoostRegressor(verbose=False)
            }

            params={
                "LinearRegression" :  {},
                "KNeighborsRegressor": {'n_neighbors':[1,2,3,4,5],
                                        'weights' : ['uniform','distance'],
                                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                                        },

                "AdaBoostRegressor" :   {'learning_rate':[.1,.01,.05,.001],
                                        'n_estimators':[8,16,32,64,128,256],
                                        },

                "GradientBoostingRegressor" :   {'learning_rate':[.1,.01,.05,.001],
                                        'n_estimators':[8,16,32,64,128,256],
                                        'loss' : ['squared_error', 'absolute_error', 'huber', 'quantile']
                                        },
                "RandomForestRegressor" : {'n_estimators': [8,16,32,64,128,256],
                                        'criterion' : ['squared_error', 'absolute_error', 'poisson'],
                                        'max_features' : ['sqrt', 'log2'],
                                        },

                "CatBoostRegressor" :   {'depth':[6,8,10],
                                        'learning_rate':[.1,.01,.05,.001],
                                        'iterations' : [30,50,100]
                                        },
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            ## To get the best model score
            best_model_score= max(sorted(model_report.values()))

            ## To get the best model name

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found ")
            logging.info("finding Best Model")    

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted=best_model.predict(X_test)

            score= r2_score(y_test,predicted)
            ## Printing the best Model key & Value
            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score}") 
            print(f"Best Model Found: {best_model_name},{best_model_score}")
            return  score

        except Exception as e:
            raise CustomException(e,sys)
        