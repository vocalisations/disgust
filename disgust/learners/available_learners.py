from disgust.learners.train_and_predict import train_and_predict_using_xgboost, \
    train_and_predict_using_rf, train_and_predict_using_grid_search_rf, \
    train_and_predict_using_grid_search_xgboost, train_and_predict_using_pycaret

available_learners = {'xgboost': train_and_predict_using_xgboost,
                      'rf': train_and_predict_using_rf,
                      'pycaret': train_and_predict_using_pycaret,
                      'grid_search_rf': train_and_predict_using_grid_search_rf,
                      'grid_search_xgboost': train_and_predict_using_grid_search_xgboost,
                      }
