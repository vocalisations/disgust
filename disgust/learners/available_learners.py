from disgust.learners.train_and_predict import train_and_predict_using_xgboost, \
    train_and_predict_using_simple_default_rf, train_and_predict_using_grid_search_rf

available_learners = {'xgboost': train_and_predict_using_xgboost,
                      'rf': train_and_predict_using_simple_default_rf,
                      'grid_search_rf': train_and_predict_using_grid_search_rf,
                      }
