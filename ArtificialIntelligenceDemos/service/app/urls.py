from django.urls import path
from .views import transform_csv, sklearn_dt, scikitty_tree_boosting, scikitty_linear_regression, scikitty_logistic_regression, scikitty_breast_cancer, scikitty_california_housing

urlpatterns = [
    path('transformCSV/', transform_csv, name='transform_csv'),
    path('sklearnDT/', sklearn_dt, name='sklearn_dt'),
    path('scikitty_tree_boosting/', scikitty_tree_boosting, name='scikitty_tree_boosting'),
    path('scikitty_linear_regression/', scikitty_linear_regression, name='scikitty_linear_regression'),
    path('scikitty_logistic_regression/', scikitty_logistic_regression, name='scikitty_logistic_regression'),
    path('scikitty_breast_cancer/', scikitty_breast_cancer, name='scikitty_breast_cancer'),
    path('scikitty_california_housing/', scikitty_california_housing, name='scikitty_california_housing')
]
