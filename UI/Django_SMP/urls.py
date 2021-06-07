from os import path
from django.urls import path
from . import views

app_name = 'smp'
urlpatterns = [
    path('', views.home, name="home"),
    path('google', views.google, name="google"),
    path('nifty', views.nifty, name="nifty"),
]