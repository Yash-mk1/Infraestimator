"""
scanner/urls.py — MODIFIED
"""

from django.urls import path
from . import views

urlpatterns = [
    path('',                views.home,           name='home'),
    path('analyse/',        views.analyse,        name='analyse'),
    path('scan/',           views.scan,           name='scan'),
    path('result/',         views.result,         name='result'),
    path('history/',        views.history,        name='history'),
    path('history/<int:scan_id>/', views.history_detail, name='history_detail'),
    path('how-it-works/',   views.how_it_works,   name='how_it_works'),
    path('technology/',     views.technology,     name='technology'),
    path('docs/',           views.docs,           name='docs'),
    path('signup/',         views.signup_view,    name='signup'),
    path('login/',          views.login_view,     name='login'),
    path('logout/',         views.logout_view,    name='logout'),
]