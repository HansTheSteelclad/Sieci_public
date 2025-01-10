from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("Oferty_pracy/", views.oferty_pracy_disp, name="Oferty_pracy"),
    path("Analiza_ofert/", views.analiza_wykresy, name="Analiza"),
    path("Analiza_pensji/", views.analiza_pensje, name="Analiza_pensje"),
    path("Analiza_ilosci_ofert/", views.analiza_ilosc, name="Analiza_ilosc_ofert"),
    path("Analiza_opisowa/", views.analiza_opisowa, name="Analiza_oferty"),
    path("Analiza_ilosci_ofert_w_czasie/", views.analiza_ilosc_2, name="Analiza_ilosc_ofert_2"),
    path("Error/", views.analiza_wykresy, name="Error"),
    path("Error/", views.analiza_pensje, name="Error"),
    path("Error/", views.analiza_ilosc, name="Error"),
    path("Error/", views.analiza_opisowa, name="Error"),
]


