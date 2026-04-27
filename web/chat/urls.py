"""URL routing for the chat app."""

from django.urls import path
from chat import views

urlpatterns = [
    # Page
    path("", views.index, name="index"),

    # SSE streaming
    path("api/chat", views.chat_stream, name="chat_stream"),

    # Session CRUD
    path("api/session/new", views.session_create, name="session_create"),
    path("api/sessions", views.session_list, name="session_list"),
    path("api/session/<str:session_id>/delete", views.session_delete, name="session_delete"),
    path("api/session/<str:session_id>/history", views.session_history, name="session_history"),
]
