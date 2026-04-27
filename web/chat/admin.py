from django.contrib import admin
from chat.models import Session, Message, CredibilityCache


@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    list_display = ("id", "topic_label", "message_count", "created_at")
    list_filter = ("created_at",)
    search_fields = ("topic_label",)
    readonly_fields = ("id", "collection_name", "created_at")


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "role", "short_content", "is_summary", "created_at")
    list_filter = ("role", "is_summary")
    raw_id_fields = ("session",)

    def short_content(self, obj):
        return obj.content[:80] + "..." if len(obj.content) > 80 else obj.content
    short_content.short_description = "Content"


@admin.register(CredibilityCache)
class CredibilityCacheAdmin(admin.ModelAdmin):
    list_display = ("url", "total", "scored_at")
    list_filter = ("scored_at",)
