"""
scanner/admin.py — MODIFIED
"""

from django.contrib import admin
from .models import ScanResult


@admin.register(ScanResult)
class ScanResultAdmin(admin.ModelAdmin):
    list_display  = [
        'pk', 'user', 'material', 'health_score',
        'condition', 'estimated_life_years', 'ai_used', 'created_at'
    ]
    list_filter   = ['condition', 'material', 'ai_used', 'critical']
    search_fields = ['user__username', 'material', 'condition']
    readonly_fields = [
        'created_at', 'original_url', 'annotated_url', 'heatmap_url'
    ]
    ordering      = ['-created_at']