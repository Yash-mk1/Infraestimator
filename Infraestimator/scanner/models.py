"""
scanner/models.py — MODIFIED
Added user FK so each scan is tied to the logged-in user.
"""

from django.db import models
from django.contrib.auth.models import User


class ScanResult(models.Model):
    # ── Owner ─────────────────────────────────────────────────────────────────
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True, blank=True,       # null = scan by anonymous user
        related_name='scans'
    )

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at = models.DateTimeField(auto_now_add=True)

    # ── Input ─────────────────────────────────────────────────────────────────
    material = models.CharField(max_length=32, default='general')

    # ── Output images ─────────────────────────────────────────────────────────
    original_image  = models.ImageField(upload_to='results/originals/', blank=True, null=True)
    annotated_image = models.ImageField(upload_to='results/annotated/', blank=True, null=True)
    heatmap_image   = models.ImageField(upload_to='results/heatmaps/',  blank=True, null=True)

    # ── Health scores ─────────────────────────────────────────────────────────
    health_score         = models.FloatField(default=0.0)
    condition            = models.CharField(max_length=32, default='Unknown')
    estimated_life_years = models.FloatField(default=0.0)
    critical             = models.BooleanField(default=False)

    # ── Sub-scores ────────────────────────────────────────────────────────────
    crack_score   = models.FloatField(default=0.0)
    seep_score    = models.FloatField(default=0.0)
    surface_score = models.FloatField(default=0.0)

    # ── Detection stats ───────────────────────────────────────────────────────
    crack_area_pct      = models.FloatField(default=0.0)
    seep_area_pct       = models.FloatField(default=0.0)
    num_crack_regions   = models.IntegerField(default=0)
    num_seep_regions    = models.IntegerField(default=0)
    largest_crack_mm_eq = models.FloatField(default=0.0)

    # ── AI metadata ───────────────────────────────────────────────────────────
    ai_used       = models.BooleanField(default=False)
    ai_confidence = models.FloatField(default=0.0)

    # ── Text fields ───────────────────────────────────────────────────────────
    warnings        = models.TextField(blank=True, default='')
    recommendations = models.TextField(blank=True, default='')

    # ── Stored image URLs (for quick retrieval) ───────────────────────────────
    original_url  = models.CharField(max_length=300, blank=True, default='')
    annotated_url = models.CharField(max_length=300, blank=True, default='')
    heatmap_url   = models.CharField(max_length=300, blank=True, default='')

    class Meta:
        ordering     = ['-created_at']
        verbose_name = 'Scan Result'

    def __str__(self):
        user_str = self.user.username if self.user else 'anonymous'
        return f"Scan #{self.pk} | {user_str} | {self.material} | {self.health_score}/10"

    def warnings_list(self):
        return [w for w in self.warnings.split('\n') if w.strip()]

    def recommendations_list(self):
        return [r for r in self.recommendations.split('\n') if r.strip()]

    def score_class(self):
        s = self.health_score
        if   s >= 8.5: return 'excellent'
        elif s >= 7.0: return 'good'
        elif s >= 5.5: return 'fair'
        elif s >= 4.0: return 'poor'
        elif s >= 2.5: return 'critical'
        else:          return 'failure'